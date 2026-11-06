"""Opt-in integration checks. The application verifies MySQL and Redis identity."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import pymysql


class PublicationConnection:
    def __init__(self, db, stage):
        self.db, self.stage, self.wrote_publication = db, stage, False

    def __getattr__(self, name):
        return getattr(self.db, name)

    def cursor(self):
        cursor = self.db.cursor()
        execute = cursor.execute

        def observe(sql, args=None):
            result = execute(sql, args)
            if "INSERT INTO rag_v2_publications" in sql:
                self.wrote_publication = True
            return result

        cursor.execute = observe
        return cursor

    def commit(self):
        if not self.wrote_publication or self.stage == "after":
            self.db.commit()
        if self.wrote_publication:
            raise pymysql.OperationalError(2013, f"injected {self.stage} commit")


@skipUnless(os.getenv("RAG_V2_RUN_INTEGRATION") == "1", "requires provisioned, dedicated MySQL/Redis/models")
class KnowledgeIntegrationTests(TestCase):
    def test_workflow_and_failure_boundaries(self):
        self.assertEqual(os.environ["RAG_V2_ENABLED"], "1")
        self.assertEqual(os.environ["DEEPSEEK_API_BASE"], "http://127.0.0.1:1")
        import api_app
        from knowledge_v2 import KnowledgeError
        from redis_utils import get_redis_client

        token = {"X-RAG-V2-Token": os.environ["RAG_V2_ADMIN_TOKEN"]}
        question = "售后政策"
        name = f"stage1-{uuid4().hex}.txt"
        original = "售后政策\n7\n\n金额\n100".encode()
        replacement = "售后政策\n15\n\n金额\n200".encode()
        saved_generation = None

        def checked(response, status=200):
            self.assertEqual(response.status_code, status, response.text)
            return response.json()

        with patch.object(api_app, "load_llm", return_value=SimpleNamespace(
            invoke=lambda *_: SimpleNamespace(content="依据所给售后政策回答。")
        )):
            with TestClient(api_app.app, headers=token) as client:
                store = api_app.app.state.knowledge
                snapshot_api = "/v2/index-snapshots"
                def build():
                    return checked(client.post(snapshot_api), 201)["snapshot_id"]

                def publication(snapshot, generation, key=None, operation="publish", status=200):
                    return checked(client.post(f"{snapshot_api}/{snapshot}/{operation}", json={
                        "expected_generation": generation, "idempotency_key": key or uuid4().hex
                    }), status)

                def policy():
                    return checked(client.post("/rag/query", json={
                        "question": question, "top_k": 12, "min_rerank_score": -100
                    }))

                def evidence(result, content, version):
                    self.assertEqual(result["code"], "SUCCESS", result)
                    matches = [item for item in result["evidence"] if item["source"] == name]
                    self.assertTrue(matches, result)
                    for item in matches:
                        self.assertEqual(item["file_hash"], md5(content).hexdigest())
                        self.assertEqual(item["version"], version)
                        self.assertIsInstance(item["chunk_id"], int)
                        self.assertTrue(item["child_id"].startswith(item["document_version_id"]))

                self.assertEqual(client.get("/v2/documents", headers={"X-RAG-V2-Token": "wrong"}).status_code, 403)
                initial = store.head()
                checked(client.post("/v2/documents", files={"file": ("broken.pdf", b"bad")}), 422)
                self.assertEqual(store.head(), initial)
                upload = checked(client.post("/v2/documents", files={"file": (name, original)},
                                             data={"version": "v1"}), 201)
                self.assertEqual(upload["revision"], 1)
                document, old_version = upload["document_id"], upload["document_version_id"]
                self.assertNotIn(name, [d["source"] for d in store.pin().manifest["members"]])
                old_snapshot = build()
                chunks = checked(client.get(f"{snapshot_api}/{old_snapshot}/chunks"))
                for child in [c for c in chunks["children"] if c["source"] == name]:
                    parent = next(p for p in chunks["parents"] if p["parent_id"] == child["parent_id"])
                    self.assertEqual(parent["document_version_id"], old_version)
                first = publication(old_snapshot, initial["generation"])["publish_generation"]
                old_view = store.pin()

                # Delay a real Redis write across the next committed publication.
                entered, release = Event(), Event()
                pending_keys = []
                real_set = api_app.set_cache
                def delayed(**kwargs):
                    if kwargs["key"].split(":")[3] == str(first):
                        pending_keys.append(kwargs["key"])
                        entered.set()
                        if not release.wait(30):
                            raise TimeoutError("test did not release old request")
                    return real_set(**kwargs)

                try:
                    with patch.object(api_app, "set_cache", side_effect=delayed), ThreadPoolExecutor(1) as pool:
                        future = pool.submit(policy)
                        try:
                            self.assertTrue(entered.wait(30))
                            new = checked(client.post(f"/v2/documents/{document}/versions",
                                files={"file": (name, replacement)}, data={"version": "v2", "expected_version": old_version}), 201)
                            self.assertEqual(new["revision"], 2)
                            self.assertNotEqual(new["content_hash"], upload["content_hash"])
                            new_snapshot = build()
                            second = publication(new_snapshot, first)["publish_generation"]
                            self.assertEqual(old_view.generation, first)
                            self.assertTrue((store.root / "snapshots" / old_snapshot / "index.faiss").exists())
                        finally:
                            release.set()
                        evidence(future.result(), original, "v1")
                finally:
                    release.set()
                with get_redis_client() as redis:
                    self.assertTrue(redis.exists(pending_keys[0]))
                response = policy()
                self.assertFalse(response["cache_hit"])
                evidence(response, replacement, "v2")
                self.assertTrue(policy()["cache_hit"])
                request = api_app.PolicyRAGRequest(question=question, top_k=12, min_rerank_score=-100)
                bypass = api_app._query_policy_rag(request, evaluation_options={"cache_mode": "bypass"}, trace={})
                self.assertFalse(bypass["cache_hit"])
                evidence(bypass, replacement, "v2")
                refused = checked(client.post("/rag/query", json={"question": question, "min_rerank_score": 100}))
                self.assertTrue(refused["refused"])
                self.assertEqual(refused["evidence"], [])
                ask = checked(client.post("/ask", json={"question": question, "min_rerank_score": -100}))
                self.assertEqual(ask["answer"], "依据所给售后政策回答。")
                self.assertTrue(ask["sources"])
                ask_refused = checked(client.post("/ask", json={"question": question, "min_rerank_score": 100}))
                self.assertEqual(ask_refused["sources"], [])

                third = publication(old_snapshot, second, operation="rollback")["publish_generation"]
                self.assertFalse(policy()["cache_hit"])
                evidence(policy(), original, "v1")
                deleted = checked(client.delete(f"/v2/documents/{document}",
                                  params={"expected_version": new["document_version_id"]}))
                self.assertEqual(deleted["document_status"], "deleted")
                with patch.object(store.embedder, "encode", side_effect=RuntimeError("injected build error")):
                    # Keep another fixture in the draft so encoding is required.
                    other = checked(client.post("/v2/documents", files={"file": (
                        f"fault-{uuid4().hex}.txt", original)}), 201)
                    checked(client.post(snapshot_api), 503)
                self.assertEqual(store.head()["generation"], third)
                self.assertTrue(any(s["build_status"] == "failed" for s in store.snapshots()))

                candidate = build()
                self.assertFalse(any(m["source"] == name for m in store.load_snapshot(candidate).manifest["members"]))
                connect = store.connect
                for boundary in ("before", "after"):
                    generation = store.head()["generation"]
                    key = uuid4().hex
                    with patch.object(store, "connect", side_effect=lambda: PublicationConnection(connect(), boundary)):
                        publication(candidate, generation, key, status=503)
                    self.assertEqual(store.head()["generation"], generation + (boundary == "after"))
                    task = next(row for row in store.publication_tasks() if row["idempotency_key"] == key)
                    self.assertEqual(task["task_status"], "succeeded" if boundary == "after" else "pending")
                    committed = publication(candidate, generation, key)
                    self.assertEqual(committed["publish_generation"], generation + 1)
                    self.assertEqual(publication(candidate, generation, key), committed)
                    candidate = build()

                another = build()
                generation = store.head()["generation"]
                def compete(snapshot):
                    try:
                        return store.publish(snapshot, generation, uuid4().hex)["snapshot_id"]
                    except KnowledgeError as error:
                        self.assertEqual(error.status, 409)
                        return None

                with ThreadPoolExecutor(2) as pool:
                    outcomes = list(pool.map(compete, (candidate, another)))
                self.assertEqual(sum(item is not None for item in outcomes), 1)
                self.assertEqual(store.head()["generation"], generation + 1)
                winner = store.head()["snapshot_id"]
                damaged = Path(store.root / "snapshots" / winner / "index.faiss")
                damaged.write_bytes(b"fault-injection in test fixture only")
                self.assertEqual(policy()["code"], "RAG_QUERY_FAILED")
                saved_generation = publication(old_snapshot, generation + 1, operation="rollback")["publish_generation"]
                evidence(policy(), original, "v1")
                checked(client.delete(f'/v2/documents/{other["document_id"]}',
                                      params={"expected_version": other["document_version_id"]}))

            with TestClient(api_app.app, headers=token) as restarted:
                new_store = api_app.app.state.knowledge
                self.assertEqual(new_store.pin().generation, saved_generation)
                self.assertEqual(new_store.pin().snapshot_id, old_snapshot)
                self.assertEqual(checked(restarted.get("/rag_status"))["processed_file_count"],
                                 len(new_store.pin().manifest["members"]))
