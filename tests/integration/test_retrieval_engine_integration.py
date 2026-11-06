"""Opt-in MySQL, Redis, real FAISS/BGE/CrossEncoder, with local LLM substitutes."""
from hashlib import sha256
import os
import time
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient


@skipUnless(os.getenv("RAG_V2_RUN_INTEGRATION") == "1", "requires the verified isolated development stack")
class RetrievalIntegrationTests(TestCase):
    def test_build_publish_cache_and_full_retrieval_experiment(self):
        self.assertEqual(os.environ["DEEPSEEK_API_BASE"], "http://127.0.0.1:1")
        import api_app
        from knowledge_v2 import KnowledgeError
        from redis_utils import get_redis_client

        token = {"X-RAG-V2-Token": os.environ["RAG_V2_ADMIN_TOKEN"]}
        model_id = "AX-" + uuid4().hex[:7]
        question = f"那 {model_id} 质量问题可以申请退款？退款运费谁承担？"
        history = [{"role": "user", "content": f"我想了解 {model_id} 售后政策。"},
                   {"role": "assistant", "content": "未经核实的假设金额 999 元"}]
        quote = question + f"\n{model_id} 质量问题可以申请退款，退款运费由商家承担。"
        text = quote + "政策适用范围与条件需审核。" * 45
        raw = text.encode()
        name = f"stage3-fixture-{uuid4().hex[:8]}.txt"
        hypothetical = "HYPOTHETICAL_ONLY_ZXQ-427"
        llm = Mock(invoke=Mock(return_value=SimpleNamespace(
            content="合成测试回答，申请与运费条款见真实资料。", usage_metadata={"input_tokens": 24, "output_tokens": 9, "total_tokens": 33})))
        hyde = Mock(return_value=SimpleNamespace(content=hypothetical,
                    usage_metadata={"input_tokens": 11, "output_tokens": 6, "total_tokens": 17}))
        path = "/v2/retrieval/debug"
        ep = "/v2/evaluation"
        with patch.object(api_app, "load_llm", return_value=llm), patch.object(api_app, "hyde_generator", return_value=hyde):
            with TestClient(api_app.app, headers=token) as client:
                def call(method, url, status=200, **kwargs):
                    response = client.request(method, url, **kwargs)
                    self.assertEqual(response.status_code, status, response.text)
                    return response.json()

                store = api_app.app.state.knowledge
                initial = store.pin()
                self.assertIsNotNone(initial.snapshot_id)
                hybrid = {"retrieval_mode": "hybrid", "bm25_enabled": True, "rrf_enabled": True}
                sparse = {"retrieval_mode": "sparse", "dense_enabled": False, "bm25_enabled": True,
                          "rerank_enabled": False}
                if initial.sparse is None:
                    fallback = call("POST", path, json={"question": "售后政策", "retrieval": sparse})
                    self.assertIn("bm25_missing_from_snapshot", fallback["trace"]["retrieval"]["degraded"])
                doc = call("POST", "/v2/documents", 201, files={"file": (name, raw)}, data={"version": "stage3-v1"})
                candidate = call("POST", "/v2/index-snapshots", 201)
                sid = candidate["snapshot_id"]
                fresh = store.load_snapshot(sid, strict_sparse=True)
                self.assertEqual(fresh.sparse["child_ids"], [c["child_id"] for c in fresh.manifest["children"]])
                self.assertEqual(fresh.sparse["corpus_snapshot_id"], fresh.corpus_snapshot_id)
                artifact = store.root / "snapshots" / sid / "bm25.json"
                original = artifact.read_bytes()
                try:
                    artifact.write_bytes(b"intentionally damaged synthetic fixture")
                    with self.assertRaises(KnowledgeError):
                        store.load_snapshot(sid, strict_sparse=True)
                    self.assertEqual(store.load_snapshot(sid).sparse_error, "bm25_artifact_invalid")
                finally:
                    artifact.write_bytes(original)
                publication = call("POST", f"/v2/index-snapshots/{sid}/publish", json={
                    "expected_generation": initial.generation, "idempotency_key": uuid4().hex})
                generation = publication["publish_generation"]
                self.assertNotEqual(initial.snapshot_id, sid)
                self.assertFalse(any(c["source"] == name for c in initial.manifest["children"]))

                def debug(cfg, mode="retrieval", q=question):
                    return call("POST", path, json={"question": q, "chat_history": history, "mode": mode,
                        "retrieval": cfg, "candidate_k": 40, "final_k": 20, "rerank_threshold": 100})
                a = debug(sparse, q=model_id)
                self.assertEqual(a["trace"]["retrieval"]["channels"], {"dense": "not_used", "sparse": "used"})
                self.assertEqual(a["trace"]["retrieval"]["routes"][0]["results"][0]["child_id"], doc["document_version_id"] + "-c1")
                gates = debug(hybrid, q="售后政策")
                no_gate = debug(hybrid | {"threshold_enabled": False}, q="售后政策")
                self.assertEqual(gates["trace"]["ranked_candidates"], no_gate["trace"]["ranked_candidates"])
                self.assertEqual(gates["result"]["evidence"], [])
                self.assertTrue(no_gate["result"]["evidence"])
                advanced = hybrid | {"threshold_enabled": False, "query_rewrite_enabled": True,
                    "query_decomposition_enabled": True, "query_expansion_enabled": True,
                    "hyde_enabled": True, "parent_child_enabled": True}
                response = debug(advanced, "answer")
                trace = response["trace"]["retrieval"]
                self.assertEqual(response["trace"]["status"], "answered", response)
                self.assertEqual(trace["subquery_gaps"], [])
                self.assertEqual([trace["query"][k]["status"] for k in ("rewrite", "decomposition", "expansion", "hyde")],
                                 ["used"] * 4)
                self.assertEqual(trace["parent_child"]["status"], "used")
                self.assertTrue(trace["parent_child"]["context"])
                self.assertEqual(trace["publish_generation"], generation)
                prompt = llm.invoke.call_args.args[0]
                self.assertNotIn(hypothetical, prompt)
                self.assertNotIn("999", prompt)
                self.assertTrue(any(source["source"] == name for source in response["result"]["sources"]))
                for source in response["result"]["sources"]:
                    self.assertIsInstance(source["chunk_id"], int)
                    self.assertTrue(source["child_id"])
                    self.assertEqual(source["index_snapshot_id"], sid)
                self.assertTrue(any(source["version"] == "stage3-v1" and source["end"] > source["start"]
                                    for source in response["result"]["sources"] if source["source"] == name))
                missing = debug(advanced, "answer", q=question + "维修金额8888元吗？")
                self.assertEqual(missing["trace"]["status"], "refused_incomplete_subqueries")

                base = {"question": f"售后政策 {uuid4().hex[:10]}", "min_rerank_score": -100}
                first = call("POST", "/ask", json=base)
                second = call("POST", "/ask", json=base | {"retrieval": hybrid, "chat_history": history})
                third = call("POST", "/ask", json=base | {"retrieval": hybrid, "chat_history": history[:1]})
                self.assertEqual(len({first["cache_key"], second["cache_key"], third["cache_key"]}), 3)
                with get_redis_client() as redis:
                    self.assertEqual(redis.exists(first["cache_key"], second["cache_key"], third["cache_key"]), 3)

                draft = call("POST", ep + "/datasets", 201, json={
                    "dataset_version": "v1", "split": "dev", "name": "Synthetic stage 3 retrieval fixture",
                    "questions": [{"question_id": "Q1", "question": "售后政策", "question_type": "fixture",
                        "answerability": "answerable", "expected_action": "answer", "difficulty": "easy",
                        "required_facts": [{"fact_id": "F1", "text": "合成退款规则", "purpose": "answer",
                                            "evidence_options": [["E1"]]}]}],
                    "evidence": [{"evidence_id": "E1", "question_id": "Q1", "source_id": "stage3-fixture",
                                  "source_path": name, "source_content_hash": sha256(raw).hexdigest(),
                                  "original_text_quote": quote, "source_offset": {"start": 0, "end": len(quote)},
                                  "role": "answer_support", "necessity": "required", "supports_fact_ids": ["F1"]}]})
                ds_path = f"{ep}/datasets/{draft['dataset_id']}/v1"
                for collection, entity in (("questions", "Q1"), ("evidence", "E1")):
                    call("POST", ds_path + "/reviews", json={"collection": collection, "entity_id": entity,
                        "action": "approve", "note": "Synthetic engineering fixture"})
                call("POST", ds_path + "/freeze")
                cfg = {"dataset_id": draft["dataset_id"], "dataset_version": "v1", "mode": "retrieval",
                    "corpus_snapshot_id": fresh.corpus_snapshot_id, "index_snapshot_id": sid, "publish_generation": generation,
                    "candidate_k": 30, "final_k": 15, "rerank_enabled": False, "threshold_enabled": False}
                ids = []
                for strategy in ({}, sparse, hybrid, hybrid | {"rerank_enabled": True}):
                    run = call("POST", ep + "/runs", 201, json=cfg | strategy)
                    identifier = run["run_id"]
                    ids.append(identifier)
                    run_path = f"{ep}/runs/{identifier}"
                    call("POST", run_path + "/start")
                    for _ in range(150):
                        report = call("GET", run_path)
                        if report["run"]["state"] == "completed":
                            break
                        time.sleep(0.1)
                    self.assertEqual(report["run"]["completed"], 1)
                    self.assertEqual(report["cases"][0]["trace"]["retrieval"]["snapshot_id"], sid)
                comparison = call("GET", ep + "/compare", params={"ids": ",".join(ids)})
                self.assertEqual(comparison["paired"]["completed_intersection"], 1)
                self.assertEqual(len(comparison["runs"]), 4)

            with TestClient(api_app.app, headers=token) as restarted:
                self.assertEqual(api_app.app.state.knowledge.pin().snapshot_id, sid)
                self.assertTrue(api_app.app.state.knowledge.pin().sparse)
                self.assertEqual(restarted.get(f"{ep}/runs/{ids[-1]}").status_code, 200)
