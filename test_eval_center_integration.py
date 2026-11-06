"""Opt-in with the dedicated stage-one database, Redis and model paths."""
import os
import json
import time
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient


@skipUnless(os.getenv("RAG_V2_RUN_INTEGRATION") == "1", "requires verified isolated stage-two database")
class EvalCenterIntegrationTests(TestCase):
    def test_full_workflow(self):
        self.assertEqual(os.environ["DEEPSEEK_API_BASE"], "http://127.0.0.1:1")
        import api_app
        from redis_utils import get_redis_client

        prefix = "/v2/evaluation"
        headers = {"X-RAG-V2-Token": os.environ["RAG_V2_ADMIN_TOKEN"]}
        with patch.object(api_app, "load_llm", return_value=SimpleNamespace(invoke=lambda *_: SimpleNamespace(
            content="依据售后政策。", usage_metadata={"input_tokens": 15, "output_tokens": 4, "total_tokens": 19}
        ))):
            with TestClient(api_app.app, headers=headers) as client:
                def call(method, path, status=200, **kwargs):
                    response = client.request(method, path, **kwargs)
                    self.assertEqual(response.status_code, status, response.text)
                    return response.json()

                previous = call("GET", "/v2/index-snapshots")["head"]
                self.assertIsNotNone(previous["snapshot_id"])
                store = api_app.app.state.knowledge
                view = store.pin()
                member = next(m for m in view.manifest["members"] if m["source"].startswith("phase1-demo"))
                raw = store.document_artifact(member["document_version_id"])["pages"][0]["text"]
                quote = "售后政策"
                offset = raw.index(quote)
                dataset = {"name": f"stage2-test-{uuid4().hex[:8]}", "dataset_version": "v1", "split": "dev",
                           "questions": [{"question_id": "Q1", "question": quote, "question_type": "fixture",
                                          "answerability": "answerable", "expected_action": "answer", "difficulty": "easy",
                                          "required_facts": [{"fact_id": "F1", "text": quote, "purpose": "answer",
                                                              "evidence_options": [["E1"]]}]}],
                           "evidence": [{"evidence_id": "E1", "question_id": "Q1", "source_id": "fixture",
                                         "source_path": member["source"], "source_content_hash": member["content_hash"],
                                         "original_text_quote": quote, "source_offset": {"start": offset, "end": offset + len(quote)},
                                         "role": "answer_support", "necessity": "required", "supports_fact_ids": ["F1"]},
                                        {"evidence_id": "E2", "question_id": "Q1", "source_id": "missing",
                                         "source_path": "missing.txt", "source_content_hash": "0" * 64,
                                         "original_text_quote": quote, "role": "context"}]}
                historical = call("POST", prefix + "/datasets/import-v3", 201)
                self.assertEqual((len(historical["body"]["questions"]), len(historical["body"]["evidence"])), (66, 226))
                self.assertEqual([sum(group["rows"] for group in batch["runs"].values())
                                  for batch in call("GET", prefix + "/historical")], [264, 132])
                created = call("POST", prefix + "/datasets", 201, json=dataset)
                identifier = created["dataset_id"]
                path = f"{prefix}/datasets/{identifier}/v1"
                mapping = call("GET", f"{path}/mappings/{view.snapshot_id}")
                self.assertEqual(mapping["review_queue"], ["E2"])
                self.assertEqual(mapping["results"][0]["mapping"]["status"], "mapped")
                self.assertTrue(mapping["results"][0]["parents"])
                for collection, entity in (("questions", "Q1"), ("evidence", "E1")):
                    for action, status in (("dispute", "disputed"), ("return", "proposed_unreviewed"),
                                           ("approve", "developer_self_reviewed")):
                        review = call("POST", path + "/reviews", json=dict(collection=collection, entity_id=entity,
                                      action=action, note="Synthetic integration fixture only"))
                        self.assertEqual(review["review_status"], status)
                self.assertEqual(len(call("GET", path + "/audit")), 6)
                frozen = call("POST", path + "/freeze")
                self.assertTrue(frozen["frozen"])
                call("PUT", path, 409, json={**frozen["body"], "expected_sha256": frozen["sha256"]})
                forked = call("POST", path + "/fork", 201, params={"new_version": "v2"})
                self.assertFalse(forked["frozen"])
                call("POST", prefix + "/datasets", 403, json=dataset | {"split": "holdout"})
                from knowledge_v2 import KnowledgeError
                custodian = uuid4().hex
                independent = uuid4().hex
                adjudicator = uuid4().hex
                with patch.dict(os.environ, {"RAG_V2_EVAL_HOLDOUT_TOKEN": custodian, "RAG_V2_EVAL_REVIEWERS": json.dumps({
                    "test-independent": {"token": independent, "role": "independent_reviewer"},
                    "test-adjudicator": {"token": adjudicator, "role": "adjudicator"},
                })}):
                    holdout_headers = {"X-Eval-Holdout": custodian}
                    duplicate = dataset | {"split": "holdout", "questions": [dataset["questions"][0] | {
                        "leakage_group_id": "synthetic-duplicate", "provenance": {"source_type": "synthetic_test"}
                    }]}
                    call("POST", prefix + "/datasets", 409, json=duplicate, headers=holdout_headers)
                    holdout = call("POST", prefix + "/datasets", 201, headers=holdout_headers,
                                   json={"name": "Synthetic holdout auth probe", "split": "holdout", "dataset_version": "v1"})
                    call("GET", f"{prefix}/datasets/{holdout['dataset_id']}/v1", 403)
                    fork_path = f"{prefix}/datasets/{identifier}/v2"
                    for action, reviewer, expected in (
                        ("approve", independent, "independently_reviewed"), ("dispute", independent, "disputed"),
                        ("adjudicate", adjudicator, "adjudicated"),
                    ):
                        reviewed = call("POST", fork_path + "/reviews", headers={"X-Eval-Reviewer": reviewer}, json={
                            "collection": "evidence", "entity_id": "E1", "action": action, "note": "Synthetic role probe"
                        })
                        self.assertEqual(reviewed["review_status"], expected)
                cfg = {"dataset_id": identifier, "dataset_version": "v1", "corpus_snapshot_id": view.corpus_snapshot_id,
                       "index_snapshot_id": view.snapshot_id, "publish_generation": view.generation, "mode": "retrieval",
                       "candidate_k": 40, "final_k": 20, "rerank_enabled": True, "rerank_threshold": -100}
                call("POST", prefix + "/runs", 409, json=cfg | {"corpus_snapshot_id": "wrong"})

                def run(config, before_start=None):
                    item = call("POST", prefix + "/runs", 201, json=config)
                    path = f"{prefix}/runs/{item['run_id']}"
                    if before_start:
                        before_start(path)
                    call("POST", path + "/start")
                    for _ in range(150):
                        report = call("GET", path)
                        if report["run"]["state"] == "completed":
                            return report
                        self.assertNotIn(report["run"]["state"], {"failed", "cancelled"})
                        time.sleep(0.1)
                    self.fail("evaluation did not complete")

                def cancelled_then_publish(path):
                    self.assertEqual(call("POST", path + "/cancel")["state"], "cancelled")
                    call("POST", f"/v2/index-snapshots/{view.snapshot_id}/rollback", json={
                        "expected_generation": view.generation, "idempotency_key": uuid4().hex})

                with get_redis_client() as redis:
                    keys = set(redis.scan_iter(match="rag:v2:*"))
                    a = run(cfg, cancelled_then_publish)
                    b = run(cfg | {"rerank_threshold": 100})
                    self.assertEqual(set(redis.scan_iter(match="rag:v2:*")), keys)
                self.assertEqual(a["run"]["config"]["publish_generation"], previous["generation"])
                self.assertEqual(a["cases"][0]["metrics"]["chunk_final_recall"]["value"], 1)
                self.assertIn("EVIDENCE_INSUFFICIENT", b["cases"][0]["failures"])
                self.assertNotIn("RERANK_LOSS", b["cases"][0]["failures"])
                ids = ",".join(x["run"]["run_id"] for x in (a, b))
                self.assertEqual(call("GET", prefix + "/compare", params={"ids": ids})["paired"]["completed_intersection"], 1)
                interrupted = call("POST", prefix + "/runs", 201, json=cfg)
                interrupted_path = f"{prefix}/runs/{interrupted['run_id']}"
                with patch.object(store, "load_snapshot", side_effect=KnowledgeError("injected artifact error", 503)):
                    call("POST", interrupted_path + "/start")
                    for _ in range(100):
                        failure = call("GET", interrupted_path)
                        if failure["run"]["state"] == "failed":
                            break
                        time.sleep(0.1)
                    self.assertEqual(failure["run"]["error_message"], "KnowledgeError")
                call("POST", interrupted_path + "/start")
                for _ in range(100):
                    recovered = call("GET", interrupted_path)
                    if recovered["run"]["state"] == "completed":
                        break
                    time.sleep(0.1)
                self.assertEqual(recovered["run"]["completed"], 1)
                self.assertEqual(len(recovered["cases"]), 1)
                answer = run(cfg | {"mode": "answer"})
                rid = answer["run"]["run_id"]
                self.assertEqual(answer["cases"][0]["engineering"]["total_tokens"], 19)
                decision = call("POST", f"{prefix}/runs/{rid}/cases/Q1/review", json={
                    "note": "Synthetic semantic labels", "actual_action": "answer", "facts": {"F1": "covered"},
                    "forbidden_violation": False, "target_conclusion": True, "false_answer": False})
                self.assertEqual(decision["answer_review"]["metrics"]["required_fact_coverage_answer"]["value"], 1)

            with TestClient(api_app.app, headers=headers) as restarted:
                response = restarted.get(f"{prefix}/runs/{rid}")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["cases"][0]["answer_review"]["cohort"], "developer")
