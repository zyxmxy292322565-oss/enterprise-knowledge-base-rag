import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import api_app
import db_utils
from evaluation_tracking import (
    cache_status,
    dataset_manifest,
    runtime_manifest,
    split_trace_artifacts,
)


class TrackingHelpersTests(unittest.TestCase):
    def test_cache_status_distinguishes_bypass_hit_miss_and_unknown(self):
        self.assertEqual(cache_status("bypass", False), "bypass")
        self.assertEqual(cache_status("normal", True), "hit")
        self.assertEqual(cache_status("normal", False), "miss")
        self.assertEqual(cache_status(None, False), "unknown")

    def test_dataset_manifest_is_stable_and_content_sensitive(self):
        first = dataset_manifest([
            {"ID": 2, "问题": "B", "标准答案": "b", "期望关键词": "kb"},
            {"ID": 1, "问题": "A", "标准答案": "a", "期望关键词": "ka"},
        ], "reviewed-v1")
        reordered = dataset_manifest([
            {"ID": 1, "问题": "A", "标准答案": "a", "期望关键词": "ka"},
            {"ID": 2, "问题": "B", "标准答案": "b", "期望关键词": "kb"},
        ], "reviewed-v1")
        changed = dataset_manifest([
            {"ID": 1, "问题": "A", "标准答案": "changed", "期望关键词": "ka"},
            {"ID": 2, "问题": "B", "标准答案": "b", "期望关键词": "kb"},
        ], "reviewed-v1")
        self.assertEqual(first["dataset_manifest_sha256"], reordered["dataset_manifest_sha256"])
        self.assertNotEqual(first["dataset_manifest_sha256"], changed["dataset_manifest_sha256"])

    def test_large_trace_text_is_split_into_controlled_artifacts(self):
        compact, artifacts = split_trace_artifacts({
            "context": "完整上下文",
            "prompt": "完整提示词",
            "faiss_candidates": [{"faiss_index": 1, "content": "候选原文"}],
        })
        self.assertNotIn("context", compact)
        self.assertNotIn("prompt", compact)
        self.assertNotIn("content", compact["faiss_candidates"][0])
        self.assertEqual(artifacts["context_text"], "完整上下文")
        self.assertEqual(artifacts["prompt_text"], "完整提示词")
        self.assertEqual(len(compact["context_sha256"]), 64)
        self.assertEqual(len(compact["faiss_candidates"][0]["content_sha256"]), 64)

    def test_runtime_manifest_hashes_read_only_source_documents_and_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
            index_path = root / "index.bin"
            index_path.write_bytes(b"index")
            state = SimpleNamespace(
                all_docs=[SimpleNamespace(metadata={"source": "synthetic.txt", "file_hash": "abc"})],
                index_cache_file=str(index_path),
                vectors_cache_file=None,
                all_docs_cache_file=None,
                chat_model=SimpleNamespace(temperature=None),
            )

            def prompt_callable():
                return "prompt"

            manifest = runtime_manifest(
                app_state=state,
                prompt_callable=prompt_callable,
                source_root=root,
                embedding_model="embed",
                rerank_model="rerank",
                llm_model="llm",
            )

        self.assertEqual(len(manifest["source_files_sha256"]), 64)
        self.assertEqual(len(manifest["document_manifest_sha256"]), 64)
        self.assertEqual(manifest["index_files"]["index_cache_file"]["file_name"], "index.bin")
        self.assertIsNone(manifest["git_commit"])
        self.assertIn("not found", manifest["git_note"])


class CacheStatisticsTests(unittest.TestCase):
    def test_bypass_and_unknown_are_excluded_from_normal_hit_rate(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (10, 0.5, 100.0, 50.0, 2, 2, 3, 3)
        connection = MagicMock()
        connection.cursor.return_value = cursor
        with patch.object(db_utils, "get_connection", return_value=connection):
            stats = db_utils.load_rag_eval_stats()
        self.assertEqual(stats["normal_cache_count"], 4)
        self.assertEqual(stats["cache_bypass_count"], 3)
        self.assertEqual(stats["cache_unknown_count"], 3)
        self.assertEqual(stats["cache_hit_rate"], 0.5)

    def test_historical_false_is_unknown_and_historical_true_is_hit(self):
        rows = [
            (1, 1, "q1", "a1", "k", 1.0, 1, 1, 0, None, 10, 5, 3, 1, "t", None, None),
            (2, 1, "q2", "a2", "k", 1.0, 1, 1, 1, None, 10, 0, 3, 1, "t", None, None),
        ]
        cursor = MagicMock()
        cursor.fetchall.return_value = rows
        connection = MagicMock()
        connection.cursor.return_value = cursor
        with patch.object(db_utils, "get_connection", return_value=connection):
            result = db_utils.load_rag_eval_results(limit=2)
        self.assertEqual(result[0]["缓存状态"], "unknown")
        self.assertEqual(result[1]["缓存状态"], "hit")


class TrackingStorageTests(unittest.TestCase):
    def _connection(self):
        cursor = MagicMock()
        cursor.lastrowid = 123

        def validate_placeholders(sql, params=()):
            self.assertEqual(sql.count("%s"), len(params), sql)

        cursor.execute.side_effect = validate_placeholders
        connection = MagicMock()
        connection.cursor.return_value = cursor
        return connection, cursor

    def test_tracked_result_case_run_and_artifact_use_one_transaction(self):
        connection, cursor = self._connection()
        with patch.object(db_utils, "get_connection", return_value=connection):
            result_id = db_utils.save_tracked_rag_eval_case_run(
                case_run_id="case-run",
                run_id="run",
                case_id=1,
                question="q",
                expected_answer="a",
                expected_keywords="k",
                actual_answer="answer",
                hit_keywords=["k"],
                keyword_score=1.0,
                hit_count=1,
                keyword_count=1,
                from_cache=False,
                cache_status="bypass",
                total_ms=10,
                llm_ms=5,
                top_k=1,
                source_count=1,
                status="answered",
                effective_parameters={"faiss_k": 2},
                retrieval_trace={"status": "answered"},
                scoring={"keyword_score": 1.0},
                timing={"total_ms": 10},
                refusal_reason=None,
                error_stage=None,
                error_type=None,
                artifacts={
                    "context_text": "context",
                    "prompt_text": "prompt",
                    "context_sha256": "c" * 64,
                    "prompt_sha256": "p" * 64,
                },
            )
        self.assertEqual(result_id, 123)
        self.assertEqual(cursor.execute.call_count, 3)
        connection.commit.assert_called_once()
        connection.rollback.assert_not_called()

    def test_tracked_write_rolls_back_if_case_run_insert_fails(self):
        connection = MagicMock()
        cursor = MagicMock()
        cursor.lastrowid = 123
        calls = 0

        def fail_second(sql, params=()):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("duplicate run/case")

        cursor.execute.side_effect = fail_second
        connection.cursor.return_value = cursor
        with patch.object(db_utils, "get_connection", return_value=connection):
            with self.assertRaises(RuntimeError):
                db_utils.save_tracked_rag_eval_case_run(
                    case_run_id="case-run",
                    run_id="run",
                    case_id=1,
                    question="q",
                    expected_answer="a",
                    expected_keywords="k",
                    actual_answer="answer",
                    hit_keywords=["k"],
                    keyword_score=1.0,
                    hit_count=1,
                    keyword_count=1,
                    from_cache=False,
                    cache_status="bypass",
                    total_ms=10,
                    llm_ms=5,
                    top_k=1,
                    source_count=1,
                    status="answered",
                    effective_parameters={},
                    retrieval_trace={},
                    scoring={},
                    timing={},
                    refusal_reason=None,
                    error_stage=None,
                    error_type=None,
                    artifacts=None,
                )
        connection.rollback.assert_called_once()
        connection.commit.assert_not_called()


class TrackedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            SimpleNamespace(page_content="证据一", metadata={"source": "one.txt", "file_hash": "a", "chunk_id": 1}),
            SimpleNamespace(page_content="证据二", metadata={"source": "two.txt", "file_hash": "b", "chunk_id": 2}),
        ]
        self.index = MagicMock()
        self.index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        self.embed = MagicMock()
        self.embed.encode.return_value = np.array([[1.0, 2.0]])
        self.rerank = MagicMock()
        self.rerank.predict.return_value = np.array([0.2, 0.9])
        self.llm = MagicMock()
        self.llm.invoke.return_value = SimpleNamespace(content="依据证据回答")
        self.case = {"id": 7, "question": "问题", "expected_answer": "回答", "expected_keywords": "依据,回答"}

    def test_effective_parameters_and_candidates_are_persisted_for_case_run(self):
        captured = {}

        def save(**kwargs):
            captured.update(kwargs)
            return 88

        with patch.object(api_app.app.state, "all_docs", self.docs, create=True), \
             patch.object(api_app.app.state, "vectors", np.ones((2, 2)), create=True), \
             patch.object(api_app.app.state, "index", self.index, create=True), \
             patch.object(api_app.app.state, "embed_model", self.embed, create=True), \
             patch.object(api_app.app.state, "rerank_model", self.rerank, create=True), \
             patch.object(api_app.app.state, "chat_model", self.llm, create=True), \
             patch.object(api_app, "get_cache", side_effect=AssertionError("cache read")), \
             patch.object(api_app, "set_cache", side_effect=AssertionError("cache write")), \
             patch.object(api_app, "incr_cache_stat", side_effect=AssertionError("cache stat")), \
             patch.object(api_app, "save_tracked_rag_eval_case_run", side_effect=save):
            result = api_app._execute_tracked_rag_eval_case(
                self.case,
                api_app.RagEvalRunRequest(top_k=1, faiss_multiplier=2),
                "run-1",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["run_id"], "run-1")
        self.assertEqual(captured["cache_status"], "bypass")
        self.assertEqual(captured["effective_parameters"]["faiss_k"], 2)
        self.assertEqual(captured["effective_parameters"]["selected_context_count"], 1)
        self.assertEqual(len(captured["retrieval_trace"]["faiss_candidates"]), 2)
        self.assertEqual(captured["retrieval_trace"]["ranked_candidates"][0]["faiss_index"], 1)
        self.assertNotIn("content", captured["retrieval_trace"]["faiss_candidates"][0])
        self.assertIn("证据二", captured["artifacts"]["context_text"])
        self.assertEqual(captured["scoring"]["keyword_score"], 1.0)

    def test_same_experiment_can_create_distinct_runs_without_overwriting(self):
        experiment = {
            "experiment_id": 9,
            "dataset_version": "v1",
            "dataset_manifest_sha256": dataset_manifest([self.case], "v1")["dataset_manifest_sha256"],
        }
        created_run_ids = []

        def create_run(**kwargs):
            created_run_ids.append(kwargs["run_id"])

        request = api_app.RagEvalRunRequest(experiment_id=9, dataset_version="v1")
        with patch.object(api_app, "get_rag_eval_experiment_definition", return_value=experiment), \
             patch.object(api_app, "_runtime_eval_manifest", return_value={"manifest": "test"}), \
             patch.object(api_app, "create_rag_eval_run", side_effect=create_run):
            first = api_app._create_or_reuse_experiment_and_run(request, [self.case])
            second = api_app._create_or_reuse_experiment_and_run(request, [self.case])

        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])
        self.assertEqual(created_run_ids, [first[1], second[1]])

    def test_migration_declares_foreign_keys_and_unique_run_case(self):
        sql = Path("migrations/20261019_rag_eval_tracking_v2.sql").read_text(encoding="utf-8")
        self.assertIn("FOREIGN KEY (experiment_id)", sql)
        self.assertIn("FOREIGN KEY (run_id)", sql)
        self.assertIn("UNIQUE (run_id, case_id)", sql)
        self.assertIn("cache_status VARCHAR(16) NULL", sql)


if __name__ == "__main__":
    unittest.main()
