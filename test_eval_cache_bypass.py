import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import api_app


class ExistingEvaluationCacheBypassTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            SimpleNamespace(
                page_content="评测专用合成证据",
                metadata={"source": "synthetic.txt", "file_hash": "hash", "chunk_id": 1},
            )
        ]
        self.index = MagicMock()
        self.index.search.return_value = (
            np.array([[0.1]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        self.embed = MagicMock()
        self.embed.encode.return_value = np.array([[1.0, 2.0]], dtype=np.float32)
        self.rerank = MagicMock()
        self.rerank.predict.return_value = np.array([0.9], dtype=np.float32)
        self.llm = MagicMock()
        self.llm.invoke.return_value = SimpleNamespace(content="合成回答")
        self.case = {
            "question": "合成问题",
            "expected_answer": "合成回答",
            "expected_keywords": "合成,回答",
        }

    def rag_state(self):
        return (
            patch.object(api_app.app.state, "all_docs", self.docs, create=True),
            patch.object(api_app.app.state, "vectors", np.ones((1, 2)), create=True),
            patch.object(api_app.app.state, "index", self.index, create=True),
            patch.object(api_app.app.state, "embed_model", self.embed, create=True),
            patch.object(api_app.app.state, "rerank_model", self.rerank, create=True),
            patch.object(api_app.app.state, "chat_model", self.llm, create=True),
        )

    def test_existing_eval_route_bypasses_redis_and_runs_full_rag_pipeline(self):
        state_patches = self.rag_state()
        with state_patches[0], state_patches[1], state_patches[2], \
             state_patches[3], state_patches[4], state_patches[5], \
             patch.object(api_app, "get_rag_eval_case_by_id", return_value=self.case), \
             patch.object(api_app, "get_cache", side_effect=AssertionError("cache read")) as cache_read, \
             patch.object(api_app, "set_cache", side_effect=AssertionError("cache write")) as cache_write, \
             patch.object(api_app, "incr_cache_stat", side_effect=AssertionError("cache stats")) as cache_stats, \
             patch.object(api_app, "save_qa_logs", side_effect=AssertionError("qa log")) as qa_log, \
             patch.object(api_app, "save_rag_request_log", side_effect=AssertionError("request log")) as perf_log, \
             patch.object(
                 api_app,
                 "_create_or_reuse_experiment_and_run",
                 return_value=(1, "run-1", {"dataset_version": "test", "dataset_manifest_sha256": "d"}, {
                     "top_k": 1, "faiss_multiplier": 2, "min_rerank_score": 0.2,
                     "max_context_chars": 6000, "max_chunk_chars": 1200,
                     "low_score_threshold": 0.6, "cache_mode": "bypass",
                 }),
             ), \
             patch.object(api_app, "save_tracked_rag_eval_case_run", return_value=101) as save_result, \
             patch.object(api_app, "finish_rag_eval_run"), \
             patch.object(api_app, "update_rag_eval_experiment_summary"):
            result = api_app.run_rag_eval_case(1, api_app.RagEvalRunRequest(top_k=1))

        self.assertTrue(result["success"])
        self.assertEqual(result["cache_mode"], "bypass")
        self.assertEqual(result["cache_status"], "bypass")
        self.assertEqual(result["run_id"], "run-1")
        self.assertIsNotNone(result["case_run_id"])
        self.assertFalse(result["ask_result"]["from_cache"])
        self.assertEqual(result["ask_result"]["answer"], "合成回答")
        self.embed.encode.assert_called_once()
        self.index.search.assert_called_once()
        self.rerank.predict.assert_called_once()
        self.llm.invoke.assert_called_once()
        save_result.assert_called_once()
        self.assertFalse(save_result.call_args.kwargs["from_cache"])
        self.assertEqual(save_result.call_args.kwargs["cache_status"], "bypass")
        for mocked in (cache_read, cache_write, cache_stats, qa_log, perf_log):
            mocked.assert_not_called()

    def test_public_ask_cache_hit_behavior_is_unchanged(self):
        cached = {"answer": "缓存回答", "sources": [], "top_k": 1, "used_doc_count": 1}
        with patch.object(api_app.app.state, "all_docs", self.docs, create=True), \
             patch.object(api_app.app.state, "vectors", np.ones((1, 2)), create=True), \
             patch.object(api_app.app.state, "index", self.index, create=True), \
             patch.object(api_app, "get_cache", return_value=cached) as cache_read, \
             patch.object(api_app, "incr_cache_stat") as cache_stats, \
             patch.object(api_app, "set_cache") as cache_write, \
             patch.object(api_app, "save_rag_request_log"):
            result = api_app.ask_question(api_app.AskRequest(question="普通问题", top_k=1))

        cache_read.assert_called_once()
        cache_stats.assert_called_once_with("hit")
        cache_write.assert_not_called()
        self.assertTrue(result["from_cache"])

    def test_public_ask_cache_miss_still_records_miss_and_writes_answer(self):
        state_patches = self.rag_state()
        with state_patches[0], state_patches[1], state_patches[2], \
             state_patches[3], state_patches[4], state_patches[5], \
             patch.object(api_app, "get_cache", return_value=None) as cache_read, \
             patch.object(api_app, "incr_cache_stat") as cache_stats, \
             patch.object(api_app, "set_cache") as cache_write, \
             patch.object(api_app, "save_qa_logs"), \
             patch.object(api_app, "save_rag_request_log"):
            result = api_app.ask_question(api_app.AskRequest(question="普通问题", top_k=1))

        cache_read.assert_called_once()
        cache_stats.assert_called_once_with("miss")
        cache_write.assert_called_once()
        self.assertFalse(result["from_cache"])
        self.assertEqual(result["answer"], "合成回答")

    def test_evaluation_execution_error_is_not_saved_as_success(self):
        self.llm.invoke.side_effect = RuntimeError("synthetic provider failure")
        state_patches = self.rag_state()
        with state_patches[0], state_patches[1], state_patches[2], \
             state_patches[3], state_patches[4], state_patches[5], \
             patch.object(api_app, "get_rag_eval_case_by_id", return_value=self.case), \
             patch.object(api_app, "get_cache", side_effect=AssertionError("cache read")), \
             patch.object(api_app, "set_cache", side_effect=AssertionError("cache write")), \
             patch.object(api_app, "incr_cache_stat", side_effect=AssertionError("cache stats")), \
             patch.object(
                 api_app,
                 "_create_or_reuse_experiment_and_run",
                 return_value=(1, "run-error", {"dataset_version": "test", "dataset_manifest_sha256": "d"}, {
                     "top_k": 3, "faiss_multiplier": 2, "min_rerank_score": 0.2,
                     "max_context_chars": 6000, "max_chunk_chars": 1200,
                     "low_score_threshold": 0.6, "cache_mode": "bypass",
                 }),
             ), \
             patch.object(api_app, "save_tracked_rag_eval_case_run") as save_result, \
             patch.object(api_app, "save_failed_rag_eval_case_run") as save_failed, \
             patch.object(api_app, "finish_rag_eval_run"), \
             patch.object(api_app, "update_rag_eval_experiment_summary"):
            result = api_app.run_rag_eval_case(1, api_app.RagEvalRunRequest())

        self.assertFalse(result["success"])
        self.assertEqual(result["cache_status"], "bypass")
        save_result.assert_not_called()
        save_failed.assert_called_once()

    def test_existing_history_reader_is_unchanged(self):
        rows = [{"ID": 1, "系统回答": "历史回答"}]
        with patch.object(api_app, "load_rag_eval_results", return_value=rows) as load:
            result = api_app.get_rag_eval_results(limit=10)

        load.assert_called_once_with(limit=10)
        self.assertTrue(result["success"])
        self.assertEqual(result["data"], rows)

    def test_streamlit_page_calls_the_existing_bypassed_eval_route(self):
        source = Path("main.py").read_text(encoding="utf-8")
        self.assertIn('f"/rag/eval/run/{case_id}"', source)
        self.assertIn('"/rag/eval/run_all"', source)
        self.assertIn("质量评测固定使用缓存旁路", source)


if __name__ == "__main__":
    unittest.main()
