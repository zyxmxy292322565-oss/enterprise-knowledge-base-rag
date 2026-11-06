import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import api_app


class EvaluationCoreTests(unittest.TestCase):
    def setUp(self):
        self.docs = [SimpleNamespace(page_content='政策证据一', metadata={'source': 'one.txt', 'file_hash': 'a', 'chunk_id': 1}),
                     SimpleNamespace(page_content='政策证据二', metadata={'source': 'two.txt', 'file_hash': 'b', 'chunk_id': 1})]
        self.index = MagicMock()
        self.index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        self.embed = MagicMock()
        self.embed.encode.return_value = np.array([[1.0, 2.0]])
        self.rerank = MagicMock()
        self.rerank.predict.return_value = np.array([0.3, 0.9])
        self.llm = MagicMock()
        self.llm.invoke.return_value = SimpleNamespace(content='依据政策回答')

    def test_evaluation_bypasses_all_cache_and_logs_and_traces_real_candidates(self):
        trace = {}
        with patch.object(api_app.app.state, 'all_docs', self.docs, create=True), \
             patch.object(api_app.app.state, 'vectors', np.ones((2, 2)), create=True), \
             patch.object(api_app.app.state, 'index', self.index, create=True), \
             patch.object(api_app.app.state, 'embed_model', self.embed, create=True), \
             patch.object(api_app.app.state, 'rerank_model', self.rerank, create=True), \
             patch.object(api_app.app.state, 'chat_model', self.llm, create=True), \
             patch.object(api_app, 'get_cache', side_effect=AssertionError('cache read')) as read, \
             patch.object(api_app, 'set_cache', side_effect=AssertionError('cache write')) as write, \
             patch.object(api_app, 'incr_cache_stat', side_effect=AssertionError('cache stats')) as stats, \
             patch.object(api_app, 'save_qa_logs', side_effect=AssertionError('qa log')) as qa, \
             patch.object(api_app, 'save_rag_request_log', side_effect=AssertionError('request log')) as perf:
            result = api_app._ask_question(api_app.AskRequest(question='问题', top_k=1),
                evaluation_options={'candidate_k': 2, 'rerank_enabled': True}, trace=trace)
        self.assertEqual(result['answer'], '依据政策回答')
        self.assertEqual(trace['status'], 'answered')
        self.assertEqual([x['faiss_index'] for x in trace['faiss_candidates']], [0, 1])
        self.assertEqual([x['faiss_index'] for x in trace['ranked_candidates']], [1, 0])
        self.assertEqual(trace['used_indices'], [1])
        self.assertIn('政策证据二', trace['context'])
        self.assertEqual(self.index.search.call_args.args[1], 2)
        for mock in (read, write, stats, qa, perf):
            mock.assert_not_called()

    def test_rerank_off_keeps_faiss_order_and_does_not_use_threshold(self):
        trace = {}
        with patch.object(api_app.app.state, 'all_docs', self.docs, create=True), \
             patch.object(api_app.app.state, 'vectors', np.ones((2, 2)), create=True), \
             patch.object(api_app.app.state, 'index', self.index, create=True), \
             patch.object(api_app.app.state, 'embed_model', self.embed, create=True), \
             patch.object(api_app.app.state, 'rerank_model', self.rerank, create=True), \
             patch.object(api_app.app.state, 'chat_model', self.llm, create=True):
            result = api_app._ask_question(api_app.AskRequest(question='问题', top_k=1, min_rerank_score=999),
                evaluation_options={'candidate_k': 2, 'rerank_enabled': False}, trace=trace)
        self.assertEqual(result['answer'], '依据政策回答')
        self.rerank.predict.assert_not_called()
        self.assertEqual(trace['used_indices'], [0])
        self.assertIsNone(trace['ranked_candidates'][0]['rerank_score'])

    def test_public_route_defaults_and_legacy_cache_bypass(self):
        with patch.object(api_app.app.state, 'all_docs', self.docs, create=True), \
             patch.object(api_app.app.state, 'vectors', np.ones((2, 2)), create=True), \
             patch.object(api_app.app.state, 'index', self.index, create=True), \
             patch.object(api_app, 'get_cache', return_value={'answer': 'cached', 'sources': []}) as read, \
             patch.object(api_app, 'incr_cache_stat'), patch.object(api_app, 'save_rag_request_log'):
            result = api_app.ask_question(api_app.AskRequest(question='问题'))
        read.assert_not_called()
        self.assertFalse(result.get('from_cache', False))
        self.assertEqual(api_app.AskRequest(question='x').top_k, 3)
        self.assertEqual(api_app.PolicyRAGRequest(question='x').faiss_multiplier, 3)

    def test_evaluation_options_are_not_public_ask_parameters(self):
        route = next(r for r in api_app.app.routes if getattr(r, 'path', '') == '/ask')
        self.assertEqual([p.name for p in route.dependant.query_params], [])

    def test_policy_evaluation_bypasses_cache_and_filters_each_candidate(self):
        trace = {}
        with patch.object(api_app.app.state, 'all_docs', self.docs, create=True), \
             patch.object(api_app.app.state, 'vectors', np.ones((2, 2)), create=True), \
             patch.object(api_app.app.state, 'index', self.index, create=True), \
             patch.object(api_app.app.state, 'embed_model', self.embed, create=True), \
             patch.object(api_app.app.state, 'rerank_model', self.rerank, create=True), \
             patch.object(api_app, 'get_cache', side_effect=AssertionError('cache read')) as read, \
             patch.object(api_app, 'set_cache', side_effect=AssertionError('cache write')) as write, \
             patch.object(api_app, 'incr_cache_stat', side_effect=AssertionError('cache stats')) as stats:
            result = api_app._query_policy_rag(api_app.PolicyRAGRequest(question='问题', top_k=2, min_rerank_score=0.5),
                evaluation_options={'candidate_k': 2, 'rerank_enabled': True}, trace=trace)
        self.assertTrue(result['success'])
        self.assertEqual([x['source'] for x in result['evidence']], ['two.txt'])
        self.assertEqual([x['faiss_index'] for x in trace['ranked_candidates']], [1, 0])
        for mock in (read, write, stats): mock.assert_not_called()

    def test_low_score_refusal_does_not_call_llm(self):
        trace = {}
        with patch.object(api_app.app.state, 'all_docs', self.docs, create=True), \
             patch.object(api_app.app.state, 'vectors', np.ones((2, 2)), create=True), \
             patch.object(api_app.app.state, 'index', self.index, create=True), \
             patch.object(api_app.app.state, 'embed_model', self.embed, create=True), \
             patch.object(api_app.app.state, 'rerank_model', self.rerank, create=True), \
             patch.object(api_app.app.state, 'chat_model', self.llm, create=True):
            result = api_app._ask_question(api_app.AskRequest(question='问题', top_k=1, min_rerank_score=2),
                evaluation_options={'candidate_k': 2, 'rerank_enabled': True}, trace=trace)
        self.assertEqual(trace['status'], 'refused_low_relevance')
        self.assertEqual(result['sources'], [])
        self.llm.invoke.assert_not_called()
