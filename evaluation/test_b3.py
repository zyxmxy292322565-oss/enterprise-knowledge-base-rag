import ast
import importlib.util
import json
import os
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

    def test_public_route_defaults_and_cache_path(self):
        with patch.object(api_app.app.state, 'all_docs', self.docs, create=True), \
             patch.object(api_app.app.state, 'vectors', np.ones((2, 2)), create=True), \
             patch.object(api_app.app.state, 'index', self.index, create=True), \
             patch.object(api_app, 'get_cache', return_value={'answer': 'cached', 'sources': []}) as read, \
             patch.object(api_app, 'incr_cache_stat'), patch.object(api_app, 'save_rag_request_log'):
            result = api_app.ask_question(api_app.AskRequest(question='问题'))
        read.assert_called_once()
        self.assertTrue(result['from_cache'])
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


class EvaluationStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        values = {'EVALUATION_ISOLATED': '1', 'EVALUATION_CACHE_DIR': '/app/cache',
                  'EVALUATION_MYSQL_HOST': 'mysql', 'EVALUATION_REDIS_HOST': 'redis',
                  'MYSQL_HOST': 'mysql', 'MYSQL_PORT': '3306', 'MYSQL_DATABASE': 'rag_evaluation',
                  'MYSQL_USER': 'rag_eval', 'REDIS_HOST': 'redis', 'REDIS_PORT': '6379',
                  'REDIS_DB': '0', 'MYSQL_PASSWORD': 'test', 'DEEPSEEK_API_KEY': 'test'}
        original_resolve = Path.resolve
        def resolve(path, *args, **kwargs):
            if path == Path('/app'): return Path('/app')
            if path == Path('cache'): return Path('/app/cache')
            return original_resolve(path, *args, **kwargs)
        spec = importlib.util.spec_from_file_location('isolated_evaluation_test', Path(__file__).parent / 'evaluation_app.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, values, clear=True), patch.object(Path, 'cwd', return_value=Path('/app')), patch.object(Path, 'resolve', resolve):
            spec.loader.exec_module(module)
        cls.module = module

    def test_experiment_run_case_association_and_duplicate_rejected(self):
        module = self.module
        experiments, runs, cases = {}, {}, {}
        def execute(sql, params=(), fetch=False):
            if sql.startswith('INSERT INTO evaluation_experiments'):
                experiments[params[0]] = params[1]
            elif sql.startswith('SELECT id FROM evaluation_experiments'):
                return (params[0],) if params[0] in experiments else None
            elif sql.startswith('INSERT INTO evaluation_runs'):
                runs[params[0]] = params[1]
            elif sql.startswith('SELECT e.id'):
                exp = runs.get(params[0])
                return (exp, experiments[exp]) if exp else None
            elif sql.startswith('SELECT id FROM evaluation_case_runs'):
                return ('exists',) if params in cases else None
            elif sql.startswith('INSERT INTO evaluation_case_runs'):
                cases[(params[1], params[2])] = json.loads(params[3])
            else:
                raise AssertionError(sql)
        def ask(request, *, evaluation_options, trace):
            trace['status'] = 'answered'
            return {'answer': 'answer', 'timing': {}}
        manifest = {'index_files_sha256': {'index': 'digest'}}
        with patch.object(module, '_execute', side_effect=execute), \
             patch.object(module, 'snapshot_manifest', return_value=manifest), \
             patch.object(module.api_app, '_ask_question', side_effect=ask):
            experiment = module.create_experiment(module.ExperimentRequest(dataset_version='reviewed-v1'))
            run = module.create_run(experiment['experiment_id'])
            record = module.run_case(run['run_id'], module.CaseRequest(case_id='case-1', question='question'))
            self.assertEqual(record['experiment_id'], experiment['experiment_id'])
            self.assertEqual(record['run_id'], run['run_id'])
            self.assertEqual(cases[(run['run_id'], 'case-1')]['dataset_version'], 'reviewed-v1')
            with self.assertRaises(module.HTTPException) as error:
                module.run_case(run['run_id'], module.CaseRequest(case_id='case-1', question='question'))
            self.assertEqual(error.exception.status_code, 409)

    def test_missing_or_formal_hosts_fail_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'isolation'):
                self.module.validate_isolation()


if __name__ == '__main__':
    unittest.main()
