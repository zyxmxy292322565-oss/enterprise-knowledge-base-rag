"""No network/index/database writes. HTTP dependency calls are explicitly mocked."""
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from policy_evidence_support import assess_policy_support, POLICY_EVIDENCE_CACHE_VERSION

CONTENT = ('适用范围：AirPods、无线耳机。签收后7天内，商品完好、配件齐全、不影响二次销售时可申请无理由退货。'
           '签收后30天内质量问题可申请检测；人为损坏需核实。申请不等于退款批准。具体保修期限以凭证为准。')

class SupportGuardTests(unittest.TestCase):
    def check(self, question, expected, content=CONTENT):
        result = assess_policy_support(question, [{'content':content}])
        self.assertIs(result['sufficient_for_answer'], expected, result)
        return result

    def test_airpods_normal(self): self.check('AirPods Pro 2 无理由退货的申请期限是什么？', True)
    def test_original_negative(self): self.check('月球采矿机器人核聚变引擎的终身免费退款承诺是什么？', False)
    def test_different_unknown_subject(self): self.check('海底勘探潜艇的退货政策是什么？', False)
    def test_mouse_cannot_inherit_earphone(self): self.check('无线鼠标故障可以套用耳机的退货政策吗？', False)
    def test_unsupported_shipping_promise(self): self.check('AirPods Pro 2 退货时商家保证承担往返运费吗？', False)
    def test_unconditional_promise(self): self.check('AirPods Pro 2可以无条件退款吗？', False)
    def test_lifetime_claim(self): self.check('耳机的终身退款规则是什么？', False)
    def test_generic_policy(self): self.check('退货政策是什么？', True)
    def test_ordinary_time_question(self): self.check('AirPods签收几天可以退？', True)
    def test_quality_path(self): self.check('耳机质量问题售后的申请期限是什么？', True)
    def test_engine_generated_query(self): self.check('耳机类商品签收后，因疑似质量问题申请return_refund的期限、条件及例外是什么？', True)
    def test_no_reason_generated_query(self): self.check('耳机类商品签收后，因不喜欢/无理由申请return_refund的期限、条件及例外是什么？', True)
    def test_sources_do_not_substitute_for_text(self):
        self.assertFalse(assess_policy_support('AirPods退货政策是什么？', [{'source':'AirPods.txt','content':'请联系客服'}])['sufficient_for_answer'])
    def test_cross_chunk_scope_and_conditions(self):
        self.assertTrue(assess_policy_support('AirPods质量问题售后期限是什么？', [{'content':'AirPods属于耳机。'}, {'content':'质量问题30天内可申请检测。'}])['sufficient_for_answer'])
    def test_negation_not_stripped_from_excerpts(self):
        evidence=[{'content':CONTENT}]
        assess_policy_support('AirPods退货政策是什么？', evidence)
        self.assertEqual(evidence, [{'content':CONTENT}])
    def test_empty(self): self.check('退货政策是什么？', False, '')
    def test_new_cache_namespace(self):
        self.assertNotEqual(POLICY_EVIDENCE_CACHE_VERSION, 'evidence-version-v2')

class SupportHTTPTests(unittest.TestCase):
    def query(self, question, *, content=CONTENT, score=.29, error=False):
        import numpy as np
        import api_app
        from fastapi.testclient import TestClient
        index=MagicMock(); index.search.return_value=(np.array([[.1]]),np.array([[0]]))
        embed=MagicMock(); embed.encode.return_value=np.array([[.1,.2]])
        rerank=MagicMock(); rerank.predict.return_value=np.array([score])
        if error: rerank.predict.side_effect=RuntimeError('controlled reranker failure')
        doc=SimpleNamespace(page_content=content,metadata={'source':'earphone.txt','file_hash':'hash','chunk_id':1})
        with ExitStack() as stack:
            for key,value in {'index':index,'vectors':np.ones((1,2)),'all_docs':[doc],'embed_model':embed,'rerank_model':rerank}.items():
                stack.enter_context(patch.object(api_app.app.state,key,value,create=True))
            stack.enter_context(patch.object(api_app,'get_cache',return_value=None))
            cache=stack.enter_context(patch.object(api_app,'set_cache'))
            response=TestClient(api_app.app).post('/rag/query',json={'question':question})
        self.assertEqual(response.status_code,200)
        return response.json(), cache

    def test_high_score_unknown_is_not_support(self):
        r,cache=self.query('月球采矿机器人核聚变引擎的终身免费退款承诺是什么？',score=.99)
        self.assertTrue(r['success'])  # retrieval completed, business refusal
        self.assertEqual(r['code'],'POLICY_NOT_FOUND');self.assertTrue(r['refused'])
        self.assertEqual(r['evidence'],[]);self.assertEqual(r['evidence_assessment']['candidate_count'],1)
        self.assertFalse(r['evidence_assessment']['sufficient_for_answer']);cache.assert_not_called()
    def test_low_but_passing_score_relevant_remains_usable(self):
        r,cache=self.query('AirPods Pro 2无理由退货的申请期限是什么？')
        self.assertEqual(r['code'],'SUCCESS');self.assertTrue(r['evidence_assessment']['sufficient_for_answer'])
        self.assertTrue(r['evidence']);self.assertIn('lexical-support-v1',cache.call_args.kwargs['key'])
    def test_reranker_fault_is_not_business_refusal(self):
        r,_=self.query('AirPods退货政策是什么？',error=True)
        self.assertFalse(r['success']);self.assertEqual(r['code'],'RAG_QUERY_FAILED')
    def test_missing_candidate_is_normal_refusal(self):
        r,_=self.query('AirPods退货政策是什么？',score=.01)
        self.assertTrue(r['success']);self.assertEqual(r['code'],'POLICY_NOT_FOUND')

if __name__ == '__main__': unittest.main()
