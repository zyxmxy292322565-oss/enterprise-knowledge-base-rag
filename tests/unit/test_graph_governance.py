from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import time

from agent_router import Advice, plan_router
from graph_governance import policy_decisions, assess_policies
from knowledge_graph import alias_matches, projected_paths
from reliability import ReliabilityOptions
from retrieval_engine import RetrievalConfig, retrieve


class GovernanceTests(TestCase):
    def test_explicit_graph_budget_preserves_default_and_upper_bound(self):
        from pydantic import ValidationError
        self.assertEqual(RetrievalConfig().graph_timeout_ms,1000)
        self.assertEqual(ReliabilityOptions().graph_timeout_ms,1000)
        with self.assertRaises(ValidationError):ReliabilityOptions(graph_timeout_ms=5001)
        with self.assertRaises(ValidationError):RetrievalConfig(graph_timeout_ms=5001)
        from reliability import route
        config,_=route('关联政策',[],ReliabilityOptions(graph_timeout_ms=3000))
        self.assertEqual(config.graph_timeout_ms,3000)
    def test_graph_subqueries_have_bounded_group_time_and_share_request_deadline(self):
        from test_retrieval_engine import fixture
        def graph(view, question, hops, deadline=None, **kwargs):
            time.sleep(.36)
            if time.perf_counter() >= deadline:
                raise TimeoutError('request deadline')
            return {'paths':[], 'timed_out':False, 'conflicts':[]}
        view, embedder = fixture()
        view.graph_lookup = graph
        config = RetrievalConfig(retrieval_mode='graph',dense_enabled=False,bm25_enabled=False,
                                 graph_enabled=True,query_decomposition_enabled=True)
        with patch.dict('os.environ', {'RAG_V3_GRAPH_ENABLED':'1'}):
            result = retrieve(view,'退款；运费',[],config,3,embedder,
                deadline=time.perf_counter()+3, tool_budget=[10], suggested_subqueries=['退款','运费'])
            routes = [row for row in result.trace['routes'] if row['family']=='graph']
            self.assertEqual(len(routes),3)
            self.assertTrue(all(row['status']=='used' for row in routes))
            result = retrieve(view,'退款；运费',[],config,3,embedder,
                deadline=time.perf_counter()+.6, tool_budget=[10], suggested_subqueries=['退款','运费'])
            self.assertTrue(any(row['status']!='used' for row in result.trace['routes']))
    def test_router_builtin_field_uses_structured_context(self):
        result = assess_policies([], [], '2026-11-05', {
            'product':'AX-DIR','scope':'demo','slots':{'核验':'是'},
            'required_conditions':['product','scope','核验']})
        self.assertEqual(result['missing_slots'], [])
        self.assertEqual(result['action'], 'ABSTAIN')
        missing = assess_policies([], [], '2026-11-05', {'required_conditions':['product']})
        self.assertEqual(missing['missing_slots'], ['product'])
    def test_similar_model_suffix_is_not_same_seed(self):
        self.assertTrue(alias_matches("ax-123","ax-123退款"))
        self.assertFalse(alias_matches("ax-123","ax-123-office退款"))
    def test_canonical_duplicate_retains_both_original_proofs(self):
        edge = {"source_entity_id":"a","target_entity_id":"b","relation_type":"ALLOWS"}
        paths = projected_paths([(["a","b"],[edge | {"relation_id":"r1"}]),
                                 (["a","b"],[edge | {"relation_id":"r2"}])],
                                {"r1":[{"chunk_id":"c1"}],"r2":[{"chunk_id":"c2"}]})
        self.assertEqual(len(paths),1)
        self.assertEqual(paths[0]["equivalent_relation_paths"],[["r2"]])
        self.assertEqual(paths[0]["evidence_chunk_ids"],["c1","c2"])
    def policy(self, key, **overrides):
        return {"relation_id":key, "metadata":dict(valid_from="2020-01-01",valid_until=None,
            applicable_products=[],applicable_categories=[],scope="mouse",conditions={},exceptions={},
            policy_status="ACTIVE",supersedes_relation_id=None) | overrides}

    def test_history_explicit_supersession_and_unrelated_scope(self):
        old = self.policy("old")
        new = self.policy("new", valid_from="2024-01-01",supersedes_relation_id="old")
        unrelated = self.policy("other",applicable_products=["other"])
        self.assertEqual(policy_decisions([old,new,unrelated],"2022-01-01",{"product":"a","scope":"mouse"}),
                         {"old":"APPLICABLE","new":"TIME_DIFFERENCE","other":"SCOPE_DIFFERENCE"})
        self.assertEqual(policy_decisions([old,new],"2025-01-01")["old"],"SUPERSEDED")
        self.assertEqual(policy_decisions([old,self.policy("new")],"2025-01-01")["old"],"APPLICABLE")

    def test_policy_exceptions_are_not_implicitly_satisfied(self):
        rule=self.policy("r", conditions={"verified":"yes"},exceptions={"broken":"yes"})
        self.assertEqual(policy_decisions([rule],"2025-01-01",{"scope":"mouse","slots":{"verified":"yes"}})["r"],"CONDITION_DIFFERENCE")
        self.assertEqual(policy_decisions([rule],"2025-01-01",{"scope":"mouse","slots":{"verified":"yes","broken":"no"}})["r"],"APPLICABLE")

    def test_router_schema_rejects_untrusted_tools_and_falls_back_without_config(self):
        with self.assertRaises(ValueError):
            Advice.model_validate(dict(query_type="simple",complexity="low",selected_channels=["SQL"],routing_reason="x"))
        with patch.dict("os.environ",{"RAG_V3_ROUTER_ENABLED":"0"}):
            strategy,plan,cost=plan_router("退款是什么？",[],ReliabilityOptions(model_router=True,graph_enabled=True),float("inf"))
        self.assertFalse(strategy.graph_enabled)
        self.assertEqual(plan["router_decision"]["status"],"fallback")
        self.assertEqual(cost,0)

    def test_zero_budget_skips_dense_channel(self):
        v=SimpleNamespace(all_docs=[object()],sparse=None,snapshot_id="s",corpus_snapshot_id="c",generation=1)
        with patch("retrieval_engine.bm25") as sparse:
            result=retrieve(v,"q",[],RetrievalConfig(),3,None,tool_budget=[0])
        sparse.assert_not_called()
        self.assertEqual(result.trace["routes"][0]["reason"],"tool_budget")
