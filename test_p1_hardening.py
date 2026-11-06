from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest import TestCase
from graph_governance import assess_policies
from request_budget import ExecutionBudget, BudgetExceeded, cancel_signal
from reliability import ReliabilityOptions

def rule(key,**changes):
    meta=dict(valid_from="2020-01-01",valid_until=None,policy_status="ACTIVE",scope="s",
              applicable_products=[],applicable_categories=[],conditions={},exceptions={},supersedes_relation_id=None)
    meta.update(changes)
    return dict(relation_id=key,policy_id="family-"+key,policy_version="v1",metadata=meta)

class PolicyAssessmentTests(TestCase):
    def test_missing_and_mismatched_conditions_are_different(self):
        r=rule("a",conditions={"verified":"yes"},exceptions={"damaged":"yes"})
        self.assertEqual(assess_policies([r],[],"2026-01-01",{"scope":"s","slots":{}})["action"],"CLARIFY")
        self.assertEqual(assess_policies([r],[],"2026-01-01",{"scope":"s","slots":{"verified":"no"}})["action"],"ABSTAIN")
        self.assertEqual(assess_policies([r],[],"2026-01-01",{"scope":"s","slots":{"verified":"yes","damaged":"no"}})["action"],"ANSWER")
        self.assertEqual(assess_policies([r],[],"2026-01-01",{"scope":"s","slots":{"verified":"yes","damaged":"yes"}})["action"],"ABSTAIN")

    def test_time_product_scope_and_empty_evidence(self):
        for row,context in [(rule("a",valid_until="2021-01-01"),{"scope":"s"}),(rule("a",applicable_products=["A"]),{"scope":"s","product":"B"}),(rule("a"),{"scope":"other"})]:
            self.assertEqual(assess_policies([row],[],"2026-01-01",context)["action"],"ABSTAIN")
        self.assertEqual(assess_policies([],[],"2026-01-01",{})["action"],"ABSTAIN")

    def test_conflict_and_unresolved_versions(self):
        a,b=rule("a"),rule("b")
        edges=[dict(relation_id="a",target_entity_id="action",relation_type="ALLOWS"),dict(relation_id="b",target_entity_id="action",relation_type="EXCLUDES")]
        result=assess_policies([a,b],edges,"2026-01-01",{"scope":"s"})
        self.assertEqual(result["conflicts"][0]["classification"],"TRUE_CONFLICT")
        b.update(policy_id=a["policy_id"],policy_version="v2")
        self.assertEqual(assess_policies([a,b],edges,"2026-01-01",{"scope":"s"})["conflicts"][0]["classification"],"UNRESOLVED")
        b["metadata"]["supersedes_relation_id"]="a"
        self.assertEqual(assess_policies([a,b],edges,"2026-01-01",{"scope":"s"})["action"],"ANSWER")

class RequestBudgetTests(TestCase):
    def test_atomic_parallel_limit_and_no_refund_after_failure(self):
        budget=ExecutionBudget(ReliabilityOptions(max_tool_calls=5))
        def consume(_):
            try:budget.acquire("retrieval","retrieval");return True
            except BudgetExceeded:return False
        with ThreadPoolExecutor(12) as pool:
            self.assertEqual(sum(pool.map(consume,range(40))),5)
        self.assertEqual(budget.used,5)

    def test_zero_model_graph_deadline_and_cancel(self):
        for options,kind in [(ReliabilityOptions(max_tool_calls=0),"retrieval"),(ReliabilityOptions(max_model_calls=0),"model"),(ReliabilityOptions(max_graph_queries=0),"graph")]:
            with self.assertRaises(BudgetExceeded):
                ExecutionBudget(options).acquire("stage",kind)
        budget=ExecutionBudget(ReliabilityOptions());budget.deadline=0
        with self.assertRaises(BudgetExceeded):budget.acquire("late","model")
        event=Event();event.set()
        token=cancel_signal.set(event)
        try:
            with self.assertRaises(BudgetExceeded):ExecutionBudget(ReliabilityOptions()).acquire("cancelled","retrieval")
        finally:cancel_signal.reset(token)

    def test_hyde_shares_request_budget(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from request_budget import current
        from retrieval_engine import retrieve, RetrievalConfig
        from test_retrieval_engine import fixture
        view, embedder=fixture()
        budget=ExecutionBudget(ReliabilityOptions(max_tool_calls=1))
        token=current.set(budget)
        generator=Mock(return_value=SimpleNamespace(content="synthetic"))
        try:
            retrieve(view,"query",[],RetrievalConfig(hyde_enabled=True),3,embedder,generator,tool_budget=[1])
        finally:current.reset(token)
        generator.assert_called_once()
        embedder.encode.assert_not_called()
        self.assertEqual((budget.used,budget.models),(1,1))

    def test_failed_call_cannot_be_retried_after_exhaustion(self):
        from request_budget import current, stage
        budget=ExecutionBudget(ReliabilityOptions(max_tool_calls=1))
        token=current.set(budget)
        try:
            with self.assertRaises(ValueError):
                with stage("first","model"):raise ValueError("injected")
            with self.assertRaises(BudgetExceeded):
                with stage("retry","model"):pass
        finally:current.reset(token)
        self.assertEqual(budget.used,1)

    def test_inapplicable_governed_policy_cannot_fall_through_to_general_rule(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock, patch
        from reliability import control
        found=NS(candidates=[{"faiss_index":0,"doc":Mock()}],trace={"graph":{"policy_assessment":{"action":"ABSTAIN","policy_count":1}}})
        view=NS(snapshot_id="s",corpus_snapshot_id="c",generation=1)
        with patch("reliability.retrieve",return_value=found),patch("reliability.verified_rules") as verify:
            result=control(None,view,"退款",[],ReliabilityOptions(),None,None)
        self.assertEqual(result["answer_action"],"ABSTAIN")
        verify.assert_not_called()

    def test_truncated_graph_cannot_be_treated_as_complete_evidence(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock, patch
        from reliability import control
        found=NS(candidates=[{"faiss_index":0,"doc":Mock()}],trace={"graph":{"truncated":True}})
        view=NS(snapshot_id="s",corpus_snapshot_id="c",generation=1)
        with patch("reliability.retrieve",return_value=found),patch("reliability.verified_rules") as verify:
            result=control(None,view,"退款",[],ReliabilityOptions(),None,None)
        self.assertEqual(result["answer_action"],"ABSTAIN")
        verify.assert_not_called()

    def test_semantic_generation_shares_model_limit_with_evidence_judge(self):
        import json
        from types import SimpleNamespace as NS
        from unittest.mock import Mock, patch
        from semantic_reliability import run_semantic
        view=NS(snapshot_id="s",corpus_snapshot_id="c",generation=1)
        found=NS(candidates=[{"faiss_index":0,"doc":NS(metadata={"child_id":"a"})}],trace={})
        payload={"action":"ANSWER","facts":[{"text":"退款","status":"supported","evidence_ids":["a"],"reason":"fixture"}],"missing_slots":[]}
        create=Mock(return_value=NS(model="local-fixture",usage=None,choices=[NS(message=NS(content=json.dumps(payload))) ]))
        client=Mock();client.__enter__=Mock(return_value=NS(chat=NS(completions=NS(create=create))));client.__exit__=Mock(return_value=False)
        config={"RAG_V2_SEMANTIC_ENABLED":"1","RAG_V2_SEMANTIC_GENERATOR":"draft","RAG_V2_SEMANTIC_JUDGE":"judge","RAG_V2_SEMANTIC_BASE_URL":"http://localhost","RAG_V2_SEMANTIC_API_KEY":"fixture"}
        with patch.dict("os.environ",config),patch("semantic_reliability.retrieve",return_value=found),patch("semantic_reliability.evidence_for",return_value={"quote":"fixture"}):
            result=run_semantic(None,view,"退款",[],ReliabilityOptions(facts=["退款"],token_budget=65536,max_model_calls=1),None,client_factory=Mock(return_value=client))
        self.assertEqual(result["answer_action"],"ABSTAIN")
        self.assertEqual(create.call_count,1)
        self.assertEqual(result["budget"]["execution"]["used_model_calls"],1)
        self.assertEqual(result["budget"]["execution"]["termination_reason"],"model_budget")

    def test_zero_legacy_route_budget_does_not_charge_unexecuted_hyde(self):
        from unittest.mock import Mock
        from request_budget import current
        from retrieval_engine import retrieve, RetrievalConfig
        from test_retrieval_engine import fixture
        view,embedder=fixture();generator=Mock()
        budget=ExecutionBudget(ReliabilityOptions(max_tool_calls=5));token=current.set(budget)
        try:retrieve(view,"q",[],RetrievalConfig(hyde_enabled=True),3,embedder,generator,tool_budget=[0])
        finally:current.reset(token)
        generator.assert_not_called();embedder.encode.assert_not_called()
        self.assertEqual(budget.used,0)

    def test_required_graph_failure_does_not_bypass_policy_gate(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from reliability import control
        found=NS(candidates=[],trace={"routes":[{"family":"graph","status":"failed","reason":"budget"}]})
        with patch("reliability.retrieve",return_value=found):
            result=control(None,NS(snapshot_id="s",corpus_snapshot_id="c",generation=1),"退款",[],ReliabilityOptions(),None,None)
        self.assertEqual(result["answer_action"],"ABSTAIN")

    def test_controller_does_not_renew_an_already_consumed_request_budget(self):
        from request_budget import current
        from reliability import control
        from test_retrieval_engine import fixture
        view,embedder=fixture()
        options=ReliabilityOptions(max_tool_calls=1,facts=["退款"])
        budget=ExecutionBudget(options);budget.acquire("router","model")
        token=current.set(budget)
        try:result=control(None,view,"退款",[],options,embedder,None)
        finally:current.reset(token)
        embedder.encode.assert_not_called()
        self.assertEqual(result["answer_action"],"ABSTAIN")
        self.assertEqual(result["budget"]["execution"]["used_tool_calls"],1)
