from unittest import TestCase
from unittest.mock import Mock, patch

from reliability import PolicyRule, ReliabilityOptions, control, route, validate
from retrieval_engine import ChatTurn


class ReliabilityTests(TestCase):
    def test_routes_simple_contextual_and_complex(self):
        history = [ChatTurn(role="user", content="AX-1 能退吗？")]
        self.assertEqual(route("如何退款？", [], ReliabilityOptions())[1]["query_type"], "simple")
        self.assertTrue(route("那超过7天呢？", history, ReliabilityOptions())[0].query_rewrite_enabled)
        self.assertTrue(route("退款和运费？", [], ReliabilityOptions())[0].query_decomposition_enabled)

    def test_scope_conditions_dates_and_exceptions(self):
        rule = PolicyRule(fact="退款", scope="AX", verdict="不直接批准", when={"核验": "通过"},
                          unless={"损坏": "是"}, starts="2026-01-01")
        claims, _, missing, _ = validate(["退款"], [(rule, {"child_id": "A"})], ReliabilityOptions(scope="AX"))
        self.assertEqual(claims, [])
        self.assertEqual(set(missing), {"核验", "损坏", "适用日期"})
        for scope, day, damage in [("BX", "2026-06-01", "否"), ("AX", "2025-06-01", "否"),
                                    ("AX", "2026-06-01", "是")]:
            self.assertFalse(validate(["退款"], [(rule, {"child_id": "A"})], ReliabilityOptions(
                scope=scope, as_of=day, slots={"核验": "通过", "损坏": damage}))[0])
        claims = validate(["退款"], [(rule, {"child_id": "A"})], ReliabilityOptions(
            scope="AX", as_of="2026-06-01", slots={"核验": "通过", "损坏": "否"}))[0]
        self.assertEqual(claims[0]["statement"], "不直接批准")

    def test_unavailable_retrieval_stops_and_does_not_answer(self):
        view = Mock(snapshot_id="index", corpus_snapshot_id="corpus", generation=1)
        with patch("reliability.retrieve", side_effect=RuntimeError) as retrieve:
            answer = control(None, view, "退款规则是什么？", [], ReliabilityOptions(scope="AX"), None, None)
            self.assertEqual(answer["answer_action"], "ABSTAIN")
            self.assertEqual(retrieve.call_count, 2)
            self.assertEqual(answer["retrieval_attempts"][1]["error"], "RuntimeError")

    def test_invalid_evidence_cannot_be_cited(self):
        view = Mock(snapshot_id="index", corpus_snapshot_id="corpus", generation=1)
        result = Mock(candidates=[{"faiss_index": 0, "doc": Mock(page_content="RULE fake", metadata={"child_id": "A"})}],
                      trace={})
        with patch("reliability.retrieve", return_value=result), patch("reliability.rescore", return_value=([], False)), \
             patch("reliability.verified_rules", side_effect=ValueError):
            answer = control(None, view, "退款？", [], ReliabilityOptions(scope="AX"), None, None)
        self.assertEqual(answer["evidence_validation"]["status"], "INVALID")
        self.assertEqual(answer["citation_validation"]["claims"], [])

    def test_exhausted_wall_budget_never_calls_retrieval(self):
        view = Mock(snapshot_id="index", corpus_snapshot_id="corpus", generation=1)
        with patch("reliability.perf_counter", side_effect=[0, 1, 1, 1]), patch("reliability.retrieve") as retrieve:
            response = control(None, view, "退款？", [], ReliabilityOptions(timeout_budget_ms=50), None, None)
        retrieve.assert_not_called()
        self.assertTrue(response["budget"]["timed_out"])
