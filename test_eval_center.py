from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock

from pydantic import ValidationError

from eval_center_api import RunConfig, validate_draft_dataset
from eval_center_data import historical_dataset
from eval_center_metrics import answer_metrics, covered_span, ranking, retrieval_metrics, summarize
from eval_center_runner import load_evaluation_snapshot
from knowledge_v2 import KnowledgeError


class DraftEvaluationTests(TestCase):
    def setUp(self):
        self.spec = dict(dataset_id="dataset", dataset_version="v1", corpus_snapshot_id="corpus",
                         index_snapshot_id="snapshot", publish_generation=0, snapshot_mode="draft_validation",
                         mode="retrieval", candidate_k=8, final_k=3)

    def test_generation_zero_is_exclusive_to_draft_validation_and_graph_stays_disabled(self):
        self.assertEqual(RunConfig(**self.spec).publish_generation, 0)
        self.assertEqual(RunConfig(**(self.spec | {"snapshot_mode": "published", "publish_generation": 1})).snapshot_mode,
                         "published")
        for changes in ({"snapshot_mode": "published"}, {"publish_generation": 1}, {"graph_enabled": True}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                RunConfig(**(self.spec | changes))

    def test_draft_requires_bound_validation_dataset_with_reviewed_evidence(self):
        reviewed = {"review_status": "developer_self_reviewed"}
        row = {"split": "validation", "body": {"provenance": {"snapshot_id": "snapshot"},
               "questions": [reviewed], "evidence": [reviewed]}}
        validate_draft_dataset(row, "snapshot")
        for change in ({"split": "holdout"}, {"body": row["body"] | {"evidence": [
                {"review_status": "proposed_unreviewed"}]}}):
            with self.subTest(change=change), self.assertRaises(KnowledgeError):
                validate_draft_dataset(row | change, "snapshot")
        with self.assertRaises(KnowledgeError):
            validate_draft_dataset(row, "another")

    def test_draft_worker_rejects_changed_manifest_before_load(self):
        store = Mock()
        store.snapshot_status.return_value = {"manifest_sha": "original"}
        config = self.spec | {"snapshot_manifest_sha256": "original"}
        self.assertIs(load_evaluation_snapshot(store, config), store.load_snapshot.return_value)
        store.load_snapshot.assert_called_once_with("snapshot", strict_sparse=True)
        store.load_snapshot.reset_mock()
        store.snapshot_status.return_value = {"manifest_sha": "changed"}
        with self.assertRaises(KnowledgeError):
            load_evaluation_snapshot(store, config)
        store.load_snapshot.assert_not_called()


class EvaluationMetricsTests(TestCase):
    def test_s1a_worked_ranking_example(self):
        q1 = ranking(["X", "A", "A", "B"], ["X", "A", "A"], {"A", "B"}, 4, 3, "")
        q2 = ranking(["C"], ["C"], {"C"}, 4, 3, "")
        excluded = ranking([], [], set(), 4, 3, "no gold")
        self.assertEqual(q1["recall"]["value"], 1)
        self.assertEqual(q1["final_recall"]["value"], 0.5)
        self.assertAlmostEqual(q1["ndcg"]["value"], 0.38685, places=5)
        self.assertEqual(q1["mrr"]["value"], 0.5)
        self.assertEqual(q1["repeat_rate"]["value"], 1 / 3)
        self.assertIsNone(excluded["recall"]["value"])
        self.assertEqual(summarize([q1, q2, excluded])["final_recall"],
                         {"macro": 0.75, "micro": 2 / 3, "eligible": 2, "planned": 3})

    def test_joint_visible_spans(self):
        self.assertTrue(covered_span(2, 9, [(6, 9), (1, 7)]))
        self.assertFalse(covered_span(2, 9, [(7, 9), (1, 6)]))

    def test_partially_annotated_gold_is_ineligible(self):
        question = {"answerability": "answerable", "review_status": "developer_self_reviewed",
                    "required_facts": [{"purpose": "answer", "evidence_options": [["E1"]]},
                                       {"purpose": "answer", "evidence_options": []}]}
        evidence = [{"evidence_id": "E1", "role": "answer_support", "source_path": "policy.txt",
                     "source_content_hash": "hash", "review_status": "developer_self_reviewed"}]
        scores = retrieval_metrics(question, evidence, {}, [], [], {"candidate_k": 10, "final_k": 3})
        self.assertIsNone(scores["file_recall"]["value"])
        self.assertIsNone(scores["required_fact_evidence_coverage"]["value"])

    def test_semantic_labels_do_not_treat_valid_citation_as_support(self):
        question = {"expected_action": "answer", "required_facts": [{"fact_id": "F1", "purpose": "answer"}],
                    "forbidden_claims": [{"claim_id": "FC1"}]}
        review = {"actual_action": "answer", "facts": {"F1": "covered"}, "forbidden_violation": True,
                  "false_answer": True, "target_conclusion": True, "action_complete": True,
                  "claims": [{"status": s} for s in ("supported", "contradicted", "uncertain")],
                  "citations": [{"claim_id": str(i), "citation_id": str(i), "valid": True, "supported": label}
                                for i, label in enumerate((True, False))]}
        metrics = answer_metrics(question, review)
        self.assertEqual(metrics["unsupported_claim_rate"]["value"], 0.5)
        self.assertEqual(metrics["citation_valid_rate"]["value"], 1)
        self.assertEqual(metrics["citation_support_rate"]["value"], 0.5)
        self.assertIsNone(metrics["correct_clarification_rate"]["value"])

    def test_historical_copy_remains_unreviewed(self):
        path = Path(__file__).parent / "evaluation/datasets/policy_docs_v3_draft/dataset.jsonl"
        body = historical_dataset(path.read_bytes())
        self.assertEqual((len(body["questions"]), len(body["evidence"])), (66, 226))
        self.assertEqual({e["review_status"] for e in body["evidence"]}, {"proposed_unreviewed"})
        self.assertFalse(any(f["evidence_options"] for q in body["questions"] for f in q["required_facts"]))
