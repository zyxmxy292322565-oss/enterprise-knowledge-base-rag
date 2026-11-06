import json
import unittest
from unittest.mock import patch

from knowledge_graph import Assertion, Entity, conflicts_for, entity_id, extract
from reliability import ReliabilityOptions, route
from retrieval_engine import RetrievalConfig, fuse


class KnowledgeGraphTests(unittest.TestCase):
    def test_conflict_requires_same_conditions_and_overlapping_time(self):
        left = dict(relation_id="a", source_entity_id="product", target_entity_id="action",
                    relation_type="ALLOWS", polarity="affirmed", conditions=["quality"],
                    valid_from="2026-01-01", valid_to="2026-06-30")
        right = left | {"relation_id": "b", "polarity": "negated"}
        self.assertEqual(conflicts_for([left, right]), [["a", "b"]])
        self.assertEqual(conflicts_for([left, right | {"conditions": ["unopened"]}]), [])
        self.assertEqual(conflicts_for([left, right | {"valid_from": "2026-07-01", "valid_to": None}]), [])
        self.assertEqual(conflicts_for([left, right | {"relation_type": "SUPERSEDES"}]), [])

    def test_assertion_schema_rejects_nonfinite_confidence(self):
        entity = {"type": "Product", "name": "Mouse", "scope": "policy"}
        with self.assertRaises(ValueError):
            Assertion.model_validate({"source": entity, "target": entity, "predicate": "ALLOWS",
                                      "confidence": float("nan")})
    def test_scope_sensitive_normalization(self):
        def key(name, scope):
            return entity_id(Entity(type="Product", name=name, scope=scope))
        self.assertEqual(key("ＡＸ-９００", "a"), key("ax-900", "a"))
        self.assertNotEqual(key("AX-900", "a"), key("AX-900", "b"))

    def test_schema_and_narrow_natural_extraction(self):
        self.assertEqual(next(extract("ELIGIBLE_TO_APPLY 不表示退款已批准")).predicate, "EXCLUDES")
        self.assertFalse(list(extract("可提出申请")))
        with self.assertRaises(ValueError):
            list(extract("KG " + json.dumps({"predicate": "unsupported"})))

    def test_seven_modes_and_graph_rrf_dedup(self):
        for mode, dense, sparse in (("dense", True, False), ("sparse", False, True), ("hybrid", True, True),
                                     ("graph", False, False)):
            for graph in ([True] if mode == "graph" else [False, True]):
                RetrievalConfig(retrieval_mode=mode, dense_enabled=dense, bm25_enabled=sparse, graph_enabled=graph)
        result = fuse([{"id": ch, "family": ch, "status": "used", "results": [{"row": 2, "rank": 1}]}
                       for ch in ("dense", "bm25", "graph")], 60, True)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["rrf_score"], 3 / 61)

    def test_router_explicit_and_selective(self):
        self.assertFalse(route("退款是什么？", [], ReliabilityOptions(graph_enabled=True))[0].graph_enabled)
        self.assertFalse(route("AX-900 适用什么条件？", [], ReliabilityOptions())[0].graph_enabled)
        self.assertTrue(route("AX-900 适用什么条件？", [], ReliabilityOptions(graph_enabled=True))[0].graph_enabled)
