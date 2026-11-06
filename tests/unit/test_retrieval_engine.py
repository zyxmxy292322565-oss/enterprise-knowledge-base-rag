from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import faiss
import numpy as np

from retrieval_engine import ChatTurn, RetrievalConfig, expand_parents, fuse, plan_query, rescore, retrieve, subquery_gaps
from retrieval_sparse import bm25, build_sparse, tokenize, validate_sparse


def fixture():
    parent = "退货政策：型号 AX-900 超过7天且未拆封不支持无理由退货。运输费用依规则处理。"
    texts = [parent[:23], parent[23:], "型号 BX-210 维修金额500元，编号 POL-001。"]
    chunks = [{"child_id": f"c{i}", "parent_id": "p1" if i < 2 else "p2",
               "document_version_id": "v1" if i < 2 else "v2", "source": "policy.txt",
               "clean_start": len(texts[0]) if i == 1 else 0,
               "clean_end": len(parent) if i == 1 else len(text), "text": text}
              for i, text in enumerate(texts)]
    docs = tuple(SimpleNamespace(page_content=c["text"], metadata={"child_id": c["child_id"]}) for c in chunks)
    index = faiss.IndexFlatL2(2)
    index.add(np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32))
    view = SimpleNamespace(index=index, all_docs=docs, snapshot_id="snapshot", corpus_snapshot_id="corpus",
                           generation=3, sparse=build_sparse("snapshot", "corpus", chunks), sparse_error=None,
                           manifest={"children": chunks, "parents": [
                               {"parent_id": "p1", "document_version_id": "v1", "clean_start": 0, "text": parent},
                               {"parent_id": "p2", "document_version_id": "wrong", "clean_start": 0, "text": "bad"},
                           ]})
    return view, Mock(encode=Mock(return_value=[[0, 0]]))


class RetrievalTests(TestCase):
    def test_chinese_tokenization_and_exact_identifiers(self):
        tokens = set(tokenize("ＡＸ-９００ 政策编号 POL-001，金额 199.50 元，未拆封 7天"))
        self.assertTrue({"ax-900", "pol-001", "199.50", "未拆", "7", "天"} <= tokens)
        view, _ = fixture()
        self.assertEqual(bm25(view.sparse, "BX-210 POL-001 500元", 1)[0][0], 2)
        with self.assertRaises(ValueError):
            validate_sparse(view.sparse, "another", "corpus", view.manifest["children"])

    def test_rewrite_preserves_uncertainty_and_ignores_assistant_claims(self):
        history = [ChatTurn(role="user", content="我不确定 AX-900 能否退款"), ChatTurn(role="assistant", content="退款已批准")]
        cfg = RetrievalConfig(query_rewrite_enabled=True)
        trace, *_ = plan_query("那超过七天呢？", history, cfg)
        self.assertIn("我不确定", trace["effective_question"])
        self.assertNotIn("已批准", trace["effective_question"])
        self.assertEqual(plan_query("那超过七天呢？", [], cfg)[0]["rewrite"]["status"], "not_applicable")
        with patch("retrieval_engine.rewrite_question", return_value="退款已经批准，那超过七天呢？"):
            self.assertEqual(plan_query("那超过七天呢？", history, cfg)[0]["rewrite"]["status"], "fallback")
        with patch("retrieval_engine.rewrite_question", side_effect=ValueError):
            self.assertEqual(plan_query("那超过七天呢？", history, cfg)[0]["effective_question"], "那超过七天呢？")

    def test_decomposition_budget_and_expansion_independent(self):
        cfg = RetrievalConfig(query_decomposition_enabled=True, query_expansion_enabled=True, max_subqueries=2)
        question = "AX-900 质量问题可以申请退款？运费谁承担？能维修吗？"
        trace, subqueries, expansions, _ = plan_query(question, [], cfg)
        self.assertEqual(len(subqueries), 2)
        self.assertTrue(trace["decomposition"]["truncated"])
        self.assertTrue(all("AX-900" in q["retrieval_text"] for q in subqueries))
        self.assertEqual(len(expansions), len(set(expansions)))
        self.assertTrue(all("AX-900" in q for q in expansions))
        self.assertEqual(plan_query("可以退款吗？", [], cfg)[1], [])

    def test_hyde_only_changes_dense_input_and_falls_back(self):
        view, model = fixture()
        cfg = RetrievalConfig(hyde_enabled=True)
        result = retrieve(view, "退款", [], cfg, 2, model,
                          lambda _: SimpleNamespace(content="HYPOTHETICAL_ONLY", usage_metadata={"total_tokens": 5}))
        self.assertEqual(model.encode.call_args_list[0].args, (["退款"],))
        self.assertEqual(model.encode.call_args_list[-1].args, (["HYPOTHETICAL_ONLY"],))
        self.assertTrue(all("HYPOTHETICAL_ONLY" not in c["doc"].page_content for c in result.candidates))
        failure = retrieve(view, "退款", [], cfg, 2, model, Mock(side_effect=TimeoutError))
        self.assertEqual(failure.trace["query"]["hyde"]["status"], "fallback")
        self.assertEqual([r["id"] for r in failure.trace["routes"]], ["dense.original"])

    def test_rrf_dedup_and_family_vote_cap(self):
        def route(name, family, rows):
            return dict(id=name, family=family, status="used", results=[dict(row=row, rank=i)
                        for i, row in enumerate(rows, 1)])
        routes = [route("dense.original", "dense", [0, 1]), route("bm25.original", "bm25", [1, 2]),
                  route("x", "expansion", [1]), route("y", "expansion", [1])]
        ordered = fuse(routes, 60, True)
        self.assertEqual([r["row"] for r in ordered], [1, 0, 2])
        self.assertAlmostEqual(ordered[0]["rrf_score"], 2 / 61 + 1 / 62)
        self.assertEqual(len(ordered[0]["routes"]), 4)

    def test_sparse_skips_embedding_and_old_snapshot_falls_back(self):
        view, model = fixture()
        cfg = RetrievalConfig(retrieval_mode="sparse", dense_enabled=False, bm25_enabled=True)
        sparse = retrieve(view, "POL-001", [], cfg, 2, model)
        model.encode.assert_not_called()
        self.assertEqual(sparse.candidates[0]["faiss_index"], 2)
        view.sparse = None
        fallback = retrieve(view, "POL-001", [], cfg, 2, model)
        self.assertEqual(fallback.trace["degraded"], ["bm25_missing_from_snapshot"])
        self.assertEqual(fallback.candidates[0]["faiss_index"], 0)

    def test_channel_and_rerank_failures_do_not_discard_available_evidence(self):
        view, model = fixture()
        cfg = RetrievalConfig(retrieval_mode="hybrid", bm25_enabled=True, rrf_enabled=True)
        with patch("retrieval_engine.bm25", side_effect=ValueError):
            result = retrieve(view, "POL-001", [], cfg, 2, model)
        self.assertTrue(result.usable())
        self.assertIn("bm25.original", result.trace["degraded"])
        scores, enabled = rescore(result, [("q", "e")], Mock(predict=Mock(side_effect=RuntimeError)))
        self.assertEqual((scores, enabled), ([0.0], False))
        self.assertIn("rerank_failed", result.trace["degraded"])

    def test_parent_budget_version_and_subquery_gaps(self):
        view, model = fixture()
        cfg = RetrievalConfig(parent_child_enabled=True, parent_context_budget=12, query_decomposition_enabled=True)
        result = retrieve(view, "退款可以吗？运费谁承担？", [], cfg, 3, model)
        extra = expand_parents(result, result.candidates, 100)
        self.assertEqual([e["parent_id"] for e in extra], ["p1"])
        self.assertLessEqual(len(extra[0]["text"].encode()), 12)
        self.assertEqual(view.all_docs[0].metadata["child_id"], "c0")
        gaps = subquery_gaps(result, result.candidates[:1], Mock(return_value={"sufficient_for_answer": True}))
        self.assertIn("sub2", gaps)

class HyDEBudgetTests(TestCase):
    def test_hyde_and_dense_share_one_tool_budget(self):
        view, embedder = fixture()
        config = RetrievalConfig(hyde_enabled=True)
        generator = Mock(return_value=SimpleNamespace(content="synthetic"))
        budget = [1]
        retrieve(view, "test", [], config, 3, embedder, generator, tool_budget=budget)
        generator.assert_called_once()
        embedder.encode.assert_not_called()
        self.assertEqual(budget, [0])

        generator.reset_mock()
        retrieve(view, "test", [], config, 3, embedder, generator, tool_budget=budget)
        generator.assert_not_called()
