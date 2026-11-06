import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from time import sleep
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import Mock, patch

from reliability import ReliabilityOptions
from semantic_reliability import evidence_for, run_semantic, Assessment, bind_required_facts
from knowledge_v2 import ProcessingConfig, digest, process_document


class SemanticTests(TestCase):
    def test_required_fact_ids_are_backend_owned(self):
        assessment = Assessment.model_validate({"action": "ANSWER", "facts": [
            {"required_fact_id": "F1", "text": "模型的说明文字", "status": "supported",
             "evidence_ids": ["a"], "reason": "原文支持"}]})
        bound = bind_required_facts(assessment, {"F1": "登记"})
        self.assertEqual(bound.facts[0].text, "登记")
        self.assertEqual(assessment.facts[0].text, "模型的说明文字")
        for facts in ([assessment.facts[0].model_copy(update={"required_fact_id": "other"})],
                      [assessment.facts[0], assessment.facts[0]],
                      [assessment.facts[0].model_copy(update={"required_fact_id": None})]):
            with self.subTest(facts=facts), self.assertRaises(ValueError):
                bind_required_facts(assessment.model_copy(update={"facts": facts}), {"F1": "登记"})
    def test_exact_txt_and_pdf_page_proof(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=200, height=200)
        n = NameObject
        page[n("/Resources")] = DictionaryObject({n("/Font"): DictionaryObject({
            n("/F1"): DictionaryObject({n("/Type"): n("/Font"), n("/Subtype"): n("/Type1"),
                                        n("/BaseFont"): n("/Helvetica")})})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 100 Td (Refund applications require review.) Tj ET")
        page[n("/Contents")] = writer._add_object(stream)
        output = BytesIO()
        writer.write(output)
        for name, raw in (("rule.pdf", output.getvalue()), ("rule.txt", b"Refund applications require review.")):
            with self.subTest(name=name), TemporaryDirectory() as folder:
                root = Path(folder)
                artifact = process_document(name, raw, "v", ProcessingConfig())
                child = artifact["children"][0] | {"source": name, "file_hash": "legacy", "version": "v1"}
                path = root / "documents/v"
                path.mkdir(parents=True)
                (path / "source.bin").write_bytes(raw)
                store = NS(root=root, document_artifact=lambda _: artifact)
                view = NS(snapshot_id="s", all_docs=[NS(page_content=child["text"],
                          metadata=child | {"index_snapshot_id": "s"})], manifest={
                          "children": [child], "parents": artifact["parents"], "members": [{
                              "source": name, "document_version_id": "v", "legacy_file_hash": "legacy",
                              "policy_version": "v1", "content_hash": digest(raw)}]})
                self.assertEqual(evidence_for(store, view, 0)["quote"], "Refund applications require review.")
                view.all_docs[0].metadata["version"] = "bad"
                with self.assertRaises(ValueError):
                    evidence_for(store, view, 0)
    def execute(self, responses, page_content=""):
        view = NS(snapshot_id="s", corpus_snapshot_id="c", generation=2)
        candidate = {"doc": NS(metadata={"child_id": "a"}, page_content=page_content), "faiss_index": 0}
        found = NS(candidates=[candidate], trace={})
        create = Mock(side_effect=[NS(model="local-test-double", usage=None, choices=[
            NS(message=NS(content=json.dumps(body, ensure_ascii=False)))]) for body in responses])
        client = Mock()
        client.__enter__ = Mock(return_value=NS(chat=NS(completions=NS(create=create))))
        client.__exit__ = Mock(return_value=False)
        factory = Mock(return_value=client)
        with patch.dict("os.environ", dict(RAG_V2_SEMANTIC_ENABLED="1", RAG_V2_SEMANTIC_GENERATOR="draft",
                RAG_V2_SEMANTIC_JUDGE="judge", RAG_V2_SEMANTIC_BASE_URL="http://localhost",
                RAG_V2_SEMANTIC_API_KEY="fixture")), patch("semantic_reliability.retrieve", return_value=found), \
                patch("semantic_reliability.evidence_for", return_value={"quote": "质量核验后可提出申请"}):
            result = run_semantic(None, view, "可以申请退款吗？", [], ReliabilityOptions(token_budget=65536), None,
                                  client_factory=factory)
        return result, create

    def test_reviewed_policy_condition_blocks_before_model(self):
        with patch("reliability.verified_rules", return_value=[("verified", "citation")]) as rules, \
                patch("reliability.validate", return_value=([], [], ["核验"], [])):
            answer, create = self.execute([], page_content="RULE reviewed-policy")
        self.assertEqual(answer["answer_action"], "CLARIFY")
        self.assertEqual(answer["governed_policy_validation"]["missing_slots"], ["核验"])
        self.assertEqual(create.call_count, 0)
        rules.assert_called_once()

    def test_missing_condition_blocks_inconsistent_model_answer(self):
        answer, create = self.execute([
            {"action": "ANSWER", "facts": [{"text": "规则", "status": "supported",
              "evidence_ids": ["a"], "reason": "条件规则存在，用户尚未提供核验状态"}],
             "missing_slots": ["核验状态"]},
        ])
        self.assertEqual(answer["answer_action"], "CLARIFY")
        self.assertIn("核验状态", answer["final_answer"])
        self.assertEqual(create.call_count, 1)
        self.assertEqual(answer["citation_validation"]["claims"], [])

    def test_explicit_long_budget_still_has_upper_bound(self):
        from pydantic import ValidationError
        self.assertEqual(ReliabilityOptions().timeout_budget_ms, 5000)
        self.assertEqual(ReliabilityOptions(timeout_budget_ms=60000).timeout_budget_ms, 60000)
        with self.assertRaises(ValidationError):
            ReliabilityOptions(timeout_budget_ms=60001)

    def test_repair_drops_unsupported_claim_and_rechecks(self):
        answer, create = self.execute([
            {"action": "ANSWER", "facts": [{"text": "申请条件", "status": "supported", "evidence_ids": ["a"], "reason": "明示"}]},
            {"claims": [{"text": "核验后可申请", "evidence_ids": ["a"]},
                        {"text": "必定退款", "evidence_ids": ["hyde"]}]},
            {"complete": False, "verdicts": [{"claim_id": "C1", "supported": True, "evidence_ids": ["a"], "reason": "明示"},
                                              {"claim_id": "C2", "supported": False, "reason": "越界"}]},
            {"complete": True, "verdicts": [{"claim_id": "C1", "supported": True, "evidence_ids": ["a"], "reason": "明示"}]},
        ])
        self.assertEqual(answer["answer_action"], "ANSWER")
        self.assertNotIn("必定", answer["final_answer"])
        self.assertEqual(answer["repair_count"], 1)
        self.assertEqual([x.kwargs["model"] for x in create.call_args_list], ["judge", "draft", "judge", "judge"])

    def test_bad_reference_never_leaves_allowlist(self):
        answer, _ = self.execute([
            {"action": "ANSWER", "facts": [{"text": "规则", "status": "supported", "evidence_ids": ["a"], "reason": "明示"}]},
            {"claims": [{"text": "可申请", "evidence_ids": ["hyde"]}]},
            {"complete": True, "verdicts": [{"claim_id": "C1", "supported": True, "evidence_ids": ["a"], "reason": "改引"}]},
            {"complete": True, "verdicts": [{"claim_id": "C1", "supported": True, "evidence_ids": ["old"], "reason": "不合法"}]},
        ])
        self.assertEqual(answer["answer_action"], "ABSTAIN")
        self.assertEqual(answer["citation_validation"]["claims"], [])
        self.assertEqual(answer["repair_count"], 1)

    def test_real_http_read_timeout_abstains(self):
        class Slow(BaseHTTPRequestHandler):
            def do_POST(self):
                sleep(0.5)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            view = NS(snapshot_id="s", corpus_snapshot_id="c", generation=2)
            found = NS(candidates=[{"doc": NS(metadata={"child_id": "a"}), "faiss_index": 0}], trace={})
            with patch.dict("os.environ", dict(RAG_V2_SEMANTIC_ENABLED="1", RAG_V2_SEMANTIC_GENERATOR="draft",
                    RAG_V2_SEMANTIC_JUDGE="judge", RAG_V2_SEMANTIC_BASE_URL=f"http://127.0.0.1:{server.server_port}",
                    RAG_V2_SEMANTIC_API_KEY="fixture")), patch("semantic_reliability.retrieve", return_value=found), \
                    patch("semantic_reliability.evidence_for", return_value={"quote": "可申请"}):
                result = run_semantic(None, view, "可以申请吗？", [],
                                      ReliabilityOptions(timeout_budget_ms=150, token_budget=65536), None)
            self.assertEqual(result["answer_action"], "ABSTAIN")
            self.assertIn(result["error"], {"TimeoutError", "APITimeoutError"})
            self.assertLessEqual(len(result["model_usage"]), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
