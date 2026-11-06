from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from knowledge_v2 import KnowledgeError, ProcessingConfig, Snapshot, process_document, read_checked, write_file


class ProcessingTests(TestCase):
    def test_numeric_lines_paragraphs_and_raw_offsets(self):
        raw = "  规则编号\r\n123\r\n\r\n  金额\r\n200\r\n"
        result = process_document("policy.txt", raw.encode(), "v1", ProcessingConfig())
        self.assertEqual([p["text"] for p in result["parents"]], ["规则编号\n123", "金额\n200"])
        for child in result["children"]:
            value = raw[child["start"]:child["end"]]
            self.assertEqual(value.replace("\r\n", "\n"), child["text"])
        self.assertEqual(result["pages"][0]["text"], raw)

    def test_one_parent_multiple_children_and_distinct_versions(self):
        options = ProcessingConfig(child_chars=8, overlap_chars=2)
        old = process_document("policy.txt", b"1234567890abcdefghijkl", "v1", options)
        new = process_document("policy.txt", b"1234567890abcdefghijkl", "v2", options)
        self.assertEqual(len(old["parents"]), 1)
        self.assertGreater(len(old["children"]), 2)
        self.assertTrue(all(c["parent_id"] == "v1-p1" for c in old["children"]))
        self.assertEqual(new["children"][0]["content_hash"], old["children"][0]["content_hash"])
        self.assertNotEqual(new["children"][0]["child_id"], old["children"][0]["child_id"])

    def test_invalid_pdf_and_blank_text(self):
        for name, value in (("broken.pdf", b"broken"), ("blank.txt", b" \n\n")):
            with self.subTest(name=name), self.assertRaises(KnowledgeError):
                process_document(name, value, "v1", ProcessingConfig())

    def test_cache_fences_rollback_endpoint_and_model(self):
        old = Snapshot("s", "corpus", 1, None, None, (), {})
        rollback = Snapshot("s", "corpus", 3, None, None, (), {})
        self.assertEqual(len({
            old.cache_key("ask", "q", "prompt1", "test"),
            rollback.cache_key("ask", "q", "prompt1", "test"),
            old.cache_key("policy", "q", "prompt1", "test"),
            old.cache_key("ask", "q", "prompt2", "test"),
        }), 4)

    def test_checksum_and_exclusive_artifact_write(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "index"
            write_file(path, b"content")
            self.assertEqual(read_checked(path, sha256(b"content").hexdigest()), b"content")
            with self.assertRaises(FileExistsError):
                write_file(path, b"new")
            with self.assertRaises(KnowledgeError):
                read_checked(path, sha256(b"new").hexdigest())
