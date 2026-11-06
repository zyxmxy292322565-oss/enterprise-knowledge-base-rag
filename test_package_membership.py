from unittest import TestCase
from knowledge_v2 import select_document_versions, KnowledgeError


class PackageMembershipTests(TestCase):
    def setUp(self):
        self.rows = [{"document_id": "a"*32, "document_version_id": "v1"},
                     {"document_id": "b"*32, "document_version_id": "v2"}]

    def test_legacy_all_documents_is_preserved(self):
        self.assertIs(select_document_versions(self.rows, None), self.rows)

    def test_subset_preserves_real_version_and_source_order(self):
        self.assertEqual(select_document_versions(self.rows, ["b"*32]), [self.rows[1]])
        self.assertEqual(select_document_versions(self.rows, ["b"*32, "a"*32]), self.rows)

    def test_invalid_duplicate_or_missing_members_cannot_build(self):
        for members in ([], ["bad"], ["a"*32, "a"*32], ["c"*32]):
            with self.subTest(members=members), self.assertRaises(KnowledgeError):
                select_document_versions(self.rows, members)
