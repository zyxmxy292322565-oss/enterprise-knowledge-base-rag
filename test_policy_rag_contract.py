"""HTTP response contract and Redis connection configuration tests."""

import os
import fnmatch
import hashlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

import api_app
import db_utils
import redis_utils
from cache_utils import RAG_QUERY_CACHE_PREFIX, build_policy_cache_key
from policy_evidence_support import POLICY_EVIDENCE_CACHE_VERSION


class PolicyRAGContractTests(unittest.TestCase):
    def test_document_catalog_persists_management_metadata(self):
        cursor = MagicMock()
        connection = MagicMock()
        connection.cursor.return_value = cursor
        document = SimpleNamespace(metadata={
            "source": "earphone.txt", "file_hash": "hash-metadata", "chunk_id": 1,
            "category": "product_policy", "product_scope": "earphone",
            "version": "2026.10", "created_by": "admin_workbench",
        })
        with patch.object(db_utils, "get_connection", return_value=connection):
            db_utils.save_document_records([document])

        insert = cursor.execute.call_args
        self.assertIn("category", insert.args[0])
        self.assertIn("updated_at", insert.args[0])
        self.assertEqual(insert.args[1][3:7], (
            "product_policy", "earphone", "2026.10", "admin_workbench",
        ))
        connection.commit.assert_called_once()

    def test_upload_forwards_management_metadata_to_chunks_and_document_catalog(self):
        document = SimpleNamespace(page_content="耳机类质量问题 30 天内可申请售后", metadata={
            "source": "earphone.txt", "file_hash": "hash-policy", "chunk_id": 1,
        })
        with patch.object(api_app.app.state, "index", None, create=True), \
             patch.object(api_app.app.state, "vectors", None, create=True), \
             patch.object(api_app.app.state, "all_docs", [], create=True), \
             patch.object(api_app.app.state, "processed_file_hashes", set(), create=True), \
             patch.object(api_app.app.state, "embed_model", MagicMock(), create=True), \
             patch.object(api_app.app.state, "vectors_cache_file", "vectors.pkl", create=True), \
             patch.object(api_app.app.state, "all_docs_cache_file", "docs.pkl", create=True), \
             patch.object(api_app.app.state, "index_cache_file", "index.faiss", create=True), \
             patch.object(api_app, "validate_upload_file_basic", return_value={"valid": True, "file_ext": "txt"}), \
             patch.object(api_app, "load_uploaded_file_text", return_value="耳机类质量问题 30 天内可申请售后"), \
             patch.object(api_app, "validate_extracted_text", return_value={"valid": True, "text": "原始文本"}), \
             patch.object(api_app, "clean_document_text", return_value={"raw_length": 4, "cleaned_length": 4, "removed_chars": 0, "clean_rate": 0, "cleaned_text": "清洗文本"}), \
             patch.object(api_app, "validate_cleaned_text", return_value={"valid": True, "text": "清洗文本"}), \
             patch.object(api_app, "split_document_text", return_value=[document]) as split, \
             patch.object(api_app, "save_cache"), \
             patch.object(api_app, "save_document_records") as save_documents, \
             patch.object(api_app, "invalidate_rag_query_cache", return_value=4) as clear_cache:
            api_app.app.state.embed_model.encode.return_value = np.array([[0.1, 0.2]], dtype=np.float32)
            response = TestClient(api_app.app).post(
                "/upload",
                files={"file": ("earphone.txt", "耳机类质量问题 30 天内可申请售后".encode("utf-8"), "text/plain")},
                data={"category": "product_policy", "product_scope": "earphone", "version": "2026.10", "created_by": "admin_workbench"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(response.json()["category"], "product_policy")
        self.assertEqual(split.call_args.kwargs["extra_metadata"]["product_scope"], "earphone")
        self.assertEqual(split.call_args.kwargs["extra_metadata"]["version"], "2026.10")
        save_documents.assert_called_once_with([document])
        clear_cache.assert_called_once()

    def test_delete_rebuilds_faiss_and_clears_rag_answer_cache(self):
        document = SimpleNamespace(page_content="耳机政策", metadata={"source": "earphone.txt", "file_hash": "hash-delete"})
        with patch.object(api_app.app.state, "all_docs", [document], create=True), \
             patch.object(api_app, "rebuild_faiss_index_from_docs", return_value={"index_count": 3}) as rebuild, \
             patch.object(api_app, "delete_document_record_by_hash", return_value=1), \
             patch.object(api_app, "invalidate_rag_query_cache", return_value=2) as clear_cache:
            response = TestClient(api_app.app).delete("/documents/hash-delete")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(response.json()["rag_status"]["index_count"], 3)
        rebuild.assert_called_once()
        clear_cache.assert_called_once()

    def test_upload_query_delete_query_invalidates_removed_evidence(self):
        question = "智能手表签收几天可以退？"
        content = "测试售后政策：测试商品为智能手表。签收后5天内支持退货。超过5天需要人工审核。"
        file_bytes = content.encode("utf-8")
        file_hash = hashlib.md5(file_bytes).hexdigest()
        test_document = SimpleNamespace(page_content=content, metadata={
            "source": "test_policy.txt", "file_hash": file_hash, "chunk_id": 1,
        })
        base_document = SimpleNamespace(page_content="通用退货政策：请核对订单。", metadata={
            "source": "base_policy.txt", "file_hash": "base-hash", "chunk_id": 1,
        })
        index = api_app.faiss.IndexFlatL2(2)
        index.add(np.array([[1.0, 1.0]], dtype=np.float32))
        embed_model = MagicMock()
        embed_model.encode.side_effect = lambda texts: np.zeros((len(texts), 2), dtype=np.float32)
        rerank_model = MagicMock()
        rerank_model.predict.side_effect = lambda pairs: np.full(len(pairs), 0.9)
        cache = {}
        redis_client = MagicMock()
        redis_client.get.side_effect = lambda key: cache.get(key)
        redis_client.set.side_effect = lambda name, value, ex: cache.__setitem__(name, value)
        redis_client.scan_iter.side_effect = lambda match: (
            key for key in list(cache) if fnmatch.fnmatch(key, match)
        )
        redis_client.delete.side_effect = lambda key: int(cache.pop(key, None) is not None)
        cache_key = build_policy_cache_key(question, 5, 3, 0.2, "v1") + ":" + POLICY_EVIDENCE_CACHE_VERSION
        cache[cache_key] = '{"success": true, "code": "SUCCESS", "evidence": [{"source": "stale.txt"}]}'
        cache["rag:answer:legacy"] = "legacy result"

        with patch.object(api_app.app.state, "index", index, create=True), \
             patch.object(api_app.app.state, "vectors", np.array([[1.0, 1.0]], dtype=np.float32), create=True), \
             patch.object(api_app.app.state, "all_docs", [base_document], create=True), \
             patch.object(api_app.app.state, "processed_file_hashes", {"base-hash"}, create=True), \
             patch.object(api_app.app.state, "embed_model", embed_model, create=True), \
             patch.object(api_app.app.state, "rerank_model", rerank_model, create=True), \
             patch.object(api_app.app.state, "vectors_cache_file", "vectors.pkl", create=True), \
             patch.object(api_app.app.state, "all_docs_cache_file", "docs.pkl", create=True), \
             patch.object(api_app.app.state, "index_cache_file", "index.faiss", create=True), \
             patch.object(redis_utils, "get_redis_client", return_value=redis_client), \
             patch.object(api_app, "validate_upload_file_basic", return_value={"valid": True, "file_ext": "txt"}), \
             patch.object(api_app, "load_uploaded_file_text", return_value=content), \
             patch.object(api_app, "validate_extracted_text", return_value={"valid": True, "text": content}), \
             patch.object(api_app, "clean_document_text", return_value={"raw_length": len(content), "cleaned_length": len(content), "removed_chars": 0, "clean_rate": 0, "cleaned_text": content}), \
             patch.object(api_app, "validate_cleaned_text", return_value={"valid": True, "text": content}), \
             patch.object(api_app, "split_document_text", return_value=[test_document]), \
             patch.object(api_app, "save_cache") as save_index, \
             patch.object(api_app, "save_document_records") as save_document, \
             patch.object(api_app, "delete_document_record_by_hash", return_value=1) as delete_document:
            client = TestClient(api_app.app)
            uploaded = client.post("/upload", files={"file": ("test_policy.txt", file_bytes, "text/plain")})
            self.assertTrue(uploaded.json()["success"])
            self.assertNotIn(cache_key, cache)
            self.assertNotIn("rag:answer:legacy", cache)
            first = client.post("/rag/query", json={"question": question, "top_k": 5}).json()
            self.assertTrue(first["success"])
            self.assertFalse(first["cache_hit"])
            self.assertIn("test_policy.txt", [item["source"] for item in first["evidence"]])
            self.assertIn(cache_key, cache)
            deleted = client.delete(f"/documents/{file_hash}").json()
            self.assertTrue(deleted["success"])
            self.assertEqual(deleted["deleted_chunk_count"], 1)
            self.assertNotIn(cache_key, cache)
            second = client.post("/rag/query", json={"question": question, "top_k": 5}).json()
            self.assertTrue(second["success"])
            self.assertFalse(second["cache_hit"])
            self.assertNotIn("test_policy.txt", [item["source"] for item in second["evidence"]])
            # A removed product-specific policy cannot be replaced with a
            # generic paragraph and still claim that product is covered.
            self.assertEqual(second["code"], "POLICY_NOT_FOUND")
            self.assertTrue(second["refused"])
            self.assertEqual(second["evidence"], [])
            self.assertEqual(save_index.call_count, 2)
            save_document.assert_called_once_with([test_document])
            delete_document.assert_called_once_with(file_hash)

    def test_successful_query_returns_policy_evidence(self):
        index = MagicMock()
        index.search.return_value = (np.array([[0.12]]), np.array([[0]]))
        embed_model = MagicMock()
        embed_model.encode.return_value = np.array([[0.1, 0.2]])
        rerank_model = MagicMock()
        rerank_model.predict.return_value = np.array([0.9])
        document = SimpleNamespace(
            page_content="AirPods退货政策：请以订单和售后规则核验。",
            metadata={
                "source": "returns.md",
                "file_hash": "hash-1",
                "chunk_id": "chunk-1",
                "policy_version": "v1",
            },
        )
        with patch.object(api_app.app.state, "index", index, create=True), \
             patch.object(api_app.app.state, "vectors", np.array([[0.1, 0.2]]), create=True), \
             patch.object(api_app.app.state, "all_docs", [document], create=True), \
             patch.object(api_app.app.state, "embed_model", embed_model, create=True), \
             patch.object(api_app.app.state, "rerank_model", rerank_model, create=True), \
             patch.object(api_app, "get_cache", return_value=None), \
             patch.object(api_app, "set_cache") as set_cache:
            response = TestClient(api_app.app).post(
                "/rag/query", json={"question": "AirPods用了20天还能退吗？", "top_k": 5}
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsInstance(payload, dict)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["code"], "SUCCESS")
        self.assertFalse(payload["refused"])
        self.assertEqual(payload["evidence"][0]["source"], "returns.md")
        self.assertEqual(payload["evidence"][0]["content"], document.page_content)
        self.assertIn("total_ms", payload["timing"])
        set_cache.assert_called_once()

    def test_query_version_metadata_compatibility_and_conflict(self):
        """An in-memory FAISS index exercises the real /rag/query response path."""
        cases = (
            ({"version": "v1"}, "v1", "SUCCESS"),
            ({"policy_version": "v1"}, "v1", "SUCCESS"),
            ({}, None, "SUCCESS"),
            ({"version": "v1", "policy_version": "v1"}, "v1", "SUCCESS"),
            ({"version": "v2", "policy_version": "v1"}, None, "POLICY_METADATA_CONFLICT"),
        )
        for version_metadata, expected_version, expected_code in cases:
            with self.subTest(metadata=version_metadata):
                index = api_app.faiss.IndexFlatL2(2)
                index.add(np.array([[0.1, 0.2]], dtype=np.float32))
                document = SimpleNamespace(
                    page_content="耳机质量问题政策依据",
                    metadata={"source": "earphone.txt", "file_hash": "known-hash",
                              "chunk_id": 1, **version_metadata},
                )
                embed_model = MagicMock()
                embed_model.encode.return_value = np.array([[0.1, 0.2]], dtype=np.float32)
                rerank_model = MagicMock()
                rerank_model.predict.return_value = np.array([0.9])
                with patch.object(api_app.app.state, "index", index, create=True), \
                     patch.object(api_app.app.state, "vectors", np.array([[0.1, 0.2]]), create=True), \
                     patch.object(api_app.app.state, "all_docs", [document], create=True), \
                     patch.object(api_app.app.state, "embed_model", embed_model, create=True), \
                     patch.object(api_app.app.state, "rerank_model", rerank_model, create=True), \
                     patch.object(api_app, "get_cache", return_value=None), \
                     patch.object(api_app, "set_cache") as set_cache:
                    result = TestClient(api_app.app).post(
                        "/rag/query", json={"question": "耳机质量问题可以申请售后吗？", "top_k": 1}
                    ).json()
                self.assertEqual(result["code"], expected_code, result)
                if expected_code == "SUCCESS":
                    self.assertEqual(result["evidence"][0]["version"], expected_version)
                    self.assertEqual(result["evidence"][0]["source"], "earphone.txt")
                    self.assertEqual(result["evidence"][0]["file_hash"], "known-hash")
                    set_cache.assert_called_once()
                else:
                    self.assertFalse(result["success"])
                    self.assertTrue(result["refused"])
                    self.assertEqual(result["evidence"], [])
                    set_cache.assert_not_called()

    def test_version_metadata_survives_isolated_faiss_cache_roundtrip(self):
        vectors = np.array([[0.1, 0.2]], dtype=np.float32)
        index = api_app.faiss.IndexFlatL2(2)
        index.add(vectors)
        document = SimpleNamespace(page_content="耳机政策", metadata={
            "source": "earphone.txt", "file_hash": "known-hash", "version": "v2",
        })
        with tempfile.TemporaryDirectory(prefix="rag_version_test_") as directory:
            paths = [os.path.join(directory, name) for name in
                     ("vectors.pkl", "docs.pkl", "index.faiss")]
            api_app.save_cache(vectors, [document], index, *paths)
            loaded_docs, loaded_vectors, loaded_index = api_app.load_cache(*paths)
        self.assertEqual(loaded_docs[0].metadata["version"], "v2")
        self.assertEqual(loaded_docs[0].metadata["file_hash"], "known-hash")
        self.assertEqual(loaded_index.ntotal, 1)
        np.testing.assert_array_equal(loaded_vectors, vectors)

    def test_empty_knowledge_base_keeps_response_shape(self):
        with patch.object(api_app.app.state, "index", None, create=True), \
             patch.object(api_app.app.state, "vectors", None, create=True), \
             patch.object(api_app.app.state, "all_docs", [], create=True), \
             patch.object(api_app, "get_cache", return_value=None):
            response = TestClient(api_app.app).post(
                "/rag/query", json={"question": "退货政策是什么？", "top_k": 5}
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["code"], "RAG_EMPTY")
        self.assertEqual(payload["evidence"], [])


class RedisConfigurationTests(unittest.TestCase):
    @patch("redis_utils.redis.Redis")
    def test_password_is_forwarded_without_changing_existing_parameters(self, redis_constructor):
        with patch.dict(os.environ, {
            "REDIS_HOST": "redis",
            "REDIS_PORT": "6380",
            "REDIS_DB": "1",
            "REDIS_PASSWORD": "test-password",
        }):
            redis_utils.get_redis_client()

        self.assertEqual(redis_constructor.call_args.kwargs, {
            "host": "redis",
            "port": 6380,
            "db": 1,
            "decode_responses": True,
            "password": "test-password",
        })

    @patch("redis_utils.redis.Redis")
    def test_missing_password_preserves_no_auth_configuration(self, redis_constructor):
        with patch.dict(os.environ, {
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "REDIS_DB": "0",
            "REDIS_PASSWORD": "",
        }):
            redis_utils.get_redis_client()

        self.assertNotIn("password", redis_constructor.call_args.kwargs)
        self.assertEqual(redis_constructor.call_args.kwargs["db"], 0)


if __name__ == "__main__":
    unittest.main()
