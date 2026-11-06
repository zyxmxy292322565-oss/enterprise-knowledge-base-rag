import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import api_app
from knowledge_v2 import KnowledgeStore


class AuditProtectionTests(TestCase):
    def test_legacy_write_routes_require_management_token(self):
        client = TestClient(api_app.app)
        for path, method, data in [
            ("/upload", "POST", {"files": {"file": ("fixture.txt", b"fixture")}}),
            ("/test_upload", "POST", {"files": {"file": ("fixture.txt", b"fixture")}}),
            ("/documents/synthetic", "DELETE", {}),
        ]:
            for token in ("", "wrong"):
                with self.subTest(path=path, token=token), patch.dict(os.environ, {"RAG_V2_ADMIN_TOKEN": "fixture"}):
                    response = client.request(method, path, headers={"X-RAG-V2-Token": token}, **data)
                    self.assertEqual(response.status_code, 403)
            with patch.dict(os.environ, {"RAG_V2_ADMIN_TOKEN": ""}):
                self.assertEqual(client.request(method, path, **data).status_code, 403)

    def test_authorized_upload_probe_and_v2_mode_write_guard(self):
        client = TestClient(api_app.app, headers={"X-RAG-V2-Token": "fixture"})
        files = {"file": ("fixture.txt", b"fixture")}
        with patch.dict(os.environ, {"RAG_V2_ADMIN_TOKEN": "fixture"}):
            self.assertEqual(client.post("/test_upload", files=files).json()["file_size"], 7)
            with patch.object(api_app, "_knowledge", return_value=object()):
                self.assertFalse(client.post("/upload", files=files).json()["success"])
                self.assertFalse(client.delete("/documents/synthetic").json()["success"])

    def test_legacy_query_has_no_redis_cache_key(self):
        for endpoint in ("ask", "policy"):
            self.assertEqual(api_app._versioned_key(api_app.app.state, endpoint, "legacy"), "")

    def test_active_snapshot_pin_demands_strict_integrity(self):
        store = SimpleNamespace(head=lambda: {"snapshot_id": "active", "generation": 4},
                                load_snapshot=Mock())
        KnowledgeStore.pin(store)
        store.load_snapshot.assert_called_once_with("active", 4, strict_sparse=True)

    def test_history_is_in_cache_identity_without_explicit_strategy(self):
        from knowledge_v2 import Snapshot
        view = Snapshot("s", "corpus", 1, None, None, (), {})
        a = api_app.AskRequest(question="same", chat_history=[{"role": "user", "content": "context A"}])
        b = api_app.AskRequest(question="same", chat_history=[{"role": "user", "content": "context B"}])
        with patch.object(api_app.app.state, "knowledge_model_config", "model", create=True), patch.object(api_app.app.state, "knowledge_namespace", "test", create=True):
            self.assertNotEqual(api_app._versioned_key(view, "ask", "same", a), api_app._versioned_key(view, "ask", "same", b))

    def test_corrupt_cached_payload_is_a_miss(self):
        import redis_utils
        for value in ("not-json", "[]", "42"):
            with self.subTest(value=value), patch.object(redis_utils, "get_redis_client") as factory:
                factory.return_value.get.return_value = value
                self.assertIsNone(redis_utils.get_cache("fixture-key"))
