import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from container_identity import load_contract


class ContainerContractTests(TestCase):
    def fixture(self):
        identity = dict(deployment_mode="docker", mysql_datadir="/var/lib/mysql/",
            milvus_uri="http://standalone:19530", resource_root="/data",
            MYSQL_HOST="mysql", MYSQL_PORT="43317", MYSQL_DATABASE="rag_v2_docker_test",
            REDIS_HOST="redis", REDIS_PORT="46397", REDIS_DB="15",
            RAG_V2_ROOT="/data/store", HF_HOME="/data/HF_HOME",
            RAG_V2_EMBED_PATH="/models/embed", RAG_V2_RERANK_PATH="/models/rerank",
            RAG_V2_MYSQL_UUID="fixture-id", RAG_V2_REDIS_PREFIX="ragv2v3_audit_docker_test:",
            RAG_V2_REDIS_IDENTITY="fixture-instance")
        private = identity | dict(MYSQL_USER="app", MYSQL_PASSWORD="test-only",
            REDIS_PASSWORD="test-only", RAG_V2_ADMIN_TOKEN="test-only-admin",
            RAG_V2_ENABLED="1", RAG_V3_MILVUS_URI=identity["milvus_uri"])
        return identity, private

    def check(self, identity, private):
        with TemporaryDirectory() as folder:
            paths = [Path(folder)/name for name in ("identity.json", "private.json")]
            for path, data in zip(paths, (identity, private)):
                path.write_text(json.dumps(data), encoding="utf-8")
            return load_contract(*paths)

    def test_exact_contract_is_accepted(self):
        identity, private = self.fixture()
        self.assertEqual(self.check(identity, private), (identity, private))

    def test_legacy_and_mismatched_targets_are_denied(self):
        for key, value in (("MYSQL_PORT", "3306"), ("MYSQL_DATABASE", "rag_demo"),
                           ("REDIS_PORT", "6379"), ("REDIS_DB", "0"),
                           ("RAG_V2_MYSQL_UUID", "other"), ("RAG_V2_REDIS_IDENTITY", "other")):
            with self.subTest(key=key):
                identity, private = self.fixture()
                private[key] = value
                with self.assertRaises(RuntimeError):
                    self.check(identity, private)
