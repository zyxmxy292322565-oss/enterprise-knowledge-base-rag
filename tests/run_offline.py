"""Run service-free tests in a disposable Docker container with --network none."""
import json
import os
from pathlib import Path
import unittest


def main():
    folder = Path("/run/secrets")
    if os.environ.get("RAG_DEPLOYMENT_MODE") != "docker" or any(
        (folder / name).exists() for name in ("identity.json", "environment.json")
    ):
        raise SystemExit("Use a disposable test container without deployment credentials")
    identity = dict(deployment_mode="docker", mysql_datadir="/var/lib/mysql/",
        milvus_uri="http://standalone:19530", resource_root="/data",
        protected_roots=[], MYSQL_HOST="mysql", MYSQL_PORT="43317",
        MYSQL_DATABASE="rag_v2_docker_test", RAG_V2_MYSQL_UUID="fixture-id",
        REDIS_HOST="redis", REDIS_PORT="46397", REDIS_DB="15",
        RAG_V2_REDIS_PREFIX="ragv2v3_audit_docker_test:", RAG_V2_REDIS_IDENTITY="fixture-instance",
        RAG_V2_ROOT="/data/store", HF_HOME="/data/HF_HOME",
        RAG_V2_EMBED_PATH="/models/embed", RAG_V2_RERANK_PATH="/models/rerank")
    private = identity | dict(MYSQL_USER="fixture", MYSQL_PASSWORD="test-only",
        REDIS_PASSWORD="test-only", RAG_V2_ADMIN_TOKEN="test-only-admin",
        RAG_V2_ENABLED="1", RAG_V3_MILVUS_URI=identity["milvus_uri"],
        RAG_V2_ISOLATED_LLM="1", DEEPSEEK_API_KEY="local-fixture",
        DEEPSEEK_API_BASE="http://127.0.0.1:1", RAG_V3_GRAPH_ENABLED="1")
    folder.mkdir(parents=True, exist_ok=True)
    for name, values in (("identity.json", identity), ("environment.json", private)):
        with (folder / name).open("x", encoding="utf-8") as handle:
            json.dump(values, handle)
    suite = unittest.TestSuite()
    for name in ("unit", "regression"):
        suite.addTests(unittest.TestLoader().discover(str(Path(__file__).parent / name)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(total=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), skipped=len(result.skipped),
        passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped))))
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
