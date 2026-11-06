"""Explicit Linux deployment contract; no Windows/local-instance fallback."""
import json
from pathlib import Path


def load_contract(identity_path, secret_path):
    identity = json.loads(Path(identity_path).read_text(encoding="utf-8"))
    private = json.loads(Path(secret_path).read_text(encoding="utf-8"))
    required = {
        "MYSQL_HOST": "mysql", "MYSQL_PORT": "43317",
        "REDIS_HOST": "redis", "REDIS_PORT": "46397", "REDIS_DB": "15",
        "RAG_V2_ROOT": "/data/store", "HF_HOME": "/data/HF_HOME",
        "RAG_V2_EMBED_PATH": "/models/embed", "RAG_V2_RERANK_PATH": "/models/rerank",
    }
    if (identity.get("deployment_mode") != "docker"
            or identity.get("mysql_datadir") != "/var/lib/mysql/"
            or identity.get("milvus_uri") != "http://standalone:19530"
            or identity.get("resource_root") != "/data"):
        raise RuntimeError("Invalid container identity contract")
    for key, value in required.items():
        if identity.get(key) != value or str(private.get(key, "")) != value:
            raise RuntimeError("Container target mismatch: " + key)
    for key in ("MYSQL_DATABASE", "RAG_V2_MYSQL_UUID", "RAG_V2_REDIS_PREFIX", "RAG_V2_REDIS_IDENTITY"):
        if not identity.get(key) or private.get(key) != identity[key]:
            raise RuntimeError("Container identity mismatch: " + key)
    if (not identity["MYSQL_DATABASE"].startswith("rag_v2_docker_")
            or not identity["RAG_V2_REDIS_PREFIX"].startswith("ragv2v3_audit_docker_")):
        raise RuntimeError("Forbidden container resource namespace")
    for key in ("MYSQL_USER", "MYSQL_PASSWORD", "REDIS_PASSWORD", "RAG_V2_ADMIN_TOKEN"):
        if not private.get(key):
            raise RuntimeError("Missing container secret: " + key)
    if private.get("RAG_V2_ENABLED") != "1" or private.get("RAG_V3_MILVUS_URI") != identity["milvus_uri"]:
        raise RuntimeError("Container runtime contract mismatch")
    return identity, private
