from request_budget import BudgetExceeded
"""Opt-in management routes and isolated environment checks."""
from hashlib import sha256
import hmac
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
import pymysql

from knowledge_v2 import KnowledgeError, KnowledgeStore, MAX_UPLOAD, ProcessingConfig
from redis_utils import get_redis_client

logger = logging.getLogger(__name__)


def model_path(variable: str) -> Path:
    from isolated_config import readonly_model_path
    return readonly_model_path(variable)


def isolated_settings():
    required = ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD",
                "RAG_V2_MYSQL_UUID", "REDIS_HOST", "REDIS_PORT", "REDIS_DB", "REDIS_PASSWORD",
                "RAG_V2_REDIS_IDENTITY", "RAG_V2_ADMIN_TOKEN", "RAG_V2_ROOT", "HF_HOME")
    if any(not os.getenv(key) for key in required):
        raise RuntimeError("Missing explicit isolated RAG V2 settings")
    from isolated_config import require_target
    require_target("mysql")
    require_target("redis")
    env = os.environ
    root = Path(env["RAG_V2_ROOT"]).resolve()
    hf = Path(env["HF_HOME"]).resolve()
    repo = Path(__file__).resolve().parents[1]
    if (not env["MYSQL_DATABASE"].startswith("rag_v2_") or env["MYSQL_PORT"] == "3306"
            or env["REDIS_PORT"] == "6379" or len(env["RAG_V2_ADMIN_TOKEN"]) < 24
            or not Path(env["RAG_V2_ROOT"]).is_absolute() or not Path(env["HF_HOME"]).is_absolute()
            or root == hf or root in hf.parents or hf in root.parents
            or any(p == repo or repo in p.parents or p in repo.parents for p in (root, hf))
            or hf.parent != root.parent):
        raise RuntimeError("RAG V2 requires isolated resources and dedicated endpoints")
    kwargs = dict(host=env["MYSQL_HOST"], port=int(env["MYSQL_PORT"]), user=env["MYSQL_USER"],
                  password=env["MYSQL_PASSWORD"], database=env["MYSQL_DATABASE"], charset="utf8mb4",
                  cursorclass=pymysql.cursors.DictCursor, connect_timeout=4, read_timeout=8, write_timeout=8)
    expected_uuid = env["RAG_V2_MYSQL_UUID"].lower()

    def connect():
        db = pymysql.connect(**kwargs)
        try:
            with db.cursor() as c:
                c.execute("SELECT DATABASE() AS db, @@server_uuid AS server_id")
                identity = c.fetchone()
                if identity["db"] != kwargs["database"] or identity["server_id"].lower() != expected_uuid:
                    raise RuntimeError("MySQL identity mismatch")
            from isolated_config import verify_mysql
            return verify_mysql(db)
        except BaseException:
            db.close()
            raise

    with connect() as db:
        with db.cursor() as c:
            c.execute("SELECT generation FROM rag_v2_head WHERE id=1")
            if c.fetchone() is None:
                raise RuntimeError("RAG V2 migration required")
    with get_redis_client() as client:
        if client.get("rag:v2:identity") != env["RAG_V2_REDIS_IDENTITY"]:
            raise RuntimeError("Redis identity mismatch")
    return root, connect


def configure_knowledge(app, root, connect, embed_model):
    embed_path = model_path("RAG_V2_EMBED_PATH")
    rerank_path = model_path("RAG_V2_RERANK_PATH")
    settings = ProcessingConfig(
        child_chars=int(os.getenv("RAG_V2_CHILD_CHARS", "500")),
        overlap_chars=int(os.getenv("RAG_V2_OVERLAP_CHARS", "100")),
        trim_line_edges=os.getenv("RAG_V2_TRIM_LINE_EDGES", "1") == "1",
        skip_page_labels=os.getenv("RAG_V2_SKIP_PAGE_LABELS", "0") == "1",
    )
    from isolated_config import verified_model_identity
    embed_identity = verified_model_identity("RAG_V2_EMBED_PATH")
    app.state.knowledge = KnowledgeStore(root, connect, embed_model, embed_identity, settings)
    app.state.knowledge_namespace = sha256(
        (os.environ["RAG_V2_MYSQL_UUID"] + "/" + os.environ["MYSQL_DATABASE"]).encode()
    ).hexdigest()[:24]
    code = sha256(b"".join(Path(__file__).with_name(name).read_bytes()
                           for name in ("api_app.py", "models.py", "retrieval_engine.py", "retrieval_sparse.py",
                                        "policy_evidence_support.py"))).hexdigest()
    app.state.knowledge_model_config = "|".join((str(embed_path), str(rerank_path), code,
        os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), os.getenv("RAG_V2_GENERATION_REVISION", "provider-default"),
        sha256(os.getenv("DEEPSEEK_API_BASE", "").encode()).hexdigest()))


def _store(request: Request, x_rag_v2_token: str | None = Header(default=None)) -> KnowledgeStore:
    store = getattr(request.app.state, "knowledge", None)
    if store is None:
        raise HTTPException(404, "Knowledge management is disabled")
    require_management_token(x_rag_v2_token)
    return store


def require_management_token(x_rag_v2_token: str | None = Header(default=None)):
    token = os.getenv("RAG_V2_ADMIN_TOKEN", "")
    if not token or not hmac.compare_digest(token.encode(), (x_rag_v2_token or "").encode()):
        raise HTTPException(403, "Knowledge management token required")


def call(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except KnowledgeError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except BudgetExceeded:
        raise
    except (pymysql.MySQLError, OSError) as exc:
        logger.exception("Knowledge management operation failed")
        raise HTTPException(503, "Operation unavailable. Retry publication with the same key.") from exc


router = APIRouter(prefix="/v2", tags=["knowledge-v2"])


@router.get("/documents")
def documents(store: KnowledgeStore = Depends(_store)):
    draft = call(store.documents)
    active = call(store.pin)
    return {"documents": draft, "active_members": active.manifest["members"],
            "snapshot_id": active.snapshot_id, "publish_generation": active.generation}


@router.post("/documents", status_code=201)
def upload(file: UploadFile = File(...), version: str | None = Form(default=None),
           category: str = Form(default="general"), origin: str = Form(default="upload"),
           store: KnowledgeStore = Depends(_store)):
    return call(store.ingest, file.filename, file.file.read(MAX_UPLOAD + 1), version, category=category, origin=origin)


@router.post("/documents/{document_id}/versions", status_code=201)
def replace(document_id: str, file: UploadFile = File(...), expected_version: str = Form(...),
            version: str | None = Form(default=None), category: str = Form(default="general"),
            origin: str = Form(default="upload"), store: KnowledgeStore = Depends(_store)):
    return call(store.ingest, file.filename, file.file.read(MAX_UPLOAD + 1), version, document_id,
                expected_version, category=category, origin=origin)


@router.get("/documents/{document_id}/versions")
def versions(document_id: str, store: KnowledgeStore = Depends(_store)):
    return call(store.versions, document_id)


@router.get("/document-versions/{version_id}/artifact")
def artifact(version_id: str, store: KnowledgeStore = Depends(_store)):
    return call(store.document_artifact, version_id)


@router.delete("/documents/{document_id}")
def delete(document_id: str, expected_version: str = Query(...), store: KnowledgeStore = Depends(_store)):
    return call(store.delete, document_id, expected_version)


class SnapshotBuild(BaseModel):
    document_ids: list[str] | None = Field(default=None, min_length=1, max_length=1000)


@router.post("/index-snapshots", status_code=201)
def build(payload: SnapshotBuild | None = None, store: KnowledgeStore = Depends(_store)):
    return call(store.build) if payload is None or payload.document_ids is None else call(
        store.build, document_ids=payload.document_ids)


@router.get("/index-snapshots")
def snapshots(store: KnowledgeStore = Depends(_store)):
    return {"head": call(store.head), "snapshots": call(store.snapshots)}


@router.get("/index-snapshots/{snapshot_id}")
def snapshot(snapshot_id: str, store: KnowledgeStore = Depends(_store)):
    return call(store.snapshot_status, snapshot_id)


@router.get("/index-snapshots/{snapshot_id}/chunks")
def chunks(snapshot_id: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500),
           store: KnowledgeStore = Depends(_store)):
    view = call(store.load_snapshot, snapshot_id)
    children = view.manifest["children"][offset:offset + limit]
    parents = {child["parent_id"] for child in children}
    return {"snapshot_id": snapshot_id, "children": children, "total": len(view.all_docs),
            "parents": [p for p in view.manifest["parents"] if p["parent_id"] in parents]}


@router.get("/publications")
def history(store: KnowledgeStore = Depends(_store)):
    return call(store.publications)


@router.get("/publication-tasks")
def publication_tasks(store: KnowledgeStore = Depends(_store)):
    return call(store.publication_tasks)


class Publication(BaseModel):
    expected_generation: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1, max_length=120)


@router.post("/index-snapshots/{snapshot_id}/publish")
def publish(snapshot_id: str, payload: Publication, store: KnowledgeStore = Depends(_store)):
    return call(store.publish, snapshot_id, payload.expected_generation, payload.idempotency_key)


@router.post("/index-snapshots/{snapshot_id}/rollback")
def rollback(snapshot_id: str, payload: Publication, store: KnowledgeStore = Depends(_store)):
    return call(store.publish, snapshot_id, payload.expected_generation, payload.idempotency_key, rollback=True)



def published_view(store, snapshot_id):
    if snapshot_id is None:
        return store.pin()
    with store.transaction() as c:
        c.execute("SELECT MAX(publish_generation) AS generation FROM rag_v2_publications WHERE snapshot_id=%s", (snapshot_id,))
        generation = c.fetchone()["generation"]
    if generation is None:
        raise KnowledgeError("Historical query requires a published snapshot", 409)
    return store.load_snapshot(snapshot_id, generation, strict_sparse=True)
