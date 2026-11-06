"""Immutable knowledge artifacts with an InnoDB publication commit point."""
from __future__ import annotations

from request_budget import BudgetExceeded

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256, md5
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import re
import uuid

import faiss
import numpy as np
from langchain_core.documents import Document
from pypdf import PdfReader
from retrieval_sparse import PROFILE, build_sparse, validate_sparse

logger = logging.getLogger(__name__)
MAX_UPLOAD = 10 * 1024 * 1024


class KnowledgeError(Exception):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def select_document_versions(versions, document_ids):
    """Optional package membership; default preserves the complete draft build."""
    if document_ids is None:
        return versions
    if (not isinstance(document_ids, list) or not 1 <= len(document_ids) <= 1000
            or any(not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{32}", key) for key in document_ids)
            or len(set(document_ids)) != len(document_ids)):
        raise KnowledgeError("Explicit unique document IDs required", 400)
    wanted = set(document_ids)
    selected = [version for version in versions if version["document_id"] in wanted]
    if {version["document_id"] for version in selected} != wanted:
        raise KnowledgeError("Selected document is absent or not ready", 409)
    return selected


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def write_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def read_checked(path: Path, expected: str) -> bytes:
    from request_budget import checkpoint
    parts=[];checksum=sha256()
    with path.open('rb') as stream:
        while True:
            checkpoint()
            part=stream.read(1024*1024)
            if not part:break
            parts.append(part);checksum.update(part)
    if checksum.hexdigest() != expected:
        raise KnowledgeError(f"Artifact checksum mismatch: {path.name}", 503)
    return b''.join(parts)


@dataclass(frozen=True)
class ProcessingConfig:
    child_chars: int = 500
    overlap_chars: int = 100
    trim_line_edges: bool = True
    skip_page_labels: bool = False

    def __post_init__(self):
        if not 0 <= self.overlap_chars < self.child_chars <= 4000:
            raise ValueError("Invalid child/overlap length")


def process_document(source: str, data: bytes, version_id: str, config: ProcessingConfig) -> dict:
    """Offsets are code-point spans in each decoded TXT or extracted PDF page."""
    try:
        if source.lower().endswith(".pdf"):
            pdf = PdfReader(BytesIO(data))
            pages = [(number, page.extract_text() or "") for number, page in enumerate(pdf.pages, 1)]
        elif source.lower().endswith(".txt"):
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("gb18030")
            pages = [(None, text)]
        else:
            raise KnowledgeError("Only TXT and PDF are supported")
    except KnowledgeError:
        raise
    except Exception as exc:
        raise KnowledgeError("Document parsing failed") from exc
    parents, children, removed, projections = [], [], [], []
    for page, original in pages:
        normalized = []
        offsets = []
        offset = 0
        for line in original.splitlines(keepends=True):
            body = line.rstrip("\r\n")
            trimmed = body.strip() if config.trim_line_edges else body
            if config.skip_page_labels and page is not None and re.fullmatch(r"(?:Page\s+\d+|第\s*\d+\s*页)", trimmed, re.I):
                removed.append({"page": page, "start": offset, "end": offset + len(body), "reason": "explicit_page_label"})
                trimmed = ""
            start = offset + (body.find(trimmed) if trimmed else 0)
            if trimmed:
                projections.append({"page": page, "clean_start": len(normalized),
                                    "clean_end": len(normalized) + len(trimmed),
                                    "raw_start": start, "raw_end": start + len(trimmed)})
            normalized.extend(trimmed)
            offsets.extend((start + i, start + i + 1) for i in range(len(trimmed)))
            if len(line) > len(body):
                projections.append({"page": page, "clean_start": len(normalized),
                                    "clean_end": len(normalized) + 1,
                                    "raw_start": offset + len(body), "raw_end": offset + len(line)})
                normalized.append("\n")
                offsets.append((offset + len(body), offset + len(line)))
            offset += len(line)
        text = "".join(normalized)
        paragraph_start = 0
        for boundary in re.finditer(r"\n(?:[ \t]*\n)+|\Z", text):
            segment = text[paragraph_start:boundary.start()]
            start = paragraph_start + len(segment) - len(segment.lstrip())
            end = paragraph_start + len(segment.rstrip())
            paragraph_start = boundary.end()
            if start >= end:
                continue
            paragraph = text[start:end]
            parent_id = f"{version_id}-p{len(parents) + 1}"
            parent = {"parent_id": parent_id, "text": paragraph, "page": page,
                      "start": offsets[start][0], "end": offsets[end - 1][1],
                      "clean_start": start, "clean_end": end,
                      "document_version_id": version_id, "content_hash": digest(paragraph.encode())}
            parents.append(parent)
            for left in range(start, end, config.child_chars - config.overlap_chars):
                right = min(left + config.child_chars, end)
                value = text[left:right]
                if value.strip():
                    ordinal = len(children) + 1
                    children.append({"child_id": f"{version_id}-c{ordinal}", "chunk_id": ordinal,
                                     "parent_id": parent_id, "document_version_id": version_id,
                                     "text": value, "page": page, "start": offsets[left][0],
                                     "clean_start": left, "clean_end": right,
                                     "end": offsets[right - 1][1], "content_hash": digest(value.encode())})
                if right == end:
                    break
    if not children:
        raise KnowledgeError("Document has no usable text")
    return {"source": source, "pages": [{"page": page, "text": raw} for page, raw in pages],
            "parents": parents, "children": children, "removed": removed, "projections": projections,
            "processing_config": vars(config)}


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str | None
    corpus_snapshot_id: str
    generation: int
    index: object
    vectors: object
    all_docs: tuple
    manifest: dict
    sparse: dict | None = None
    sparse_error: str | None = None
    graph_lookup: object = None
    dense_binding: dict | None = None

    def cache_key(self, endpoint: str, legacy_key: str, model_config: str, namespace: str) -> str:
        payload = [endpoint, self.corpus_snapshot_id, self.snapshot_id, self.generation, model_config, legacy_key]
        return f"rag:v2:{namespace}:{self.generation}:{digest(json_bytes(payload))}"


class KnowledgeStore:
    def __init__(self, root: Path, connect, embedder, model_identity: str,
                 processing: ProcessingConfig = ProcessingConfig()):
        self.root = root
        self.connect = connect
        self.embedder = embedder
        self.model_identity = model_identity
        self.processing = processing
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def transaction(self):
        db = self.connect()
        try:
            with db.cursor() as cursor:
                yield cursor
            db.commit()
        except BaseException:
            try:
                db.rollback()
            except Exception:
                logger.exception("Rollback failed")
            raise
        finally:
            db.close()

    def head(self) -> dict:
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM rag_v2_head WHERE id=1")
            return cursor.fetchone()

    def _lock_head(self, cursor):
        cursor.execute("SELECT * FROM rag_v2_head WHERE id=1 FOR UPDATE")
        return cursor.fetchone()

    def documents(self) -> list:
        with self.transaction() as cursor:
            cursor.execute("""SELECT d.*, v.revision, v.content_hash, v.legacy_file_hash, v.policy_version,
                              v.document_status, v.category, v.origin FROM rag_v2_documents d
                              JOIN rag_v2_document_versions v ON v.document_version_id=d.current_version_id
                              ORDER BY d.source""")
            return cursor.fetchall()

    def versions(self, document_id: str) -> list:
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM rag_v2_document_versions WHERE document_id=%s ORDER BY revision DESC", (document_id,))
            return cursor.fetchall()

    def ingest(self, source: str, data: bytes, policy_version: str | None = None,
               document_id: str | None = None, expected_version: str | None = None,
               *, category: str = "general", origin: str = "upload") -> dict:
        if not source or source != Path(source).name or any(ch in source for ch in "/\\") or len(source) > 255:
            raise KnowledgeError("Invalid filename")
        if not data or len(data) > MAX_UPLOAD or len(policy_version or "") > 64 or len(category) > 64 or len(origin) > 255:
            raise KnowledgeError("Invalid document size or policy version")
        if document_id and not expected_version:
            raise KnowledgeError("Replacement requires expected_version")
        version_id = uuid.uuid4().hex
        content_hash = digest(data)
        record = process_document(source, data, version_id, self.processing)
        processed = json_bytes(record)
        with self.transaction() as cursor:
            self._lock_head(cursor)
            cursor.execute("SELECT * FROM rag_v2_documents WHERE source=%s", (source,))
            previous = cursor.fetchone()
            if document_id and (not previous or previous["document_id"] != document_id):
                raise KnowledgeError("Document/source mismatch", 404)
            if expected_version and (not previous or previous["current_version_id"] != expected_version):
                raise KnowledgeError("Document version changed", 409)
            revision = 1
            if previous:
                document_id = previous["document_id"]
                cursor.execute("SELECT * FROM rag_v2_document_versions WHERE document_version_id=%s", (previous["current_version_id"],))
                current = cursor.fetchone()
                if (current["document_status"] == "ready" and current["content_hash"] == content_hash
                        and current["policy_version"] == policy_version and current["category"] == category
                        and current["origin"] == origin and current["processed_sha"]):
                    old_record = json.loads(read_checked(
                        self.root / "documents" / current["document_version_id"] / "processed.json",
                        current["processed_sha"]))
                    if old_record["processing_config"] == vars(self.processing):
                        return current | {"chunk_count": len(old_record["children"])}
                revision = current["revision"] + 1
            else:
                document_id = uuid.uuid4().hex
                cursor.execute("INSERT INTO rag_v2_documents(document_id,source) VALUES(%s,%s)", (document_id, source))
            folder = self.root / "documents" / version_id
            folder.mkdir(parents=True)
            write_file(folder / "source.bin", data)
            write_file(folder / "processed.json", processed)
            cursor.execute("""INSERT INTO rag_v2_document_versions
                (document_version_id,document_id,revision,content_hash,legacy_file_hash,policy_version,category,origin,document_status,processed_sha)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,'ready',%s)""",
                           (version_id, document_id, revision, content_hash, md5(data).hexdigest(),
                            policy_version, category, origin, digest(processed)))
            cursor.execute("UPDATE rag_v2_documents SET current_version_id=%s WHERE document_id=%s", (version_id, document_id))
            cursor.execute("UPDATE rag_v2_head SET draft_revision=draft_revision+1 WHERE id=1")
        return {"document_id": document_id, "document_version_id": version_id, "revision": revision,
                "content_hash": content_hash, "document_status": "ready", "chunk_count": len(record["children"])}

    def delete(self, document_id: str, expected_version: str) -> dict:
        with self.transaction() as cursor:
            self._lock_head(cursor)
            cursor.execute("""SELECT v.* FROM rag_v2_documents d JOIN rag_v2_document_versions v
                ON d.current_version_id=v.document_version_id WHERE d.document_id=%s""", (document_id,))
            row = cursor.fetchone()
            if not row:
                raise KnowledgeError("Unknown document", 404)
            if row["document_version_id"] != expected_version:
                raise KnowledgeError("Document version changed", 409)
            if row["document_status"] == "deleted":
                return row
            tombstone = uuid.uuid4().hex
            cursor.execute("""INSERT INTO rag_v2_document_versions
                (document_version_id,document_id,revision,content_hash,legacy_file_hash,policy_version,document_status)
                VALUES(%s,%s,%s,%s,%s,%s,'deleted')""",
                           (tombstone, document_id, row["revision"] + 1, row["content_hash"], row["legacy_file_hash"], row["policy_version"]))
            cursor.execute("UPDATE rag_v2_documents SET current_version_id=%s WHERE document_id=%s", (tombstone, document_id))
            cursor.execute("UPDATE rag_v2_head SET draft_revision=draft_revision+1 WHERE id=1")
            return {"document_id": document_id, "document_version_id": tombstone, "document_status": "deleted"}

    def build(self, document_ids=None) -> dict:
        snapshot_id = uuid.uuid4().hex
        with self.transaction() as cursor:
            head = self._lock_head(cursor)
            cursor.execute("""SELECT d.source, v.* FROM rag_v2_documents d JOIN rag_v2_document_versions v
                              ON d.current_version_id=v.document_version_id WHERE v.document_status='ready'
                              ORDER BY d.document_id""")
            versions = cursor.fetchall()
            versions = select_document_versions(versions, document_ids)
            corpus_id = digest(json_bytes([(v["document_version_id"], v["content_hash"]) for v in versions]))
            cursor.execute("""INSERT INTO rag_v2_snapshots
                (snapshot_id,corpus_snapshot_id,draft_revision,base_generation,build_status,validation_status)
                VALUES(%s,%s,%s,%s,'building','pending')""",
                           (snapshot_id, corpus_id, head["draft_revision"], head["generation"]))
        stage = self.root / "staging" / snapshot_id
        final = self.root / "snapshots" / snapshot_id
        try:
            stage.mkdir(parents=True)
            final.parent.mkdir(parents=True, exist_ok=True)
            parents, chunks, members = [], [], []
            for v in versions:
                folder = self.root / "documents" / v["document_version_id"]
                read_checked(folder / "source.bin", v["content_hash"])
                record = json.loads(read_checked(folder / "processed.json", v["processed_sha"]))
                parents.extend(record["parents"])
                members.append({key: v[key] for key in ("source", "document_id", "document_version_id",
                                                       "content_hash", "legacy_file_hash", "policy_version",
                                                       "category", "origin", "revision", "processed_sha")}
                               | {"chunk_count": len(record["children"]), "processing_config": record["processing_config"],
                                  "created_at": v["created_at"].isoformat(), "updated_at": v["updated_at"].isoformat()})
                for child in record["children"]:
                    chunks.append(child | {"source": v["source"], "file_hash": v["legacy_file_hash"],
                                          "version": v["policy_version"], "row": len(chunks)})
            dim = self.embedder.get_sentence_embedding_dimension()
            vectors = np.empty((0,dim),dtype=np.float32)
            if chunks:
                vectors = np.ascontiguousarray(self.embedder.encode([c["text"] for c in chunks]), dtype=np.float32)
                if vectors.shape != (len(chunks), dim) or not np.isfinite(vectors).all():
                    raise KnowledgeError("Invalid embeddings")
            manifest = {"snapshot_id": snapshot_id, "corpus_snapshot_id": corpus_id,
                        "model_identity": self.model_identity, "dimension": dim,
                        "members": members, "parents": parents, "children": chunks}
            backend=os.getenv("RAG_V3_VECTOR_BACKEND","faiss")
            if backend=="milvus":
                from milvus_store import MilvusSnapshotIndex,metadata_rows
                dense=MilvusSnapshotIndex.connect(os.environ["RAG_V3_MILVUS_URI"],os.environ["RAG_V3_MILVUS_NAMESPACE"],snapshot_id,dim,len(chunks))
                receipt=dense.stage(vectors,chunks)
                receipt['record_hashes']=[r['record_sha256'] for r in metadata_rows(snapshot_id,chunks,vectors)]
                manifest['dense']=receipt
            elif backend=="faiss":
                index=faiss.IndexFlatL2(dim)
                if chunks:index.add(vectors)
                encoded=faiss.serialize_index(index).tobytes()
                write_file(stage / "index.faiss", encoded)
                manifest['index_sha']=digest(encoded)
            else:raise KnowledgeError("Unknown vector backend; fallback forbidden")
            sparse_bytes = json_bytes(build_sparse(snapshot_id, corpus_id, chunks))
            write_file(stage / "bm25.json", sparse_bytes)
            manifest["bm25"] = {"sha256": digest(sparse_bytes), "profile": PROFILE}
            raw = json_bytes(manifest)
            write_file(stage / "manifest.json", raw)
            self._load(stage, snapshot_id, digest(raw), head["generation"], strict_sparse=True)
            stage.rename(final)
            with self.transaction() as cursor:
                cursor.execute("""UPDATE rag_v2_snapshots SET build_status='ready',validation_status='passed',
                    manifest_sha=%s WHERE snapshot_id=%s""", (digest(raw), snapshot_id))
        except Exception as exc:
            logger.exception("Snapshot build failed: %s", snapshot_id)
            try:
                with self.transaction() as cursor:
                    cursor.execute("""UPDATE rag_v2_snapshots SET build_status='failed',validation_status='failed',
                        error_message=%s WHERE snapshot_id=%s AND build_status='building'""",
                                   (type(exc).__name__, snapshot_id))
            except Exception:
                logger.exception("Could not record failed build: %s", snapshot_id)
            raise KnowledgeError(f"Snapshot build failed ({snapshot_id})", 503) from exc
        return self.snapshot_status(snapshot_id)

    def document_artifact(self, version_id: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{32}", version_id):
            raise KnowledgeError("Invalid document version", 404)
        with self.transaction() as cursor:
            cursor.execute("SELECT processed_sha FROM rag_v2_document_versions WHERE document_version_id=%s", (version_id,))
            row = cursor.fetchone()
        if not row or not row["processed_sha"]:
            raise KnowledgeError("No processed artifact", 404)
        return json.loads(read_checked(self.root / "documents" / version_id / "processed.json", row["processed_sha"]))

    def snapshot_status(self, snapshot_id: str) -> dict:
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM rag_v2_snapshots WHERE snapshot_id=%s", (snapshot_id,))
            row = cursor.fetchone()
        if not row:
            raise KnowledgeError("Unknown snapshot", 404)
        return row

    def snapshots(self) -> list:
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM rag_v2_snapshots ORDER BY created_at DESC")
            return cursor.fetchall()

    def publications(self) -> list:
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM rag_v2_publications ORDER BY publish_generation DESC")
            return cursor.fetchall()

    def publication_tasks(self) -> list:
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM rag_v2_publication_tasks ORDER BY created_at DESC")
            return cursor.fetchall()

    def _load(self, path: Path, snapshot_id: str, manifest_sha: str, generation: int, *, strict_sparse=False, dense_override=None) -> Snapshot:
        manifest = json.loads(read_checked(path / "manifest.json", manifest_sha))
        if manifest["snapshot_id"] != snapshot_id or manifest["model_identity"] != self.model_identity:
            raise KnowledgeError("Snapshot identity/model mismatch", 503)
        dense=manifest.get('dense')
        if dense_override is not None:
            if dense is not None:raise KnowledgeError('Ambiguous dense binding',503)
            dense=dense_override
        if dense is not None:
            from milvus_store import MilvusSnapshotIndex,collection_name
            if (dense.get('backend')!='milvus' or dense.get('snapshot_id')!=snapshot_id
                    or dense.get('metric')!='L2' or dense.get('index_type')!='FLAT'
                    or dense.get('dimension')!=manifest['dimension']
                    or dense.get('count')!=len(manifest['children'])
                    or len(dense.get('record_hashes',[]))!=dense['count']):
                raise KnowledgeError('Invalid Milvus snapshot binding',503)
            namespace=os.environ['RAG_V3_MILVUS_NAMESPACE']
            if dense['collection']!=collection_name(namespace,snapshot_id):
                raise KnowledgeError('Unexpected Milvus collection identity',503)
            from pymilvus.exceptions import MilvusException
            expected=[{'id':row,'record_sha256':proof} for row,proof in enumerate(dense['record_hashes'])]
            try:
                index=MilvusSnapshotIndex.connect(os.environ['RAG_V3_MILVUS_URI'],namespace,snapshot_id,manifest['dimension'],dense['count'])
                receipt=index.validate(expected)
            except MilvusException as exc:
                raise KnowledgeError('Milvus snapshot unavailable',503) from exc
            if receipt['records_sha256']!=dense['records_sha256']:
                raise KnowledgeError('Milvus collection digest mismatch',503)
        else:
            index = faiss.deserialize_index(np.frombuffer(read_checked(path / "index.faiss", manifest["index_sha"]), dtype=np.uint8))
        chunks = manifest["children"]
        parents = {p["parent_id"]: p for p in manifest["parents"]}
        members = {m["document_version_id"]: m for m in manifest["members"]}
        if (index.ntotal != len(chunks) or index.d != manifest["dimension"]
                or index.d != self.embedder.get_sentence_embedding_dimension()
                or (dense is None and not isinstance(index, faiss.IndexFlatL2)) or len(parents) != len(manifest["parents"])
                or len(members) != len(manifest["members"])):
            raise KnowledgeError("Snapshot index or manifest shape mismatch", 503)
        if any(p["document_version_id"] not in members or digest(p["text"].encode()) != p["content_hash"]
               for p in parents.values()):
            raise KnowledgeError("Invalid Parent mapping", 503)
        docs, seen = [], set()
        for row, child in enumerate(chunks):
            from request_budget import checkpoint
            checkpoint()
            parent = parents.get(child["parent_id"])
            member = members.get(child["document_version_id"])
            if (child["row"] != row or child["child_id"] in seen or not parent or not member
                    or parent["document_version_id"] != child["document_version_id"]
                    or parent["page"] != child["page"] or not parent["start"] <= child["start"] < child["end"] <= parent["end"]
                    or member["source"] != child["source"] or member["legacy_file_hash"] != child["file_hash"]
                    or member["policy_version"] != child["version"]
                    or digest(child["text"].encode()) != child["content_hash"]):
                raise KnowledgeError("Invalid Child mapping", 503)
            seen.add(child["child_id"])
            docs.append(Document(page_content=child["text"], metadata={
                key: child[key] for key in ("source", "file_hash", "chunk_id", "version", "child_id", "parent_id",
                                            "document_version_id", "page", "start", "end", "content_hash")
            } | {"index_snapshot_id": snapshot_id}))
        sparse = None
        sparse_error = None
        if "bm25" in manifest:
            try:
                if manifest["bm25"]["profile"] != PROFILE:
                    raise ValueError("BM25 tokenizer profile mismatch")
                sparse = json.loads(read_checked(path / "bm25.json", manifest["bm25"]["sha256"]))
                validate_sparse(sparse, snapshot_id, manifest["corpus_snapshot_id"], chunks)
            except BudgetExceeded:
                raise
            except (KnowledgeError, OSError, KeyError, TypeError, ValueError) as exc:
                if strict_sparse:
                    raise KnowledgeError("BM25 artifact unavailable or damaged", 503) from exc
                sparse, sparse_error = None, "bm25_artifact_invalid"
                logger.warning("Dense-only fallback for %s: %s", snapshot_id, exc)
        # Existing routes only use vectors as a readiness marker. FAISS owns their immutable bytes.
        def graph_lookup(view, question, hops, **kwargs):
            from knowledge_graph import lookup
            return lookup(self, view, question, hops, **kwargs)
        return Snapshot(snapshot_id, manifest["corpus_snapshot_id"], generation, index, True, tuple(docs), manifest,
                        sparse, sparse_error, graph_lookup, dense_binding=dense)

    def load_snapshot(self, snapshot_id: str, generation: int = 0, *, strict_sparse=False) -> Snapshot:
        row = self.snapshot_status(snapshot_id)
        if row["build_status"] != "ready" or row["validation_status"] != "passed":
            raise KnowledgeError("Snapshot is not ready", 409)
        try:
            override=None
            if os.getenv('RAG_V3_VECTOR_BACKEND')=='milvus':
                with self.transaction() as cursor:
                    cursor.execute('SELECT receipt_json,receipt_sha,source_manifest_sha FROM rag_v3_vector_bindings WHERE snapshot_id=%s AND state=%s',(snapshot_id,'SEALED'))
                    binding=cursor.fetchone()
                if binding:
                    if binding['source_manifest_sha']!=row['manifest_sha'] or digest(binding['receipt_json'].encode())!=binding['receipt_sha']:
                        raise KnowledgeError('Milvus registry binding mismatch',503)
                    override=json.loads(binding['receipt_json'])
                else:
                    raw=json.loads(read_checked(self.root/'snapshots'/snapshot_id/'manifest.json',row['manifest_sha']))
                    if raw.get('dense',{}).get('backend')!='milvus':
                        raise KnowledgeError('Historical snapshot has not been migrated to Milvus',409)
            snapshot = self._load(self.root / "snapshots" / snapshot_id, snapshot_id, row["manifest_sha"], generation,
                                  strict_sparse=strict_sparse,dense_override=override)
            if snapshot.corpus_snapshot_id != row["corpus_snapshot_id"]:
                raise KnowledgeError("Snapshot corpus identity mismatch", 503)
            if os.getenv("RAG_V3_GRAPH_ENABLED") == "1":
                from knowledge_graph import validate_graph
                with self.transaction() as c:
                    c.execute("SELECT task_status FROM kg_builds WHERE snapshot_id=%s", (snapshot_id,))
                    job = c.fetchone()
                    if job and job["task_status"] == "SEALED":
                        validate_graph(self, snapshot, c)
            return snapshot
        except BudgetExceeded:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            raise KnowledgeError("Snapshot artifacts unavailable or damaged", 503) from exc

    def pin(self) -> Snapshot:
        head = self.head()
        if not head["snapshot_id"]:
            return Snapshot(None, "empty", head["generation"], None, None, (), {"members": [], "parents": [], "children": []})
        return self.load_snapshot(head["snapshot_id"], head["generation"], strict_sparse=True)

    def publish(self, snapshot_id: str, expected_generation: int, idempotency_key: str, *, rollback=False) -> dict:
        operation = "rollback" if rollback else "publish"
        if not idempotency_key or len(idempotency_key) > 120 or expected_generation < 0:
            raise KnowledgeError("Invalid publication request")

        def prior(cursor):
            cursor.execute("SELECT * FROM rag_v2_publications WHERE idempotency_key=%s", (idempotency_key,))
            row = cursor.fetchone()
            if row and (row["snapshot_id"], row["expected_generation"], row["operation"]) != (
                snapshot_id, expected_generation, operation
            ):
                raise KnowledgeError("Idempotency key reused with different arguments", 409)
            return row

        with self.transaction() as cursor:
            done = prior(cursor)
        if done:
            return done
        self.load_snapshot(snapshot_id, strict_sparse=True)
        with self.transaction() as cursor:
            cursor.execute("""INSERT INTO rag_v2_publication_tasks
                (idempotency_key,snapshot_id,expected_generation,operation,task_status)
                VALUES(%s,%s,%s,%s,'pending') ON DUPLICATE KEY UPDATE task_status=task_status""",
                           (idempotency_key, snapshot_id, expected_generation, operation))
            cursor.execute("SELECT * FROM rag_v2_publication_tasks WHERE idempotency_key=%s", (idempotency_key,))
            task = cursor.fetchone()
            if (task["snapshot_id"], task["expected_generation"], task["operation"]) != (
                snapshot_id, expected_generation, operation
            ):
                raise KnowledgeError("Idempotency key reused with different arguments", 409)
            if task["task_status"] == "rejected":
                raise KnowledgeError("Publication already rejected; use a new key and current generation", 409)
        with self.transaction() as cursor:
            head = self._lock_head(cursor)
            done = prior(cursor)
            if done:
                return done
            conflict = None
            if head["generation"] != expected_generation:
                conflict = "Publish generation changed"
            elif rollback:
                cursor.execute("SELECT 1 FROM rag_v2_publications WHERE snapshot_id=%s LIMIT 1", (snapshot_id,))
                if not cursor.fetchone():
                    conflict = "Rollback requires a previously published snapshot"
            else:
                cursor.execute("SELECT draft_revision,base_generation FROM rag_v2_snapshots WHERE snapshot_id=%s", (snapshot_id,))
                snapshot = cursor.fetchone()
                if (snapshot["draft_revision"], snapshot["base_generation"]) != (head["draft_revision"], head["generation"]):
                    conflict = "Snapshot was built from a stale draft or generation"
                if os.getenv("RAG_V3_GRAPH_ENABLED") == "1" and not conflict:
                    cursor.execute("SELECT task_status FROM kg_builds WHERE snapshot_id=%s", (snapshot_id,))
                    graph = cursor.fetchone()
                    if graph and graph["task_status"] != "SEALED":
                        conflict = "Graph must be sealed before publishing this snapshot"
            if conflict:
                cursor.execute("UPDATE rag_v2_publication_tasks SET task_status='rejected' WHERE idempotency_key=%s",
                               (idempotency_key,))
            else:
                generation = head["generation"] + 1
                cursor.execute("UPDATE rag_v2_head SET snapshot_id=%s,generation=%s WHERE id=1", (snapshot_id, generation))
                cursor.execute("""INSERT INTO rag_v2_publications
                    (publish_generation,idempotency_key,previous_snapshot_id,snapshot_id,expected_generation,operation,task_status)
                    VALUES(%s,%s,%s,%s,%s,%s,'succeeded')""",
                               (generation, idempotency_key, head["snapshot_id"], snapshot_id, expected_generation, operation))
                cursor.execute("""UPDATE rag_v2_publication_tasks SET task_status='succeeded',publish_generation=%s
                    WHERE idempotency_key=%s""", (generation, idempotency_key))
                committed = prior(cursor)
        if conflict:
            raise KnowledgeError(conflict, 409)
        return committed
