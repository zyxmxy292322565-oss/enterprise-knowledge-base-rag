from request_budget import BudgetExceeded
"""MySQL graph projection for an immutable V2 index snapshot."""
import json
import os
import re
import unicodedata
from datetime import date
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from knowledge_v2 import KnowledgeError, digest, json_bytes, read_checked, write_file
from semantic_reliability import evidence_for

EntityType = Literal["Product", "ProductCategory", "Policy", "PolicyRule", "Condition", "Exception",
                     "Action", "TimeConstraint", "AmountConstraint", "Organization", "Evidence"]
RelationType = Literal["APPLIES_TO", "HAS_CONDITION", "HAS_EXCEPTION", "REQUIRES", "ALLOWS", "EXCLUDES",
                       "SUPERSEDES", "RELATED_TO", "SUPPORTED_BY"]


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: EntityType
    name: str = Field(min_length=1, max_length=150)
    scope: str = Field(min_length=1, max_length=100)
    aliases: list[str] = Field(default_factory=list, max_length=10)


class Assertion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Entity
    target: Entity
    predicate: RelationType
    valid_from: date | None = None
    valid_to: date | None = None
    polarity: Literal["affirmed", "negated"] = "affirmed"
    conditions: list[str] = Field(default_factory=list, max_length=12)
    exceptions: list[str] = Field(default_factory=list, max_length=12)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    evidence_quote: str | None = Field(default=None, min_length=1, max_length=4000)
    evidence_chunk_id: str | None = Field(default=None, max_length=80)


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def alias_matches(alias, query):
    return bool(re.search((r"(?<![a-z0-9_-])" if alias[0].isascii() else "") + re.escape(alias) +
                          (r"(?![a-z0-9_-])" if alias[-1].isascii() else ""),query))


def entity_id(entity):
    return digest(json_bytes([entity.type, normalized(entity.scope), normalized(entity.name)]))


def extract(text):
    """Explicit curated annotations plus conservative eligibility-state extraction."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("KG "):
            yield Assertion.model_validate_json(line[3:])
        elif "ELIGIBLE_TO_APPLY" in line and "不表示退款" in line and "已批准" in line:
            yield Assertion(source=Entity(type="Condition", name="ELIGIBLE_TO_APPLY", scope="after_sales"),
                            target=Entity(type="Action", name="退款已批准", scope="after_sales"),
                            predicate="EXCLUDES")


def mutable(cursor, store, sid):
    store._lock_head(cursor)
    cursor.execute("SELECT 1 FROM rag_v2_publications WHERE snapshot_id=%s LIMIT 1", (sid,))
    if cursor.fetchone():
        raise KnowledgeError("Published graph snapshots are immutable", 409)
    cursor.execute("SELECT * FROM kg_builds WHERE snapshot_id=%s FOR UPDATE", (sid,))
    return cursor.fetchone()


def build(store, view, mode="rules", batch_size=100):
    from graph_extraction import PROMPT_VERSION, SCHEMA_VERSION, extract_llm, model_identity

    sid = view.snapshot_id
    signature = (mode, model_identity() if mode == "llm" else "rules-v1", PROMPT_VERSION, SCHEMA_VERSION)
    if mode == "llm" and (os.getenv("RAG_V3_KG_LLM_ENABLED") != "1" or not signature[1]
                          or not os.getenv("RAG_V3_KG_BASE_URL") or not os.getenv("RAG_V3_KG_API_KEY")):
        raise KnowledgeError("Extraction model not explicitly configured", 409)
    # MySQL advisory lock belongs to this connection and releases on process exit.
    with store.transaction() as lock:
        lock.execute("SELECT GET_LOCK(%s,0) AS acquired", ("kg2:" + sid,))
        if lock.fetchone()["acquired"] != 1:
            raise KnowledgeError("Graph extraction already running", 409)
        return _build_locked(store, view, mode, batch_size, signature, extract_llm)


def _build_locked(store, view, mode, batch_size, signature, extract_llm):
    sid = view.snapshot_id
    with store.transaction() as c:
        existing = mutable(c, store, sid)
        if existing and existing["task_status"] in {"DRAFT", "SEALED"}:
            return {"snapshot_id": sid, "entities": existing["extracted_entities"],
                    "relations": existing["extracted_relations"], "status": existing["task_status"]}
        if not existing:
            c.execute("""INSERT INTO kg_builds(snapshot_id,build_id,corpus_id,task_status,total_chunks)
                         VALUES(%s,%s,%s,'BUILDING',%s)""", (sid, uuid4().hex, view.corpus_snapshot_id, len(view.all_docs)))
            c.execute("INSERT INTO kg_extraction_runs VALUES(%s,%s,%s,%s,%s)", (sid, *signature))
        c.execute("SELECT mode,model,prompt_version,schema_version FROM kg_extraction_runs WHERE snapshot_id=%s", (sid,))
        frozen = c.fetchone()
        if not frozen or tuple(frozen.values()) != signature:
            raise KnowledgeError("Extraction configuration differs from frozen task", 409)
        c.execute("SELECT child_id,status FROM kg_extraction_chunks WHERE snapshot_id=%s", (sid,))
        statuses = {r["child_id"]: r["status"] for r in c.fetchall()}
    pending = [(i, doc) for i, doc in enumerate(view.all_docs) if statuses.get(doc.metadata["child_id"]) != "DONE"]
    pending.sort(key=lambda pair: statuses.get(pair[1].metadata["child_id"]) == "FAILED")
    for i, doc in pending[:batch_size]:
        child_id = doc.metadata["child_id"]
        assertions, failure_reason = [], None
        try:
            if mode == "llm":
                proof = evidence_for(store, view, i)
                assertions = extract_llm(proof["quote"], child_id)
            else:
                assertions = list(extract(doc.page_content))
                proof = evidence_for(store, view, i) if assertions else None
            if mode == "rules" and assertions and any(a not in list(extract(proof["quote"])) for a in assertions):
                raise ValueError("Extracted assertion is not present in original text")
            items = []
            for a in assertions:
                if (a.valid_from and a.valid_to and a.valid_from > a.valid_to) or entity_id(a.source) == entity_id(a.target):
                    raise ValueError("Invalid assertion dates or endpoints")
                binding = proof.copy()
                if mode == "llm":
                    if (a.evidence_chunk_id != child_id or not a.evidence_quote or
                            proof["quote"].count(a.evidence_quote) != 1):
                        raise ValueError("Evidence quote missing, ambiguous or outside Child")
                    if any(normalized(alias) not in normalized(proof["quote"])
                           for node in (a.source, a.target) for alias in node.aliases):
                        raise ValueError("Proposed alias is absent from original Child")
                    binding["start"] += proof["quote"].index(a.evidence_quote)
                    binding["end"] = binding["start"] + len(a.evidence_quote)
                    binding["quote"] = a.evidence_quote
                items.append({"assertion": a.model_dump(mode="json"), "proof": binding})
            status, error = "DONE", None
        except Exception as exc:
            status, error, items = "FAILED", type(exc).__name__, []
            if str(exc) in {"Invalid assertion dates or endpoints",
                            "Evidence quote missing, ambiguous or outside Child",
                            "Proposed alias is absent from original Child",
                            "Extracted assertion is not present in original text"}:
                failure_reason = str(exc)
        if mode == "llm":
            # Preserve rejected model candidates for diagnosis without persisting
            # provider exception bodies, headers or credentials.
            audit_folder = store.root / "graph-extraction-attempts" / sid
            audit_folder.mkdir(parents=True, exist_ok=True)
            write_file(audit_folder / (uuid4().hex + ".json"), json_bytes({
                "snapshot_id": sid, "child_id": child_id, "status": status,
                "error_type": error, "validation_failure": failure_reason,
                "mode": mode, "model": signature[1], "prompt_version": signature[2],
                "schema_version": signature[3], "review_status": "proposed_unreviewed",
                "candidate_assertions": [a.model_dump(mode="json") for a in assertions]}))
        with store.transaction() as c:
            c.execute("""INSERT INTO kg_extraction_chunks(snapshot_id,child_id,status,attempts,result,error_type)
                VALUES(%s,%s,%s,1,%s,%s) ON DUPLICATE KEY UPDATE status=VALUES(status), attempts=attempts+1,
                result=VALUES(result),error_type=VALUES(error_type)""", (sid, child_id, status, json.dumps(items), error))
    with store.transaction() as c:
        c.execute("SELECT status,result FROM kg_extraction_chunks WHERE snapshot_id=%s", (sid,))
        journal = c.fetchall()
        done = sum(r["status"] == "DONE" for r in journal)
        failed = len(journal) - done
        c.execute("""UPDATE kg_builds SET task_status=%s,processed_chunks=%s,failed_chunks=%s WHERE snapshot_id=%s""",
                  ("PARTIAL" if failed or done < len(view.all_docs) else "BUILDING", done, failed, sid))
    if failed or done < len(view.all_docs):
        return {"snapshot_id": sid, "status": "PARTIAL", "processed_chunks": done, "failed_chunks": failed}
    try:
        members = {m["document_version_id"]: m["document_id"] for m in view.manifest["members"]}
        prepared = [(Assertion.model_validate(item["assertion"]), item["proof"],
                     members[item["proof"]["document_version_id"]])
                    for row in journal for item in json.loads(row["result"])]
        entities, relations = set(), set()
        with store.transaction() as c:
            mutable(c, store, sid)
            for a, proof, document_id in prepared:
                if a.valid_from and a.valid_to and a.valid_from > a.valid_to:
                    raise ValueError("Invalid policy date range")
                left, right = entity_id(a.source), entity_id(a.target)
                if left == right:
                    raise ValueError("Self relation")
                for key, node in ((left, a.source), (right, a.target)):
                    entities.add(key)
                    c.execute("""INSERT INTO kg_entities(snapshot_id,entity_id,entity_type,entity_name,normalized_name,description)
                                 VALUES(%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE entity_id=entity_id""",
                              (sid, key, node.type, node.name, normalized(node.name), node.scope))
                    for alias in set(map(normalized, [node.name, *node.aliases])):
                        if not alias or len(alias) > 200:
                            raise ValueError("Invalid alias")
                        c.execute("INSERT IGNORE INTO kg_entity_aliases VALUES(%s,%s,%s)", (sid, key, alias))
                identity = [left, right, a.predicate, str(a.valid_from), str(a.valid_to)]
                if mode == "llm" or a.polarity != "affirmed" or a.conditions or a.exceptions:
                    identity.extend([a.polarity, sorted(set(a.conditions)), sorted(set(a.exceptions))])
                key = digest(json_bytes(identity))
                relations.add(key)
                c.execute("""INSERT INTO kg_relations(snapshot_id,relation_id,source_entity_id,target_entity_id,
                             relation_type,extraction_method,valid_from,valid_to) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
                             ON DUPLICATE KEY UPDATE relation_id=relation_id""",
                          (sid, key, left, right, a.predicate, mode, a.valid_from, a.valid_to))
                c.execute("""INSERT IGNORE INTO kg_relation_semantics VALUES(%s,%s,%s,%s,%s)""",
                          (sid, key, a.polarity, json.dumps(a.conditions), json.dumps(a.exceptions)))
                position = {k: proof[k] for k in ("page", "start", "end")}
                c.execute("""INSERT IGNORE INTO kg_relation_evidence VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                          (sid, key, digest(json_bytes(proof)), document_id, proof["document_version_id"],
                           proof["file_hash"], proof["child_id"], proof["parent_id"], json.dumps(position), proof["quote"]))
            c.execute("""UPDATE kg_builds SET task_status='DRAFT',processed_chunks=total_chunks,
                         extracted_entities=%s,extracted_relations=%s,completed_at=NOW(6) WHERE snapshot_id=%s""",
                      (len(entities), len(relations), sid))
    except Exception as exc:
        with store.transaction() as c:
            c.execute("""UPDATE kg_builds SET task_status='FAILED', failed_chunks=1,error_message=%s,
                         completed_at=NOW(6) WHERE snapshot_id=%s""", (type(exc).__name__, sid))
        raise KnowledgeError("Graph extraction or evidence validation failed", 422) from exc
    return {"snapshot_id": sid, "entities": len(entities), "relations": len(relations)}


def review(store, sid, kind, key, status, actor, note):
    table, field = ("kg_entities", "entity_id") if kind == "entity" else ("kg_relations", "relation_id")
    with store.transaction() as c:
        job = mutable(c, store, sid)
        if not job or job["task_status"] != "DRAFT":
            raise KnowledgeError("Graph not editable", 409)
        if status != "APPROVED":
            c.execute("""SELECT
                EXISTS(SELECT 1 FROM kg_entity_merges WHERE snapshot_id=%s AND undone_at IS NULL)
                OR EXISTS(SELECT 1 FROM kg_policy_versions WHERE snapshot_id=%s) AS governed""", (sid,sid))
            if c.fetchone()["governed"]:
                raise KnowledgeError("Governed graph approval cannot be withdrawn in place", 409)
        c.execute(f"SELECT status FROM {table} WHERE snapshot_id=%s AND {field}=%s", (sid, key))
        row = c.fetchone()
        if not row:
            raise KnowledgeError("Unknown graph record", 404)
        c.execute(f"UPDATE {table} SET status=%s WHERE snapshot_id=%s AND {field}=%s", (status, sid, key))
        c.execute("""INSERT INTO kg_reviews(snapshot_id,entity_kind,object_id,previous_status,status,actor,note)
                     VALUES(%s,%s,%s,%s,%s,%s,%s)""", (sid, kind, key, row["status"], status, actor, note))
    return {"status": status}


def confirm_alias(store, sid, entity, alias, evidence_id, actor, reason):
    alias = normalized(alias)
    with store.transaction() as c:
        job = mutable(c, store, sid)
        if not alias or not job or job["task_status"] != "DRAFT":
            raise KnowledgeError("Alias requires an editable graph", 409)
        c.execute("""SELECT e.evidence_text FROM kg_relation_evidence e JOIN kg_relations r
                     ON e.snapshot_id=r.snapshot_id AND e.relation_id=r.relation_id
                     WHERE e.snapshot_id=%s AND e.evidence_id=%s AND
                     (r.source_entity_id=%s OR r.target_entity_id=%s)""", (sid, evidence_id, entity, entity))
        if not any(alias in normalized(row["evidence_text"]) for row in c.fetchall()):
            raise KnowledgeError("Alias needs original evidence attached to the entity", 422)
        c.execute("INSERT IGNORE INTO kg_entity_aliases VALUES(%s,%s,%s)", (sid, entity, alias))
        if c.rowcount:
            c.execute("""INSERT INTO kg_reviews(snapshot_id,entity_kind,object_id,previous_status,status,actor,note)
                       VALUES(%s,'alias',%s,'ABSENT','APPROVED',%s,%s)""",
                      (sid, entity, actor, json.dumps({"alias": alias, "evidence_id": evidence_id, "reason": reason},
                                                     ensure_ascii=False)))
    return {"entity_id": entity, "alias": alias}


def validate_graph(store, view, c, deadline=None):
    from request_budget import checkpoint
    checkpoint(deadline)
    sid = view.snapshot_id
    children = {doc.metadata["child_id"]: i for i, doc in enumerate(view.all_docs)}
    c.execute("SELECT corpus_id FROM kg_builds WHERE snapshot_id=%s", (sid,))
    job = c.fetchone()
    if not job or job["corpus_id"] != view.corpus_snapshot_id:
        raise KnowledgeError("Graph corpus identity mismatch", 503)
    c.execute("SELECT schema_version FROM kg_extraction_runs WHERE snapshot_id=%s", (sid,))
    schema = c.fetchone()
    from graph_extraction import SCHEMA_VERSION
    if not schema or schema["schema_version"] != SCHEMA_VERSION:
        raise KnowledgeError("Graph schema missing or incompatible", 503)
    c.execute("""SELECT 1 FROM kg_reviews r WHERE r.snapshot_id=%s AND r.entity_kind='inheritance'
        AND r.status IN ('PENDING','NEEDS_REVIEW') AND r.review_id=(SELECT MAX(x.review_id) FROM kg_reviews x
        WHERE x.snapshot_id=r.snapshot_id AND x.entity_kind=r.entity_kind AND x.object_id=r.object_id) LIMIT 1""",(sid,))
    if c.fetchone():
        raise KnowledgeError("Unresolved governance inheritance",409)
    c.execute("SELECT * FROM kg_entities WHERE snapshot_id=%s AND status='APPROVED'", (sid,))
    approved_entities={row["entity_id"]:row for row in c.fetchall()}
    for key, row in approved_entities.items():
        checkpoint(deadline)
        try:
            node=Entity(type=row["entity_type"],name=row["entity_name"],scope=row["description"] or "")
            if entity_id(node)!=key or normalized(node.name)!=row["normalized_name"]:
                raise ValueError("Invalid canonical entity identity")
        except ValueError as exc:
            raise KnowledgeError("Invalid approved entity identity",409) from exc
    c.execute("SELECT * FROM kg_relations WHERE snapshot_id=%s AND status='APPROVED'",(sid,))
    approved_edges=c.fetchall()
    c.execute("SELECT * FROM kg_relation_semantics WHERE snapshot_id=%s",(sid,))
    semantics={r["relation_id"]:r for r in c.fetchall()}
    from typing import get_args
    for edge in approved_edges:
        checkpoint(deadline)
        sem=semantics.get(edge["relation_id"])
        if not sem or edge["relation_type"] not in get_args(RelationType):
            raise KnowledgeError("Relation schema or semantics missing",409)
        conditions=json.loads(sem["conditions_json"])
        exceptions=json.loads(sem["exceptions_json"])
        identity=[edge["source_entity_id"],edge["target_entity_id"],edge["relation_type"],str(edge["valid_from"]),str(edge["valid_to"])]
        if edge["extraction_method"]=="llm" or sem["polarity"]!="affirmed" or conditions or exceptions:
            identity.extend([sem["polarity"],sorted(set(conditions)),sorted(set(exceptions))])
        if digest(json_bytes(identity))!=edge["relation_id"] or sem["polarity"] not in {"affirmed","negated"}:
            raise KnowledgeError("Relation identity mismatch",409)
        if edge["valid_from"] and edge["valid_to"] and edge["valid_to"]<edge["valid_from"]:
            raise KnowledgeError("Invalid relation validity interval",409)
        if edge["relation_type"]=="SUPERSEDES" and any(approved_entities.get(edge[k],{}).get("entity_type") not in {"Policy","PolicyRule"} for k in ("source_entity_id","target_entity_id")):
            raise KnowledgeError("SUPERSEDES requires reviewed policy endpoints",409)
    c.execute("""SELECT COUNT(*) AS n FROM kg_relations r
        LEFT JOIN kg_entities a ON a.snapshot_id=r.snapshot_id AND a.entity_id=r.source_entity_id
        LEFT JOIN kg_entities b ON b.snapshot_id=r.snapshot_id AND b.entity_id=r.target_entity_id
        WHERE r.snapshot_id=%s AND r.status='APPROVED' AND (a.entity_id IS NULL OR b.entity_id IS NULL OR a.status!='APPROVED' OR b.status!='APPROVED'
        OR NOT EXISTS(SELECT 1 FROM kg_relation_evidence e WHERE e.snapshot_id=r.snapshot_id AND e.relation_id=r.relation_id))""", (sid,))
    if c.fetchone()["n"]:
        raise KnowledgeError("Approved edges require approved endpoints and original evidence", 409)
    c.execute("""SELECT e.* FROM kg_relation_evidence e JOIN kg_relations r
        ON r.snapshot_id=e.snapshot_id AND r.relation_id=e.relation_id
        WHERE e.snapshot_id=%s AND r.status='APPROVED'""", (sid,))
    evidence_rows = c.fetchall()
    versions = sorted({row["document_version"] for row in evidence_rows})
    processed_hashes = {}
    if versions:
        c.execute("SELECT document_version_id,processed_sha FROM rag_v2_document_versions WHERE document_version_id IN (" +
                  ",".join(["%s"] * len(versions)) + ")", tuple(versions))
        processed_hashes = {row["document_version_id"]:row["processed_sha"] for row in c.fetchall()}
    artifacts = {}
    def artifact(version):
        if version not in artifacts:
            expected = processed_hashes.get(version)
            if not expected:
                raise KnowledgeError("Missing processed source identity",409)
            artifacts[version] = json.loads(read_checked(store.root / "documents" / version / "processed.json", expected))
        return artifacts[version]
    from types import SimpleNamespace
    members_by_version={m['document_version_id']:m for m in view.manifest['members']}
    parents_by_id={p['parent_id']:p for p in view.manifest['parents']}
    checked_sources=set()
    def verify_source(version,expected):
        checkpoint(deadline)
        if version not in checked_sources:
            read_checked(store.root/'documents'/version/'source.bin',expected)
            checked_sources.add(version)
    source_store = SimpleNamespace(root=store.root,document_artifact=artifact,
        evidence_members=members_by_version,evidence_parents=parents_by_id,verify_source=verify_source) if versions else store
    proofs = {}
    for row in evidence_rows:
        checkpoint(deadline)
        try:
            idx = children[row["chunk_id"]]
            if idx not in proofs:
                proofs[idx] = evidence_for(source_store, view, idx)
            proof = proofs[idx]
            position = json.loads(row["source_position"])
            start, end = position["start"], position["end"]
            if (row["document_id"] != members_by_version[proof['document_version_id']]['document_id']
                    or row["document_version"] != proof["document_version_id"]
                    or row["file_hash"] != proof["file_hash"] or row["parent_id"] != proof["parent_id"]
                    or position["page"] != proof["page"] or not proof["start"] <= start < end <= proof["end"]
                    or proof["quote"][start - proof["start"]:end - proof["start"]] != row["evidence_text"]):
                raise ValueError("Invalid original evidence binding")
        except BudgetExceeded:
            raise
        except (KeyError, OSError, ValueError, TypeError, StopIteration) as exc:
            raise KnowledgeError("Approved graph evidence cannot be verified against its source", 409) from exc


    # Policy authority is part of the sealed graph, including explicit supersession proof.
    from graph_governance import Policy
    c.execute("SELECT * FROM kg_policy_versions WHERE snapshot_id=%s",(sid,))
    policies = c.fetchall()
    by_relation = {row["relation_id"]:row for row in policies}
    edge_map = {row["relation_id"]:row for row in approved_edges}
    evidence_map = {(row['relation_id'],row["evidence_id"]):row for row in evidence_rows}
    evidence_groups={}
    for proof in evidence_rows:
        checkpoint(deadline)
        evidence_groups.setdefault(proof['evidence_id'],[]).append(proof)
    for row in policies:
        checkpoint(deadline)
        try:
            meta = json.loads(row["metadata_json"])
            item = Policy.model_validate({key:value for key,value in meta.items() if key in Policy.model_fields})
            edge = edge_map[row["relation_id"]]
            proof = evidence_map[(row['relation_id'],row["evidence_id"])]
            if (meta.get("review_status") != "APPROVED" or not meta.get("reviewed_by")
                    or any(getattr(item,key)!=row[key] for key in ("policy_id","policy_version","policy_entity_id","relation_id","evidence_id"))
                    or proof["relation_id"]!=row["relation_id"]
                    or row["policy_entity_id"] not in (edge["source_entity_id"],edge["target_entity_id"])
                    or approved_entities.get(row["policy_entity_id"],{}).get("entity_type") not in {"Policy","PolicyRule"}):
                raise ValueError("Invalid policy authority binding")
            if item.supersedes_relation_id:
                old = by_relation[item.supersedes_relation_id]
                old_meta = json.loads(old["metadata_json"])
                replacements=[proof for proof in evidence_groups.get(item.supersession_proof_id,[])
                    if edge_map.get(proof['relation_id'],{}).get('relation_type')=='SUPERSEDES'
                    and edge_map[proof['relation_id']]['source_entity_id']==item.policy_entity_id
                    and edge_map[proof['relation_id']]['target_entity_id']==old['policy_entity_id']]
                if len(replacements)!=1:raise ValueError('Supersession proof identity is ambiguous')
                replacement = replacements[0]
                supersedes = edge_map[replacement["relation_id"]]
                if (supersedes["relation_type"]!="SUPERSEDES" or supersedes["source_entity_id"]!=item.policy_entity_id
                        or supersedes["target_entity_id"]!=old["policy_entity_id"]
                        or old["policy_id"]!=item.policy_id or old["policy_version"]==item.policy_version
                        or old_meta["valid_from"]>=item.valid_from.isoformat()
                        or any(old_meta[key]!=meta[key] for key in ("scope","applicable_products","applicable_categories","conditions","exceptions"))):
                    raise ValueError("Invalid supersession authority")
        except (ValueError,KeyError,TypeError) as exc:
            raise KnowledgeError("Policy governance cannot be verified against approved evidence",409) from exc


def seal(store, sid):
    view = store.load_snapshot(sid, strict_sparse=True)
    children = {doc.metadata["child_id"]: i for i, doc in enumerate(view.all_docs)}
    with store.transaction() as c:
        job = mutable(c, store, sid)
        if not job or job["task_status"] != "DRAFT":
            raise KnowledgeError("Graph not sealable", 409)
        c.execute("""SELECT 1 FROM kg_reviews r WHERE r.snapshot_id=%s AND r.entity_kind='inheritance'
            AND r.status IN ('PENDING','NEEDS_REVIEW') AND r.review_id=(SELECT MAX(x.review_id) FROM kg_reviews x
            WHERE x.snapshot_id=r.snapshot_id AND x.entity_kind=r.entity_kind AND x.object_id=r.object_id) LIMIT 1""",(sid,))
        if c.fetchone():
            raise KnowledgeError("Inheritance requires explicit review or dismissal",409)
        validate_graph(store, view, c)
        c.execute("UPDATE kg_builds SET task_status='SEALED' WHERE snapshot_id=%s", (sid,))
    return {"snapshot_id": sid, "status": "SEALED"}


def conflicts_for(edges):
    """Conservative contradiction candidates, without choosing policy authority."""
    groups, output = {}, []
    for edge in edges:
        predicate = edge["relation_type"]
        if predicate not in {"ALLOWS", "EXCLUDES", "REQUIRES"}:
            continue
        family = "permission" if predicate in {"ALLOWS", "EXCLUDES"} else predicate
        conditions = tuple(sorted(set(edge.get("conditions", []))))
        exceptions = tuple(sorted(set(edge.get("exceptions", []))))
        key = (edge["source_entity_id"], edge["target_entity_id"], family, conditions, exceptions)
        negative = (predicate == "EXCLUDES") != (edge.get("polarity") == "negated")
        for old, old_negative in groups.get(key, []):
            if (negative != old_negative
                    and not (old.get("valid_to") and edge.get("valid_from") and old["valid_to"] < edge["valid_from"])
                    and not (edge.get("valid_to") and old.get("valid_from") and edge["valid_to"] < old["valid_from"])):
                output.append([old["relation_id"], edge["relation_id"]])
        groups.setdefault(key, []).append((edge, negative))
    return output


def with_semantics(cursor, sid, edges):
    cursor.execute("SELECT * FROM kg_relation_semantics WHERE snapshot_id=%s", (sid,))
    meta = {row["relation_id"]: row for row in cursor.fetchall()}
    for edge in edges:
        row = meta.get(edge["relation_id"], {})
        edge.update(polarity=row.get("polarity", "affirmed"),
                    conditions=json.loads(row.get("conditions_json", "[]")),
                    exceptions=json.loads(row.get("exceptions_json", "[]")))


def projected_paths(paths, proofs):
    """Deduplicate canonical path shape while retaining all original proof rows."""
    result = {}
    for nodes, edges in paths:
        ids = [e["relation_id"] for e in edges]
        if any(key not in proofs for key in ids):
            continue
        shape = (tuple(nodes), tuple((e["source_entity_id"],e["target_entity_id"],e["relation_type"],
                 e.get("valid_from"),e.get("valid_to"),e.get("polarity","affirmed"),
                 tuple(sorted(e.get("conditions",[]))),tuple(sorted(e.get("exceptions",[])))) for e in edges))
        item = result.setdefault(shape, {"entity_path": nodes,"relation_path": edges,"relation_ids": ids,
                                        "equivalent_relation_paths":[],"evidence":[],"evidence_chunk_ids":[]})
        if ids != item["relation_ids"] and ids not in item["equivalent_relation_paths"]:
            item["equivalent_relation_paths"].append(ids)
        for key in ids:
            for proof in proofs[key]:
                if proof not in item["evidence"]:
                    item["evidence"].append(proof)
                if proof["chunk_id"] not in item["evidence_chunk_ids"]:
                    item["evidence_chunk_ids"].append(proof["chunk_id"])
    return list(result.values())


def adjacency_sql(group_size):
    binds = ",".join(["%s"] * group_size)
    columns = "relation_id,source_entity_id,target_entity_id,relation_type,valid_from,valid_to"
    def branch(index, endpoint):
        return f"""(SELECT {columns} FROM kg_relations FORCE INDEX ({index})
            WHERE snapshot_id=%s AND status='APPROVED' AND relation_type NOT IN ('RELATED_TO','SUPERSEDES')
            AND {endpoint} IN ({binds})
            AND (valid_from IS NULL OR valid_from<=%s) AND (valid_to IS NULL OR valid_to>=%s)
            ORDER BY relation_id LIMIT 21)"""
    # The global first 21 must occur within the first 21 of either ordered branch.
    return "SELECT * FROM (" + branch("outbound","source_entity_id") + " UNION " + branch("inbound","target_entity_id") + ") AS neighbors ORDER BY relation_id LIMIT 21"


def resolve_seeds(aliases, question, overlay, context, entities):
    """Return every unique grounded seed; unresolved candidates never become seeds."""
    matches={}
    for row in aliases:
        if alias_matches(row['alias'],question):
            matches.setdefault(row['alias'],set()).add(overlay.get(row['entity_id'],row['entity_id']))
    seeds, ambiguous, details=[],[],[]
    for alias,ids in matches.items():
        candidates=[entities[key] for key in sorted(ids) if key in entities]
        retained=candidates
        basis=[]
        if len(retained)>1 and context.get('entity_type'):
            retained=[r for r in retained if r['entity_type']==context['entity_type']];basis.append('entity_type')
        if len(retained)>1 and context.get('scope'):
            scoped=[r for r in retained if normalized(r.get('description') or '')==normalized(context['scope'])]
            if scoped:retained=scoped;basis.append('scope')
        if len(retained)>1 and context.get('product'):
            scoped=[r for r in retained if r['entity_type']=='Product' and normalized(r['entity_name'])==normalized(context['product'])]
            if scoped:retained=scoped;basis.append('product')
        if len(retained)>1 and context.get('category'):
            scoped=[r for r in retained if r['entity_type']=='ProductCategory' and normalized(r['entity_name'])==normalized(context['category'])]
            if scoped:retained=scoped;basis.append('category')
        if len(retained)>1 and context.get('_eligible_policy_entities'):
            scoped=[r for r in retained if r['entity_id'] in context['_eligible_policy_entities']]
            if scoped:retained=scoped;basis.append('reviewed_policy_applicability')
        detail={'alias':alias,'candidates':[{k:r.get(k) for k in ['entity_id','entity_type','entity_name','description']} for r in candidates],
                'matching_basis':['reviewed_alias',*basis],'resolved_entity_id':retained[0]['entity_id'] if len(retained)==1 else None}
        details.append(detail)
        if len(retained)!=1:ambiguous.append(alias)
        elif retained[0]['entity_id'] not in seeds:seeds.append(retained[0]['entity_id'])
    return seeds,ambiguous,details

def lookup(store, view, question, hops=2, *, limit=100, deadline=None, policy_context=None, timeout_ms=1000):
    from graph_governance import active_merges, policy_decisions, policy_rows

    if hops not in (1,2) or not 1 <= limit <= 100 or not 50 <= timeout_ms <= 5000:
        raise KnowledgeError("Graph query bounds exceeded",422)
    started = perf_counter()
    end = min(deadline or started + timeout_ms / 1000, started + timeout_ms / 1000)
    sid = view.snapshot_id
    paths, seeds, ambiguous = [], [], []
    with store.transaction() as c:
        c.execute("SET SESSION MAX_EXECUTION_TIME=200")
        c.execute("""SELECT b.corpus_id FROM kg_builds b WHERE b.snapshot_id=%s AND b.task_status='SEALED'
                     AND EXISTS(SELECT 1 FROM rag_v2_publications p WHERE p.snapshot_id=b.snapshot_id)""", (sid,))
        job = c.fetchone()
        if not job or job["corpus_id"] != view.corpus_snapshot_id:
            raise KnowledgeError("Graph unavailable for pinned snapshot", 409)
        context = policy_context or {}
        as_of = date.fromisoformat(context["as_of"]).isoformat() if context.get("as_of") else None
        if not as_of:
            c.execute("SELECT UTC_DATE() AS today")
            as_of = c.fetchone()["today"].isoformat()
        from request_budget import checkpoint
        validate_graph(store, view, c, deadline=end)
        checkpoint(end)
        overlay = active_merges(c, sid, context.get('at_time'))
        policy_records=policy_rows(c,sid)
        decisions = policy_decisions(policy_records, as_of, context)
        context=context|{'_eligible_policy_entities':{p['policy_entity_id'] for p in policy_records if decisions.get(p['relation_id'])=='APPLICABLE'}}
        q = normalized(question)
        # Limit exact substring matching to the indexed alias vocabulary of this small MVP.
        c.execute("""SELECT a.alias,a.entity_id FROM kg_entity_aliases a JOIN kg_entities e
                     ON e.snapshot_id=a.snapshot_id AND e.entity_id=a.entity_id
                     WHERE a.snapshot_id=%s AND e.status='APPROVED' AND LOCATE(a.alias,%s)>0 ORDER BY LENGTH(a.alias) DESC,a.alias,a.entity_id LIMIT 5001""", (sid,q))
        aliases = c.fetchall()
        if len(aliases) > 5000:
            raise KnowledgeError("Alias budget exceeded", 409)
        c.execute("SELECT entity_id,entity_type,entity_name,description FROM kg_entities WHERE snapshot_id=%s AND status='APPROVED'",(sid,))
        entities={r['entity_id']:r for r in c.fetchall()}
        all_seeds,ambiguous,entity_matches=resolve_seeds(aliases,q,overlay,context,entities)
        seeds=all_seeds[:5]
        frontier = [([seed], []) for seed in seeds]
        expanded = set()
        truncated = len(all_seeds)>len(seeds)
        executed_seeds=set()
        for _ in range(hops):
            next_frontier = []
            for nodes, edges in frontier:
                checkpoint()
                if perf_counter() >= end or len(paths) >= limit:
                    truncated=True
                    break
                node = nodes[-1]
                if node in expanded:
                    continue
                expanded.add(node)
                if not edges:executed_seeds.add(node)
                group = [node, *(key for key, target in overlay.items() if target == node)]
                c.execute(adjacency_sql(len(group)), (sid,*group,as_of,as_of,sid,*group,as_of,as_of))
                edge_rows = c.fetchall()
                truncated = truncated or len(edge_rows)>20
                for edge in edge_rows[:20]:
                    checkpoint()
                    if decisions.get(edge["relation_id"], "APPLICABLE") != "APPLICABLE":
                        continue
                    for key in ("source_entity_id", "target_entity_id"):
                        edge["original_" + key] = edge[key]
                        edge[key] = overlay.get(edge[key], edge[key])
                    for key in ("valid_from", "valid_to"):
                        edge[key] = edge[key].isoformat() if edge[key] else None
                    target = edge["target_entity_id"] if edge["source_entity_id"] == node else edge["source_entity_id"]
                    if target in nodes:
                        continue
                    path = (nodes + [target], edges + [edge])
                    paths.append(path)
                    next_frontier.append(path)
                    if len(paths) >= limit:
                        break
            frontier = next_frontier
        ids = sorted({e["relation_id"] for _, edges in paths for e in edges})
        proofs = {}
        if ids and perf_counter() < end:
            c.execute("SELECT * FROM kg_relation_evidence WHERE snapshot_id=%s AND relation_id IN (" +
                      ",".join(["%s"] * len(ids)) + ") ORDER BY relation_id,evidence_id LIMIT 400", (sid, *ids))
            evidence_rows = c.fetchall()
            truncated = truncated or len(evidence_rows) >= 400
            rows = {d.metadata["child_id"]: i for i, d in enumerate(view.all_docs)}
            for row in evidence_rows:
                checkpoint(end)
                idx = rows.get(row["chunk_id"])
                if idx is not None:
                    metadata = view.all_docs[idx].metadata
                    position = json.loads(row["source_position"])
                    if (metadata["document_version_id"] == row["document_version"] and metadata["file_hash"] == row["file_hash"]
                            and metadata["parent_id"] == row["parent_id"] and position["page"] == metadata["page"]
                            and metadata["start"] <= position["start"] < position["end"] <= metadata["end"]):
                        proofs.setdefault(row["relation_id"], []).append(row | {"row": idx, "source_position": position})
            edges_by_id = {e["relation_id"]: e for _, edges in paths for e in edges}
            c.execute("SELECT * FROM kg_relation_semantics WHERE snapshot_id=%s AND relation_id IN (" +
                      ",".join(["%s"] * len(ids)) + ")", (sid, *ids))
            for row in c.fetchall():
                edges_by_id[row["relation_id"]].update(polarity=row["polarity"],
                    conditions=json.loads(row["conditions_json"]), exceptions=json.loads(row["exceptions_json"]))
        else:
            edges_by_id = {}
        from graph_governance import assess_policies
        reached = set(seeds) | {node for nodes, _ in paths for node in nodes}
        relevant = [row for row in policy_records if row["policy_entity_id"] in reached]
        assessment = assess_policies(relevant, list(edges_by_id.values()), as_of, context)
    return {"truncated": truncated or len(paths)>=limit or len(executed_seeds)<len(all_seeds),
            'seed_execution':{'total':len(all_seeds),'scheduled':len(seeds),'executed':len(executed_seeds),
              'unexecuted':len(all_seeds)-len(executed_seeds),'unexecuted_ids':[s for s in all_seeds if s not in executed_seeds],
              'reason':'seed_limit' if len(all_seeds)>5 else 'deadline_or_path_limit' if len(executed_seeds)<len(all_seeds) else None},
            'entity_matches':entity_matches,
            "limits":{"max_seeds":5,"max_hops":2,"max_paths":limit,"max_edges_per_node":20}, "policy_assessment": assessment, "snapshot_id": sid, "seeds": seeds, "ambiguous": ambiguous,
            "policy_as_of": as_of, "policy_decisions": decisions,
            "timed_out": perf_counter() >= end,
            "conflicts": [{"classification": "TRUE_CONFLICT", "relation_ids": pair,
                           "evidence": [p for key in pair for p in proofs[key]]}
                          for pair in conflicts_for([edge for key, edge in edges_by_id.items() if key in proofs])],
            "paths": projected_paths(paths,proofs)}
