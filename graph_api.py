import hmac
import json
import os
from datetime import date, datetime
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from knowledge_graph import build, confirm_alias, conflicts_for, lookup, review, seal, with_semantics
from knowledge_v2_api import _store, call
from knowledge_v2 import KnowledgeError
from graph_governance import Merge, Policy, candidates, merge, policy_decisions, policy_rows, save_policy, undo

router = APIRouter(prefix="/v3/graph", tags=["knowledge-graph"])


def graph_store(store=Depends(_store)):
    if os.getenv("RAG_V3_GRAPH_ENABLED") != "1":
        raise HTTPException(404, "Graph disabled")
    return store


@router.get("/overview")
def overview(snapshot_id: str | None = None, store=Depends(graph_store)):
    def read():
        head = store.head()
        sid = snapshot_id or head["snapshot_id"]
        with store.transaction() as c:
            c.execute("SELECT * FROM kg_builds ORDER BY started_at DESC LIMIT 100")
            jobs = c.fetchall()
            c.execute("SELECT * FROM kg_entities WHERE snapshot_id=%s ORDER BY entity_id LIMIT 200", (sid,))
            entities = c.fetchall()
            c.execute("SELECT * FROM kg_entity_aliases WHERE snapshot_id=%s ORDER BY alias LIMIT 1000", (sid,))
            aliases = c.fetchall()
            c.execute("SELECT * FROM kg_relations WHERE snapshot_id=%s ORDER BY relation_id LIMIT 200", (sid,))
            relations = c.fetchall()
            with_semantics(c, sid, relations)
            c.execute("SELECT * FROM kg_extraction_runs WHERE snapshot_id=%s", (sid,))
            extraction = c.fetchone()
            c.execute("SELECT child_id,status,attempts,error_type FROM kg_extraction_chunks WHERE snapshot_id=%s ORDER BY child_id LIMIT 500", (sid,))
            chunks = c.fetchall()
            c.execute("SELECT * FROM kg_reviews WHERE snapshot_id=%s ORDER BY review_id DESC LIMIT 100", (sid,))
            reviews = c.fetchall()
            c.execute("SELECT * FROM kg_relation_evidence WHERE snapshot_id=%s ORDER BY relation_id LIMIT 400", (sid,))
            evidence = c.fetchall()
            c.execute("SELECT status,COUNT(*) AS count FROM kg_relations WHERE snapshot_id=%s GROUP BY status", (sid,))
            counts = c.fetchall()
        return {"active": head, "snapshot_id": sid, "jobs": jobs, "entities": entities, "aliases": aliases,
                "relations": relations, "evidence": evidence, "review_counts": counts,
                "extraction": extraction, "chunks": chunks, "reviews": reviews, "conflicts": conflicts_for(relations)}
    return call(read)


@router.post("/{sid}/build", status_code=201)
def create(sid: str, mode: Literal["rules", "llm"] = "rules", batch_size: int = Query(100, ge=1, le=100),
           store=Depends(graph_store)):
    return call(build, store, call(store.load_snapshot, sid, strict_sparse=True), mode, batch_size)


class Review(BaseModel):
    kind: Literal["entity", "relation"]
    id: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: Literal["APPROVED", "REJECTED", "NEEDS_REVIEW"]
    note: str = Field(min_length=1, max_length=1000)


@router.post("/{sid}/review")
def review_record(sid: str, payload: Review, store=Depends(graph_store),
                  x_kg_reviewer: str | None = Header(default=None)):
    actor = reviewer_identity(x_kg_reviewer)
    return call(review, store, sid, payload.kind, payload.id, payload.status, actor, payload.note)


def reviewer_identity(credential):
    token = os.getenv("RAG_V3_GRAPH_REVIEW_TOKEN", "")
    actor = os.getenv("RAG_V3_GRAPH_REVIEW_ACTOR", "")
    if len(token) < 24 or not actor or not hmac.compare_digest(token.encode(), (credential or "").encode()):
        raise HTTPException(403, "Graph reviewer credential required")
    return actor


class AliasReview(BaseModel):
    entity_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    alias: str = Field(min_length=1, max_length=150)
    note: str = Field(min_length=1, max_length=1000)


@router.post("/{sid}/aliases")
def approve_alias(sid: str, payload: AliasReview, store=Depends(graph_store),
                  x_kg_reviewer: str | None = Header(default=None)):
    return call(confirm_alias, store, sid, payload.entity_id, payload.alias, payload.evidence_id,
                reviewer_identity(x_kg_reviewer), payload.note)


@router.post("/{sid}/seal")
def seal_graph(sid: str, store=Depends(graph_store)):
    return call(seal, store, sid)


@router.get("/lookup")
def query_graph(question: str = Query(min_length=1, max_length=2000), hops: int = Query(2, ge=1, le=2),
                as_of: date | None = None, scope: str | None = None, product: str | None = None,
                category: str | None = None, at_time: datetime | None = None, entity_type: str | None = None, snapshot_id: str | None = Query(default=None, pattern=r"^[a-f0-9]{32}$"), store=Depends(graph_store)):
    from knowledge_v2_api import published_view
    return call(lookup, store, call(published_view, store, snapshot_id), question, hops,
                policy_context={"as_of": as_of.isoformat() if as_of else None, "scope": scope,
                                "product":product,"category":category,'at_time':at_time.isoformat() if at_time else None,'entity_type':entity_type})

class PolicyRevocation(BaseModel):
    event_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    effective_at: datetime
    reason: str = Field(min_length=1,max_length=1000)

@router.post('/{sid}/policies/{relation_id}/revoke')
def revoke(sid: str,relation_id: str,payload: PolicyRevocation,store=Depends(graph_store),x_kg_reviewer: str | None=Header(default=None)):
    from graph_governance import revoke_policy
    return call(revoke_policy,store,sid,relation_id,payload.event_id,payload.effective_at,reviewer_identity(x_kg_reviewer),payload.reason)


@router.get("/{sid}/candidates")
def match(sid: str, query: str = Query(min_length=1, max_length=150), kind: str | None = None,
          store=Depends(graph_store)):
    return call(candidates, store, sid, query, kind)


@router.post("/{sid}/merges")
def apply_merge(sid: str, payload: Merge, store=Depends(graph_store), x_kg_reviewer: str | None = Header(default=None)):
    return call(merge, store, sid, payload, reviewer_identity(x_kg_reviewer))


class Undo(BaseModel):
    reason: str = Field(min_length=1, max_length=1000)


@router.post("/{sid}/merges/{operation}/undo")
def undo_merge(sid: str, operation: str, payload: Undo, store=Depends(graph_store),
               x_kg_reviewer: str | None = Header(default=None)):
    return call(undo, store, sid, operation, reviewer_identity(x_kg_reviewer), payload.reason)


@router.get("/{sid}/governance")
def governance(sid: str, as_of: date | None = None, store=Depends(graph_store)):
    def read():
        with store.transaction() as c:
            c.execute("SELECT * FROM kg_entity_merges WHERE snapshot_id=%s ORDER BY created_at DESC LIMIT 200", (sid,))
            merges = c.fetchall()
            rows = policy_rows(c,sid)
            c.execute("SELECT UTC_DATE() AS today")
            day = as_of or c.fetchone()["today"]
        return {"merges": merges,"policies": rows,"as_of":day.isoformat(),
                "policy_decisions":policy_decisions(rows,day.isoformat())}
    return call(read)


@router.post("/{sid}/policies")
def register_policy(sid: str, payload: Policy, store=Depends(graph_store), x_kg_reviewer: str | None = Header(default=None)):
    return call(save_policy,store,sid,payload,reviewer_identity(x_kg_reviewer))


class InheritanceRequest(BaseModel):
    source_snapshot_id: str = Field(pattern=r"^[a-f0-9]{32}$")


@router.post("/{sid}/inheritance")
def inherit_records(sid: str, payload: InheritanceRequest, store=Depends(graph_store),
                    x_kg_reviewer: str | None = Header(default=None)):
    from graph_inheritance import inherit
    return call(inherit, store, payload.source_snapshot_id, sid, reviewer_identity(x_kg_reviewer))


class InheritanceDismissal(BaseModel):
    object_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    reason: str = Field(min_length=1, max_length=1000)


@router.post("/{sid}/inheritance/dismiss")
def dismiss_inheritance(sid: str, payload: InheritanceDismissal, store=Depends(graph_store),
                        x_kg_reviewer: str | None = Header(default=None)):
    actor=reviewer_identity(x_kg_reviewer)
    def apply():
        from graph_governance import draft
        with store.transaction() as c:
            draft(c,store,sid)
            c.execute("SELECT status FROM kg_reviews WHERE snapshot_id=%s AND entity_kind='inheritance' AND object_id=%s ORDER BY review_id DESC LIMIT 1",(sid,payload.object_id))
            old=c.fetchone()
            if not old:
                raise KnowledgeError("Inheritance event not found",404)
            if old["status"] != "REJECTED":
                c.execute('''SELECT o.*,r.source_snapshot FROM kg_inheritance_objects o JOIN kg_inheritance_runs r ON r.run_id=o.run_id
                    JOIN kg_reviews q ON q.review_id=o.review_id WHERE r.target_snapshot=%s AND q.object_id=%s''',(sid,payload.object_id))
                inherited=c.fetchone()
                if inherited and inherited['status']=='AUTO_REBOUND':
                    detail=json.loads(inherited['details'])
                    kind,key=inherited['kind'],inherited['target_object']
                    if kind in {'entity','relation'}:
                        table,column=('kg_entities','entity_id') if kind=='entity' else ('kg_relations','relation_id')
                        c.execute(f"UPDATE {table} SET status='NEEDS_REVIEW' WHERE snapshot_id=%s AND {column}=%s",(sid,key))
                    elif kind=='alias' and not detail.get('target_existed'):
                        c.execute('DELETE FROM kg_entity_aliases WHERE snapshot_id=%s AND entity_id=%s AND alias=%s',(sid,detail['entity_id'],detail['alias']))
                    elif kind=='policy' and not detail.get('target_existed'):
                        c.execute('DELETE FROM kg_policy_versions WHERE snapshot_id=%s AND relation_id=%s',(sid,key))
                    elif kind=='merge':
                        c.execute('UPDATE kg_entity_merges SET undone_at=UTC_TIMESTAMP(6),undone_by=%s,undo_reason=%s WHERE snapshot_id=%s AND operation_id=%s AND undone_at IS NULL',(actor,payload.reason,sid,key))
                    c.execute("UPDATE kg_inheritance_objects SET status='REJECTED' WHERE run_id=%s AND kind=%s AND source_object=%s",(inherited['run_id'],kind,inherited['source_object']))
                c.execute("INSERT INTO kg_reviews(snapshot_id,entity_kind,object_id,previous_status,status,actor,note) VALUES(%s,'inheritance',%s,%s,'REJECTED',%s,%s)",
                          (sid,payload.object_id,old["status"],actor,payload.reason))
        return {"status":"REJECTED","object_id":payload.object_id}
    return call(apply)
