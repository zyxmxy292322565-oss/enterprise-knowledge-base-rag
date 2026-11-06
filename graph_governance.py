"""Draft-only entity overlays and policy applicability metadata."""
import json
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from knowledge_v2 import KnowledgeError

KEY = r"^[a-f0-9]{64}$"


class Merge(BaseModel):
    operation_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    source_id: str = Field(pattern=KEY)
    target_id: str = Field(pattern=KEY)
    reason: str = Field(min_length=1, max_length=1000)


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_id: str = Field(min_length=1, max_length=100)
    policy_version: str = Field(min_length=1, max_length=80)
    policy_entity_id: str = Field(pattern=KEY)
    relation_id: str = Field(pattern=KEY)
    evidence_id: str = Field(pattern=KEY)
    policy_name: str = Field(min_length=1, max_length=160)
    scope: str = Field(min_length=1, max_length=100)
    applicable_products: list[str] = Field(default_factory=list, max_length=20)
    applicable_categories: list[str] = Field(default_factory=list, max_length=20)
    conditions: dict[str, str] = Field(default_factory=dict, max_length=12)
    exceptions: dict[str, str] = Field(default_factory=dict, max_length=12)
    valid_from: date
    valid_until: date | None = None
    policy_status: Literal["ACTIVE", "WITHDRAWN"] = "ACTIVE"
    authority_source: str = Field(min_length=1, max_length=200)
    supersedes_relation_id: str | None = Field(default=None, pattern=KEY)
    supersession_proof_id: str | None = Field(default=None, pattern=KEY)
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def check_dates(self):
        if self.valid_until and self.valid_until < self.valid_from:
            raise ValueError("Invalid policy dates")
        if bool(self.supersedes_relation_id) != bool(self.supersession_proof_id):
            raise ValueError("Supersession needs explicit reviewed proof")
        return self


def draft(cursor, store, sid):
    from knowledge_graph import mutable
    job = mutable(cursor, store, sid)
    if not job or job["task_status"] != "DRAFT":
        raise KnowledgeError("Governance requires an editable draft graph", 409)


def active_merges(c, sid, as_of=None):
    if as_of:
        moment=datetime.fromisoformat(str(as_of).replace('Z','+00:00'))
        if moment.tzinfo:moment=moment.astimezone(timezone.utc).replace(tzinfo=None)
        c.execute('''SELECT source_id,target_id FROM kg_entity_merges WHERE snapshot_id=%s
            AND created_at<=%s AND (undone_at IS NULL OR undone_at>%s) LIMIT 1001''',(sid,moment,moment))
        rows=c.fetchall()
        if len(rows)>1000:raise KnowledgeError('Merge overlay budget exceeded',409)
        return {r['source_id']:r['target_id'] for r in rows}
    c.execute("""SELECT source_id,target_id FROM kg_entity_merges
                 WHERE snapshot_id=%s AND undone_at IS NULL LIMIT 1001""", (sid,))
    rows = c.fetchall()
    if len(rows) > 1000:
        raise KnowledgeError("Merge overlay budget exceeded", 409)
    return {r["source_id"]: r["target_id"] for r in rows}


def merge(store, sid, item, actor):
    with store.transaction() as c:
        draft(c, store, sid)
        c.execute("SELECT * FROM kg_entity_merges WHERE snapshot_id=%s AND operation_id=%s", (sid, item.operation_id))
        prior = c.fetchone()
        if prior:
            if (prior["source_id"], prior["target_id"], prior["actor"], prior["reason"]) != (
                    item.source_id, item.target_id, actor, item.reason):
                raise KnowledgeError("Merge key already used with other input", 409)
            return prior
        c.execute("""SELECT entity_type,description,status FROM kg_entities
            WHERE snapshot_id=%s AND entity_id IN (%s,%s)""", (sid, item.source_id, item.target_id))
        rows = c.fetchall()
        if (len(rows) != 2 or any(row["status"] != "APPROVED" for row in rows) or
                (rows[0]["entity_type"], rows[0]["description"]) != (rows[1]["entity_type"], rows[1]["description"])):
            raise KnowledgeError("Merge needs two approved entities of the same type and scope", 409)
        current = active_merges(c, sid)
        if (item.source_id in current or item.target_id in current or item.source_id in current.values()
                or sum(key == item.target_id for key in current.values()) >= 8):
            raise KnowledgeError("Merge chain, duplicate or group limit", 409)
        c.execute("""INSERT INTO kg_entity_merges(snapshot_id,operation_id,source_id,target_id,actor,reason)
                     VALUES(%s,%s,%s,%s,%s,%s)""", (sid,item.operation_id,item.source_id,item.target_id,actor,item.reason))
        c.execute("SELECT * FROM kg_entity_merges WHERE snapshot_id=%s AND operation_id=%s", (sid,item.operation_id))
        return c.fetchone()


def undo(store, sid, operation, actor, reason):
    with store.transaction() as c:
        draft(c, store, sid)
        c.execute("""UPDATE kg_entity_merges SET undone_at=NOW(6),undone_by=%s,undo_reason=%s
                     WHERE snapshot_id=%s AND operation_id=%s AND undone_at IS NULL""", (actor,reason,sid,operation))
        c.execute("SELECT * FROM kg_entity_merges WHERE snapshot_id=%s AND operation_id=%s", (sid,operation))
        row = c.fetchone()
        if not row:
            raise KnowledgeError("Unknown merge", 404)
        return row


def candidates(store, sid, query, kind=None):
    from knowledge_graph import normalized
    q = normalized(query)
    with store.transaction() as c:
        overlay = active_merges(c,sid)
        c.execute("""SELECT * FROM kg_entities WHERE snapshot_id=%s AND (%s IS NULL OR entity_type=%s)
                     ORDER BY entity_id LIMIT 3001""", (sid,kind,kind))
        nodes = c.fetchall()
        if len(nodes) > 3000:
            raise KnowledgeError("Narrow the entity type to stay within the candidate budget", 422)
        c.execute("SELECT alias,entity_id FROM kg_entity_aliases WHERE snapshot_id=%s LIMIT 6001", (sid,))
        aliases = c.fetchall()
        if len(aliases) > 6000:
            raise KnowledgeError("Alias candidate budget exceeded", 422)
        by_id = {}
        for row in aliases:
            by_id.setdefault(row["entity_id"], []).append(row["alias"])
        found = []
        for node in nodes:
            names = by_id.get(node["entity_id"], [])
            exact = q in names or q == node["normalized_name"]
            score = 1.0 if exact else SequenceMatcher(None, q, node["normalized_name"], autojunk=False).ratio()
            if score >= 0.4:
                found.append(node | {"aliases": names, "score": score, "basis": "exact_or_alias" if exact else "lexical_similarity",
                                     "recommendation": "NEEDS_REVIEW", "original_name": query,
                                     "canonical_id": overlay.get(node["entity_id"], node["entity_id"])})
        found.sort(key=lambda r: (-r["score"],r["entity_id"]))
        for node in found[:12]:
            c.execute("""SELECT e.*,r.relation_type FROM kg_relations r JOIN kg_relation_evidence e
                ON e.snapshot_id=r.snapshot_id AND e.relation_id=r.relation_id WHERE r.snapshot_id=%s
                AND (r.source_entity_id=%s OR r.target_entity_id=%s) ORDER BY e.evidence_id LIMIT 4""",
                      (sid,node["entity_id"],node["entity_id"]))
            node["evidence"] = c.fetchall()
    return {"recommendation": "NEEDS_REVIEW" if found else "CREATE", "candidates": found[:12]}


def policy_rows(c, sid):
    c.execute("""SELECT p.*,e.document_id,e.document_version,e.file_hash,e.chunk_id,e.source_position,e.evidence_text
                 FROM kg_policy_versions p JOIN kg_relation_evidence e ON e.snapshot_id=p.snapshot_id
                 AND e.relation_id=p.relation_id AND e.evidence_id=p.evidence_id WHERE p.snapshot_id=%s LIMIT 1001""", (sid,))
    rows = c.fetchall()
    if len(rows) > 1000:
        raise KnowledgeError("Policy governance budget exceeded", 409)
    c.execute('SELECT relation_id,MIN(effective_at) AS effective_at FROM kg_policy_revocations WHERE snapshot_id=%s GROUP BY relation_id',(sid,))
    revoked={r['relation_id']:r['effective_at'].isoformat() for r in c.fetchall()}
    result=[]
    for row in rows:
        meta=json.loads(row.pop('metadata_json'))
        if row['relation_id'] in revoked:meta['withdrawn_effective_at']=revoked[row['relation_id']]
        result.append(row|{'metadata':meta})
    return result

def revoke_policy(store,sid,relation_id,event_id,effective_at,actor,reason):
    """Append an effective-time event; never overwrite a historical policy row."""
    moment=effective_at
    if moment.tzinfo:moment=moment.astimezone(timezone.utc).replace(tzinfo=None)
    with store.transaction() as c:
        c.execute('SELECT * FROM kg_policy_revocations WHERE event_id=%s',(event_id,));prior=c.fetchone()
        if prior:
            if (prior['snapshot_id'],prior['relation_id'],prior['effective_at'],prior['actor'],prior['reason'])!=(sid,relation_id,moment,actor,reason):raise KnowledgeError('Revocation key conflict',409)
            return prior
        c.execute('SELECT 1 FROM kg_policy_versions WHERE snapshot_id=%s AND relation_id=%s',(sid,relation_id))
        if not c.fetchone():raise KnowledgeError('Unknown governed policy',404)
        c.execute('INSERT INTO kg_policy_revocations(event_id,snapshot_id,relation_id,effective_at,actor,reason) VALUES(%s,%s,%s,%s,%s,%s)',(event_id,sid,relation_id,moment,actor,reason))
        c.execute("INSERT INTO kg_reviews(snapshot_id,entity_kind,object_id,previous_status,status,actor,note) VALUES(%s,'policy',%s,'APPROVED','WITHDRAWN',%s,%s)",
            (sid,relation_id,actor,json.dumps({'event_id':event_id,'effective_at':moment.isoformat(),'reason':reason})))
    return {'event_id':event_id,'effective_at':moment.isoformat(),'status':'WITHDRAWN'}

def withdrawn_at(meta, as_of, context=None):
    when=meta.get('withdrawn_effective_at')
    if not when:return meta['policy_status']!='ACTIVE'
    target=(context or {}).get('at_time') or (str(as_of)[:10]+'T23:59:59.999999')
    def moment(value):
        dt=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt
    return moment(target)>=moment(when)


def save_policy(store, sid, item, actor):
    with store.transaction() as c:
        draft(c, store, sid)
        c.execute("""SELECT 1 FROM kg_relation_evidence e JOIN kg_relations r ON r.snapshot_id=e.snapshot_id
             AND r.relation_id=e.relation_id JOIN kg_entities p ON p.snapshot_id=r.snapshot_id AND p.entity_id=%s
             WHERE e.snapshot_id=%s AND e.relation_id=%s AND e.evidence_id=%s AND r.status='APPROVED'
             AND p.status='APPROVED' AND p.entity_type IN ('Policy','PolicyRule')
             AND p.entity_id IN (r.source_entity_id,r.target_entity_id)""",
                  (item.policy_entity_id,sid,item.relation_id,item.evidence_id))
        if not c.fetchone():
            raise KnowledgeError("Policy requires an approved policy endpoint and bound evidence", 422)
        existing = {row["relation_id"]: row for row in policy_rows(c,sid)}
        if item.relation_id in existing:
            raise KnowledgeError("Policy record already exists in this draft", 409)
        if item.supersedes_relation_id:
            old = existing.get(item.supersedes_relation_id)
            if not old or old["policy_entity_id"] == item.policy_entity_id:
                raise KnowledgeError("Superseded policy needs a distinct registered version", 422)
            old_meta = old["metadata"]
            if (old["policy_id"] != item.policy_id or old["policy_version"] == item.policy_version
                    or old_meta["valid_from"] >= item.valid_from.isoformat()
                    or any(old_meta[key] != item.model_dump(mode="json")[key] for key in (
                           "scope","applicable_products","applicable_categories","conditions","exceptions"))):
                raise KnowledgeError("Supersession cannot change scope, conditions or effective chronology", 422)
            c.execute("""SELECT 1 FROM kg_relations r JOIN kg_relation_evidence e ON e.snapshot_id=r.snapshot_id
                AND e.relation_id=r.relation_id WHERE r.snapshot_id=%s AND r.source_entity_id=%s
                AND r.target_entity_id=%s AND r.relation_type='SUPERSEDES' AND r.status='APPROVED'
                AND e.evidence_id=%s""", (sid,item.policy_entity_id,old["policy_entity_id"],item.supersession_proof_id))
            if not c.fetchone():
                raise KnowledgeError("Approved SUPERSEDES evidence is required", 422)
        meta = item.model_dump(mode="json") | {"review_status":"APPROVED", "reviewed_by":actor}
        c.execute("INSERT INTO kg_policy_versions VALUES(%s,%s,%s,%s,%s,%s,%s)", (sid,item.relation_id,item.policy_id,
                  item.policy_version,item.policy_entity_id,item.evidence_id,json.dumps(meta,ensure_ascii=False)))
        c.execute("INSERT INTO kg_reviews(snapshot_id,entity_kind,object_id,previous_status,status,actor,note) VALUES(%s,'policy',%s,'UNREVIEWED','APPROVED',%s,%s)", (sid,item.relation_id,actor,item.reason))
    return meta


def policy_decisions(rows, as_of, context=None):
    preview = context is None
    context = context or {}
    selected, replacement_active = {}, set()
    for row in rows:
        m = row["metadata"]
        date_ok = m["valid_from"] <= as_of and (not m["valid_until"] or as_of <= m["valid_until"])
        scope_ok = ((not m["applicable_products"] or context.get("product") in m["applicable_products"])
                    and (not m["applicable_categories"] or context.get("category") in m["applicable_categories"])
                    and (preview or context.get("scope") == m["scope"]))
        slots = context.get("slots", {})
        conditions = (all(slots.get(k)==v for k,v in m["conditions"].items())
                      and all(k in slots and slots[k]!=v for k,v in m["exceptions"].items()))
        selected[row["relation_id"]] = ('UNRESOLVED' if m['policy_status']!='ACTIVE' and not m.get('withdrawn_effective_at') else
           "WITHDRAWN" if withdrawn_at(m,as_of,context) else
           "TIME_DIFFERENCE" if not date_ok else "SCOPE_DIFFERENCE" if not scope_ok else
           "CONDITION_DIFFERENCE" if not conditions else "APPLICABLE")
        if scope_ok and conditions and date_ok and not withdrawn_at(m,as_of,context):
            replacement_active.add(row["relation_id"])
    for row in sorted(rows,key=lambda r:r["metadata"]["valid_from"], reverse=True):
        m = row["metadata"]
        old = m.get("supersedes_relation_id")
        if old in selected and row["relation_id"] in replacement_active:
            selected[old] = "SUPERSEDED"
    return selected

def assess_policies(rows, edges, as_of, context):
    """Decide applicability from reviewed structured conditions, never lexical similarity."""
    decisions = policy_decisions(rows, as_of, context)
    router_context = context.get('slots', {}) | {key: context[key] for key in
        ('product', 'category', 'scope', 'as_of') if context.get(key) is not None}
    missing = {k for k in context.get('required_conditions',[]) if not router_context.get(k)}
    for row in rows:
        m = row["metadata"]
        if withdrawn_at(m,as_of,context) or not (m["valid_from"] <= as_of and (not m["valid_until"] or as_of <= m["valid_until"])):
            continue
        if context.get("scope") and context["scope"] != m["scope"]:
            continue
        if m["applicable_products"] and context.get("product") and context["product"] not in m["applicable_products"]:
            continue
        if m["applicable_categories"] and context.get("category") and context["category"] not in m["applicable_categories"]:
            continue
        slots=context.get("slots",{})
        if any(k in slots and slots[k]!=v for k,v in m["conditions"].items()) or any(slots.get(k)==v for k,v in m["exceptions"].items()):
            continue
        if not context.get("scope"): missing.add("scope")
        if m["applicable_products"] and not context.get("product"): missing.add("product")
        if m["applicable_categories"] and not context.get("category"): missing.add("category")
        missing.update((set(m["conditions"]) | set(m["exceptions"])) - set(context.get("slots", {})))
    # Explicit parentheses keep missing-condition keys independent of set precedence.
    missing = {key for key in missing if key in {"scope","product","category"} or key not in context.get("slots",{})}
    selected = [r for r in rows if decisions[r["relation_id"]] == "APPLICABLE"]
    edge_map = {e["relation_id"]:e for e in edges}
    conflicts = []
    for index,left in enumerate(selected):
        a = edge_map.get(left["relation_id"])
        if not a or a["relation_type"] not in {"ALLOWS","EXCLUDES"}:continue
        for right in selected[index+1:]:
            b = edge_map.get(right["relation_id"])
            if not b or a["target_entity_id"] != b["target_entity_id"] or b["relation_type"] not in {"ALLOWS","EXCLUDES"}:continue
            neg_a = (a["relation_type"]=="EXCLUDES") != (a.get("polarity")=="negated")
            neg_b = (b["relation_type"]=="EXCLUDES") != (b.get("polarity")=="negated")
            if neg_a != neg_b:
                kind = "UNRESOLVED" if left["policy_id"]==right["policy_id"] and left["policy_version"]!=right["policy_version"] else "TRUE_CONFLICT"
                conflicts.append({"classification":kind,"relation_ids":[left["relation_id"],right["relation_id"]]})
    return {"policy_count":len(rows), "action":"CLARIFY" if missing else "CONFLICT" if conflicts else "ANSWER" if selected else "ABSTAIN",
            "missing_slots":sorted(missing),"decisions":decisions,"conflicts":conflicts}
