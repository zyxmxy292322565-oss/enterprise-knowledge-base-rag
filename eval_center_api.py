"""Opt-in Evaluation Center HTTP API backed by the isolated phase-one store."""
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query
import logging
from pymysql import MySQLError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from eval_center_data import Evidence, Question, historical_dataset, map_anchor
from eval_center_metrics import FAILURES, answer_metrics, semantic_failures
from eval_center_runner import capture, load_evaluation_snapshot, start_run
from eval_center_store import SPLITS, audit, case, compare, dataset, now, packed, review_entity, run, validated_body
from knowledge_v2 import KnowledgeError
from knowledge_v2_api import _store
from retrieval_engine import RetrievalConfig

router = APIRouter(prefix="/v2/evaluation", tags=["evaluation-center"])
VERSION = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$"
logger = logging.getLogger(__name__)


def permitted_holdout(token):
    secret = os.getenv("RAG_V2_EVAL_HOLDOUT_TOKEN", "")
    return len(secret) >= 24 and hmac.compare_digest(secret.encode(), (token or "").encode())


def scope(row, token):
    if row["split"] == "holdout" and not permitted_holdout(token):
        raise KnowledgeError("Holdout custodian credential required", 403)


def actor(token):
    if token:
        for name, entry in json.loads(os.getenv("RAG_V2_EVAL_REVIEWERS", "{}")).items():
            if len(entry["token"]) >= 24 and hmac.compare_digest(entry["token"].encode(), token.encode()):
                return name, entry["role"]
        raise KnowledgeError("Invalid reviewer credential", 403)
    name = os.getenv("RAG_V2_EVAL_DEVELOPER")
    if not name:
        raise KnowledgeError("Configure a developer identity or reviewer credential", 403)
    return name, "developer"


def checked(operation):
    try:
        return operation()
    except KnowledgeError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (MySQLError, OSError) as exc:
        logger.exception("Evaluation Center operation failed")
        raise HTTPException(503, "Evaluation operation unavailable") from exc


class DatasetBody(BaseModel):
    dataset_version: str = Field(pattern=VERSION)
    split: Literal["dev", "validation", "holdout"]
    name: str = Field(min_length=1, max_length=255)
    questions: list[Question] = Field(default_factory=list, max_length=500)
    evidence: list[Evidence] = Field(default_factory=list, max_length=4000)
    provenance: dict = Field(default_factory=dict)


class EditBody(BaseModel):
    questions: list[Question] = Field(max_length=500)
    evidence: list[Evidence] = Field(max_length=4000)
    expected_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def serialize(row):
    return row | {"sha256": sha256(packed(row["body"]).encode()).hexdigest()}


def validate_holdout(cursor, questions):
    cursor.execute("SELECT body FROM rag_eval_center_datasets WHERE split IN ('dev','validation')")
    earlier = [q for row in cursor.fetchall() for q in json.loads(row["body"])["questions"]]
    normalized = lambda q: "".join(q["question"].casefold().split())
    exposed = {normalized(q) for q in earlier}
    groups = {q.get("leakage_group_id") for q in earlier}
    identifiers = {q["question_id"] for q in earlier}
    for q in questions:
        if not q.get("leakage_group_id") or not q.get("provenance", {}).get("source_type"):
            raise KnowledgeError("Holdout requires origin and leakage_group_id")
        if q["question_id"] in identifiers or normalized(q) in exposed or q["leakage_group_id"] in groups:
            raise KnowledgeError("Holdout overlaps a development or validation dataset", 409)


def insert_dataset(cursor, key, payload, previous=None):
    body = validated_body(payload["questions"], payload["evidence"], key, payload["split"], previous)
    body.update(name=payload["name"], provenance=payload["provenance"], schema_version="1.0")
    cursor.execute("SELECT 1 FROM rag_eval_center_datasets WHERE dataset_id=%s AND dataset_version=%s", key)
    if cursor.fetchone():
        raise KnowledgeError("Dataset version already exists", 409)
    if payload["split"] == "holdout":
        validate_holdout(cursor, body["questions"])
    cursor.execute("INSERT INTO rag_eval_center_datasets(dataset_id,dataset_version,split,body) VALUES(%s,%s,%s,%s)",
                   (*key, payload["split"], packed(body)))
    return serialize(dataset(cursor, key))


@router.get("/datasets")
def list_datasets(store=Depends(_store), x_eval_holdout: str | None = Header(default=None)):
    def operation():
        with store.transaction() as cursor:
            cursor.execute("SELECT * FROM rag_eval_center_datasets")
            rows = cursor.fetchall()
        output = []
        for row in sorted(rows, key=lambda item: item["created_at"], reverse=True):
            if row["split"] == "holdout" and not permitted_holdout(x_eval_holdout):
                continue
            body = json.loads(row.pop("body"))
            counts = {}
            for q in body["questions"]:
                counts[q["answerability"]] = counts.get(q["answerability"], 0) + 1
            output.append(row | {"name": body["name"], "question_count": len(body["questions"]),
                                 "evidence_count": len(body["evidence"]), "answerability_counts": counts,
                                 "reviewed_questions": sum(q["review_status"] in {
                                     "developer_self_reviewed", "independently_reviewed", "adjudicated"} for q in body["questions"])})
        return output
    return checked(operation)


@router.post("/datasets", status_code=201)
def create_dataset(payload: DatasetBody, store=Depends(_store), x_eval_holdout: str | None = Header(default=None)):
    def operation():
        scope({"split": payload.split}, x_eval_holdout)
        with store.transaction() as cursor:
            return insert_dataset(cursor, (uuid4().hex, payload.dataset_version), payload.model_dump())
    return checked(operation)


@router.post("/datasets/import-v3", status_code=201)
def import_v3(store=Depends(_store)):
    def operation():
        source = Path(__file__).resolve().parent / "evaluation/datasets/policy_docs_v3_draft/dataset.jsonl"
        payload = historical_dataset(source.read_bytes()) | {"name": "Policy V3 历史开发集", "split": "dev"}
        key = ("a431673975874a5686ee4841292019c7", "policy-v3-stage2-dev-v1")
        with store.transaction() as cursor:
            cursor.execute("SELECT 1 FROM rag_eval_center_datasets WHERE dataset_id=%s AND dataset_version=%s", key)
            if cursor.fetchone():
                return serialize(dataset(cursor, key))
            return insert_dataset(cursor, key, payload)
    return checked(operation)


@router.get("/historical")
def historical(store=Depends(_store)):
    def operation():
        root = Path(__file__).resolve().parent / "evaluation" / "experiments"
        sources = (
            ("retrieval", "policy_v3_retrieval_20261019_154322", "raw_case_records.jsonl",
             "006a458999fb0d6d2c572921cfa074f05a7e5c9b7717aea2aa0a939bdfaa40aa"),
            ("answer", "policy_v3_answer_quality_20261019_162040", "raw_answer_records.jsonl",
             "b98d0e46fe3c87cbe714b4348712f0bf77d13b9f79700ebb9d05ed238e78f3e6"),
        )
        output = []
        for mode, folder, name, expected in sources:
            raw = (root / folder / name).read_bytes()
            if sha256(raw).hexdigest() != expected:
                raise KnowledgeError("Historical record hash mismatch", 503)
            groups = {}
            for envelope in map(json.loads, raw.splitlines()):
                record = envelope.get("record", envelope)
                key = record["run_id"]
                group = groups.setdefault(key, {"rows": 0, "dataset_version": record["dataset_version"],
                                               "experiment_id": record["experiment_id"]})
                group["rows"] += 1
            manifest = (root / folder / "run_manifest.json").read_bytes()
            output.append({"provenance": "historical_import_reference", "mode": mode, "source": f"{folder}/{name}",
                           "raw_sha256": expected, "manifest_sha256": sha256(manifest).hexdigest(),
                           "manifest": json.loads(manifest), "runs": groups})
        return output
    return checked(operation)


@router.get("/datasets/{dataset_id}/{version}")
def get_dataset(dataset_id: str, version: str, store=Depends(_store), x_eval_holdout: str | None = Header(default=None)):
    def operation():
        with store.transaction() as cursor:
            row = dataset(cursor, (dataset_id, version))
        scope(row, x_eval_holdout)
        return serialize(row)
    return checked(operation)


@router.post("/datasets/{dataset_id}/{version}/fork", status_code=201)
def fork(dataset_id: str, version: str, new_version: str = Query(pattern=VERSION),
         store=Depends(_store), x_eval_holdout: str | None = Header(default=None)):
    def operation():
        with store.transaction() as cursor:
            original = dataset(cursor, (dataset_id, version), lock=True)
            scope(original, x_eval_holdout)
            payload = original["body"] | {"split": original["split"]}
            payload["provenance"] = payload["provenance"] | {"forked_from_version": version}
            return insert_dataset(cursor, (dataset_id, new_version), payload)
    return checked(operation)


@router.put("/datasets/{dataset_id}/{version}")
def edit_dataset(dataset_id: str, version: str, payload: EditBody, store=Depends(_store),
                 x_eval_holdout: str | None = Header(default=None), x_eval_reviewer: str | None = Header(default=None)):
    def operation():
        key = dataset_id, version
        with store.transaction() as cursor:
            row = dataset(cursor, key, lock=True)
            scope(row, x_eval_holdout)
            if row["frozen"] or serialize(row)["sha256"] != payload.expected_sha256:
                raise KnowledgeError("Dataset frozen or changed", 409)
            name, role = actor(x_eval_reviewer)
            before = row["body"]
            row["body"] = before | validated_body([q.model_dump() for q in payload.questions],
                [e.model_dump() for e in payload.evidence], key, row["split"], before)
            if row["split"] == "holdout":
                validate_holdout(cursor, row["body"]["questions"])
            for collection, id_field in (("questions", "question_id"), ("evidence", "evidence_id")):
                old = {x[id_field]: x for x in before[collection]}
                new = {x[id_field]: x for x in row["body"][collection]}
                for identifier in old.keys() | new.keys():
                    if old.get(identifier) != new.get(identifier):
                        audit(cursor, *key, identifier, dict(kind=collection, action="edit",
                            actor=name, role=role, before=old.get(identifier),
                            after=new.get(identifier)))
            cursor.execute("UPDATE rag_eval_center_datasets SET body=%s WHERE dataset_id=%s AND dataset_version=%s",
                           (packed(row["body"]), *key))
            return serialize(row)
    return checked(operation)


@router.post("/datasets/{dataset_id}/{version}/freeze")
def freeze(dataset_id: str, version: str, store=Depends(_store), x_eval_holdout: str | None = Header(default=None)):
    def operation():
        key = dataset_id, version
        with store.transaction() as cursor:
            row = dataset(cursor, key, lock=True)
            scope(row, x_eval_holdout)
            if not row["body"]["questions"]:
                raise KnowledgeError("Empty dataset")
            if row["split"] == "holdout":
                validate_holdout(cursor, row["body"]["questions"])
            cursor.execute("UPDATE rag_eval_center_datasets SET frozen=TRUE WHERE dataset_id=%s AND dataset_version=%s", key)
            return serialize(dataset(cursor, key))
    return checked(operation)


class Review(BaseModel):
    collection: Literal["questions", "evidence"]
    entity_id: str
    action: Literal["approve", "dispute", "return", "adjudicate"]
    note: str = Field(min_length=1, max_length=4000)


@router.post("/datasets/{dataset_id}/{version}/reviews")
def review(dataset_id: str, version: str, payload: Review, store=Depends(_store),
           x_eval_holdout: str | None = Header(default=None), x_eval_reviewer: str | None = Header(default=None)):
    def operation():
        name, role = actor(x_eval_reviewer)
        with store.transaction() as cursor:
            row = dataset(cursor, (dataset_id, version), lock=True)
            scope(row, x_eval_holdout)
            return review_entity(cursor, row, payload.collection, payload.entity_id, name, role, payload.action, payload.note)
    return checked(operation)


def audit_rows(cursor, scope_id, version):
    cursor.execute("SELECT * FROM rag_eval_center_audit WHERE scope_id=%s AND version=%s",
                   (scope_id, version))
    return [row | {"event": json.loads(row["event"])} for row in sorted(cursor.fetchall(), key=lambda r: r["audit_id"])]


@router.get("/datasets/{dataset_id}/{version}/audit")
def dataset_audit(dataset_id: str, version: str, store=Depends(_store), x_eval_holdout: str | None = Header(default=None)):
    def operation():
        with store.transaction() as cursor:
            scope(dataset(cursor, (dataset_id, version)), x_eval_holdout)
            return audit_rows(cursor, dataset_id, version)
    return checked(operation)


@router.get("/datasets/{dataset_id}/{version}/mappings/{snapshot_id}")
def mappings(dataset_id: str, version: str, snapshot_id: str, question_id: str | None = None,
             store=Depends(_store), x_eval_holdout: str | None = Header(default=None)):
    def operation():
        with store.transaction() as cursor:
            row = dataset(cursor, (dataset_id, version))
        scope(row, x_eval_holdout)
        view = store.load_snapshot(snapshot_id)
        output = []
        for evidence in row["body"]["evidence"]:
            if question_id is not None and evidence["question_id"] != question_id:
                continue
            mapped = map_anchor(store, view, evidence)
            ids = {b["child_id"] for b in mapped["bindings"]}
            children = [c for c in view.manifest["children"] if c["child_id"] in ids]
            parent_ids = {c["parent_id"] for c in children}
            output.append({"evidence": evidence, "mapping": mapped, "children": children,
                           "parents": [p for p in view.manifest["parents"] if p["parent_id"] in parent_ids]})
        return {"results": output, "review_queue": [r["evidence"]["evidence_id"] for r in output
                                                    if r["mapping"]["status"] != "mapped"]}
    return checked(operation)


class SemanticRunOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_router: bool = False
    timeout_budget_ms: int = Field(default=5000, ge=50, le=60000)
    token_budget: int = Field(default=4096, ge=128, le=65536)


class RunConfig(RetrievalConfig):
    reliability_enabled: bool = False
    reliability_validator: Literal["rules", "semantic"] = "rules"
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    dataset_version: str
    corpus_snapshot_id: str
    index_snapshot_id: str
    snapshot_mode: Literal["published", "draft_validation"] = "published"
    publish_generation: int = Field(ge=0)
    semantic_options: SemanticRunOptions | None = None
    mode: Literal["retrieval", "answer"]
    candidate_k: int = Field(ge=1, le=200)
    final_k: int = Field(ge=1, le=50)
    rerank_threshold: float = Field(default=0.2, allow_inf_nan=False)
    review_cohort: Literal["developer", "independent_human"] = "developer"

    @model_validator(mode="after")
    def valid_k(self):
        if (self.publish_generation == 0) != (self.snapshot_mode == "draft_validation"):
            raise ValueError("Draft validation requires generation 0; published runs require a publication")
        if self.snapshot_mode == "draft_validation" and self.graph_enabled:
            raise ValueError("Draft validation cannot query the graph")
        if self.semantic_options is not None and self.reliability_validator != "semantic":
            raise ValueError("Semantic run options require semantic validation")
        if self.reliability_validator == "semantic" and not self.reliability_enabled:
            raise ValueError("Semantic validation requires the reliability controller")
        if self.reliability_enabled and self.mode != "answer":
            raise ValueError("Reliability controller requires answer mode")
        if self.final_k > self.candidate_k:
            raise ValueError("final_k exceeds candidate_k")
        return self


def exposed(record):
    return {k: v for k, v in record.items() if k not in {"inputs", "worker_id"}}


def validate_draft_dataset(row, snapshot_id):
    body = row["body"]
    if (row["split"] != "validation" or body.get("provenance", {}).get("snapshot_id") != snapshot_id
            or not body["questions"] or not body["evidence"]
            or any(record["review_status"] != "developer_self_reviewed"
                   for collection in ("questions", "evidence") for record in body[collection])):
        raise KnowledgeError("Draft validation requires a bound, developer-reviewed validation dataset", 409)


@router.post("/runs", status_code=201)
def create_run(payload: RunConfig, store=Depends(_store), x_eval_holdout: str | None = Header(default=None)):
    from api_app import app

    def operation():
        key = payload.dataset_id, payload.dataset_version
        options = payload.model_dump()
        with store.transaction() as cursor:
            row = dataset(cursor, key)
            scope(row, x_eval_holdout)
            if not row["frozen"]:
                raise KnowledgeError("Freeze dataset version before creating a run", 409)
        if payload.snapshot_mode == "draft_validation":
            if payload.review_cohort != "developer":
                raise KnowledgeError("Draft validation requires the developer cohort", 409)
            validate_draft_dataset(row, payload.index_snapshot_id)
        config = capture(store, app, options)
        if payload.snapshot_mode == "draft_validation":
            view = load_evaluation_snapshot(store, config)
            if any(map_anchor(store, view, evidence)["status"] != "mapped" for evidence in row["body"]["evidence"]):
                raise KnowledgeError("Draft validation requires every evidence anchor to map", 409)
        with store.transaction() as cursor:
            row = dataset(cursor, key, lock=True)
            scope(row, x_eval_holdout)
            body = row["body"]
            if not row["frozen"]:
                raise KnowledgeError("Freeze dataset version before creating a run", 409)
            if row["split"] == "holdout" and payload.review_cohort != "independent_human":
                raise KnowledgeError("Holdout requires independent review cohort", 409)
            identifier = uuid4().hex
            config.update(dataset_sha256=serialize(row)["sha256"], split=row["split"])
            cursor.execute("""INSERT INTO rag_eval_center_runs
                (run_id,dataset_id,dataset_version,state,total,config,inputs)
                VALUES(%s,%s,%s,'created',%s,%s,%s)""",
                (identifier, *key, len(body["questions"]), packed(config), packed(body)))
            return exposed(run(cursor, identifier))
    return checked(operation)


@router.get("/runs")
def runs(store=Depends(_store), x_eval_holdout: str | None = Header(default=None)):
    def operation():
        with store.transaction() as cursor:
            cursor.execute("""SELECT run_id,dataset_id,dataset_version,state,total,completed,config,error_message,
                created_at,started_at,finished_at,heartbeat_at FROM rag_eval_center_runs""")
            rows = cursor.fetchall()
        for row in rows:
            row["config"] = json.loads(row["config"])
        return [r for r in sorted(rows, key=lambda item: item["created_at"], reverse=True)
                if r["config"]["split"] != "holdout" or permitted_holdout(x_eval_holdout)]
    return checked(operation)


@router.post("/runs/{run_id}/{action}")
def control(run_id: str, action: Literal["start", "cancel"], store=Depends(_store),
            x_eval_holdout: str | None = Header(default=None)):
    from api_app import app

    def operation():
        with store.transaction() as cursor:
            record = run(cursor, run_id, lock=True)
            scope(record["config"], x_eval_holdout)
            if action == "cancel":
                cursor.execute("""UPDATE rag_eval_center_runs SET state=IF(state='running','cancelling','cancelled'),
                    finished_at=IF(worker_id IS NULL,NOW(6),NULL) WHERE run_id=%s AND state IN ('created','queued','running')""",
                    (run_id,))
                if not cursor.rowcount:
                    raise KnowledgeError("Run cannot be cancelled", 409)
        if action == "start":
            start_run(store, app, run_id)
        with store.transaction() as cursor:
            return exposed(run(cursor, run_id))
    return checked(operation)


def run_cases(cursor, identifier):
    cursor.execute("SELECT body FROM rag_eval_center_cases WHERE run_id=%s ORDER BY question_id", (identifier,))
    return [json.loads(row["body"]) for row in cursor.fetchall()]


@router.get("/runs/{run_id}")
def report(run_id: str, store=Depends(_store), x_eval_holdout: str | None = Header(default=None)):
    def operation():
        with store.transaction() as cursor:
            record = run(cursor, run_id)
            scope(record["config"], x_eval_holdout)
            cases = run_cases(cursor, run_id)
            events = audit_rows(cursor, run_id, "result")
        return {"run": exposed(record), "summary": compare([(record, cases)]), "cases": cases, "audit": events}
    return checked(operation)


@router.get("/compare")
def compare_runs(ids: str, store=Depends(_store), x_eval_holdout: str | None = Header(default=None)):
    def operation():
        identifiers = list(dict.fromkeys(ids.split(",")))
        if not 2 <= len(identifiers) <= 10:
            raise KnowledgeError("Select 2 to 10 different runs")
        with store.transaction() as cursor:
            entries = []
            for identifier in identifiers:
                record = run(cursor, identifier)
                scope(record["config"], x_eval_holdout)
                entries.append((record, run_cases(cursor, identifier)))
        return compare(entries)
    return checked(operation)


class Claim(BaseModel):
    claim_id: str
    text: str
    status: Literal["supported", "contradicted", "unsupported", "uncertain"]
    evidence_child_ids: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    citation_id: str
    claim_id: str
    child_ids: list[str]
    valid: bool | None
    supported: bool | None


class CaseReview(BaseModel):
    note: str = Field(min_length=1)
    failures: list[str] | None = None
    primary_failure: str | None = None
    actual_action: Literal["answer", "clarify", "abstain", "conflict", "mixed", "invalid"] | None = None
    facts: dict[str, Literal["covered", "missing", "incorrect", "uncertain"]] = Field(default_factory=dict)
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    forbidden_violation: bool | None = None
    target_conclusion: bool | None = None
    false_answer: bool | None = None
    action_complete: bool | None = None


@router.post("/runs/{run_id}/cases/{question_id}/review")
def case_review(run_id: str, question_id: str, payload: CaseReview, store=Depends(_store),
                x_eval_holdout: str | None = Header(default=None), x_eval_reviewer: str | None = Header(default=None)):
    def operation():
        name, role = actor(x_eval_reviewer)
        if role not in {"developer", "independent_reviewer", "adjudicator"}:
            raise KnowledgeError("Invalid reviewer role", 403)
        with store.transaction() as cursor:
            record = run(cursor, run_id)
            scope(record["config"], x_eval_holdout)
            before = case(cursor, run_id, question_id, lock=True)
            after = dict(before)
            if payload.failures is not None:
                if set(payload.failures) - set(FAILURES) or (payload.failures and payload.primary_failure not in payload.failures):
                    raise KnowledgeError("Invalid failure labels or primary failure")
                after.update(failures=list(dict.fromkeys(payload.failures)),
                             primary_failure=payload.primary_failure if payload.failures else None)
            if payload.actual_action:
                if record["config"]["mode"] != "answer" or before["state"] == "error":
                    raise KnowledgeError("Semantic review requires a valid answer run")
                content = payload.model_dump()
                question = before["question"]
                if set(content["facts"]) - {f["fact_id"] for f in question["required_facts"]}:
                    raise KnowledgeError("Unknown fact_id")
                claims = {c.claim_id for c in payload.claims}
                if len(claims) != len(payload.claims):
                    raise KnowledgeError("Duplicate claim_id")
                children = {c["child_id"] for c in before["final"]}
                if any(c.claim_id not in claims or (c.valid is True and (not c.child_ids or not set(c.child_ids) <= children))
                       or (c.supported is True and c.valid is not True) for c in payload.citations):
                    raise KnowledgeError("Citation must bind a claim and visible evidence")
                cohort = "developer" if role == "developer" else "independent_human"
                if record["config"]["review_cohort"] != cohort:
                    raise KnowledgeError("Reviewer cohort differs from frozen run configuration", 403)
                eligible = question["review_status"] in ({"developer_self_reviewed"} if cohort == "developer" else
                                                          {"independently_reviewed", "adjudicated"})
                after["answer_review"] = {"cohort": cohort, "actor": name, "reviewed_at": now(),
                                          "labels": content, "metrics": answer_metrics(question, content) if eligible else {},
                                          "scoring_status": "eligible" if eligible else "question_unreviewed_for_cohort"}
                if eligible and payload.failures is None:
                    failures = semantic_failures(question, content)
                    after.update(failures=failures, primary_failure=next(iter(failures), None))
            audit(cursor, run_id, "result", question_id, dict(kind="case_review", actor=name, role=role,
                note=payload.note, before={k: before.get(k) for k in ("failures", "primary_failure", "answer_review")},
                after={k: after.get(k) for k in ("failures", "primary_failure", "answer_review")}))
            cursor.execute("UPDATE rag_eval_center_cases SET body=%s WHERE run_id=%s AND question_id=%s",
                           (packed(after), run_id, question_id))
            return after
    return checked(operation)
