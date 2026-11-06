"""Transactional storage for versioned datasets and single-worker evaluation runs."""
from datetime import datetime, timezone
import json
from uuid import uuid4

from eval_center_data import Evidence, Question
from eval_center_metrics import FAILURES, TRUSTED, answer_metrics, summarize
from knowledge_v2 import KnowledgeError, json_bytes

SPLITS = {"dev", "validation", "holdout"}
REVIEW_ACTIONS = {
    "approve": {"developer": "developer_self_reviewed", "independent_reviewer": "independently_reviewed"},
    "dispute": {"developer": "disputed", "independent_reviewer": "disputed", "adjudicator": "disputed"},
    "return": {"developer": "proposed_unreviewed", "independent_reviewer": "proposed_unreviewed",
               "adjudicator": "proposed_unreviewed"},
    "adjudicate": {"adjudicator": "adjudicated"},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def packed(obj):
    return json_bytes(obj).decode()


def audit(cursor, scope, version, entity, event):
    cursor.execute("INSERT INTO rag_eval_center_audit(scope_id,version,entity_id,event) VALUES(%s,%s,%s,%s)",
                   (scope, version, entity, packed(event)))


def dataset(cursor, key, lock=False):
    cursor.execute("SELECT * FROM rag_eval_center_datasets WHERE dataset_id=%s AND dataset_version=%s" +
                   (" FOR UPDATE" if lock else ""), key)
    row = cursor.fetchone()
    if not row:
        raise KnowledgeError("Unknown dataset version", 404)
    row["body"] = json.loads(row["body"])
    return row


def run(cursor, run_id, lock=False):
    cursor.execute("SELECT * FROM rag_eval_center_runs WHERE run_id=%s" + (" FOR UPDATE" if lock else ""), (run_id,))
    row = cursor.fetchone()
    if not row:
        raise KnowledgeError("Unknown evaluation run", 404)
    row.update(config=json.loads(row["config"]), inputs=json.loads(row["inputs"]))
    return row


def case(cursor, run_id, question_id, lock=False):
    cursor.execute("SELECT body FROM rag_eval_center_cases WHERE run_id=%s AND question_id=%s" +
                   (" FOR UPDATE" if lock else ""), (run_id, question_id))
    row = cursor.fetchone()
    if not row:
        raise KnowledgeError("Unknown case", 404)
    return json.loads(row["body"])


def validated_body(questions, evidence, key, split, previous=None):
    body = {"questions": [], "evidence": []}
    old = previous or body
    for collection, records, model, id_field, revision in (
        ("questions", questions, Question, "question_id", "question_revision"),
        ("evidence", evidence, Evidence, "evidence_id", "evidence_revision"),
    ):
        historical = {obj[id_field]: obj for obj in old[collection]}
        seen = set()
        for record in records:
            obj = model.model_validate(record).model_dump()
            identifier = obj[id_field]
            if identifier in seen:
                raise KnowledgeError(f"Duplicate {id_field}")
            seen.add(identifier)
            before = historical.get(identifier)
            server = {"dataset_id", "dataset_version", "split", "created_at", "updated_at", "review_status",
                      revision, "gold_evidence_ids", "review_events", "change_history", "chunk_bindings",
                      "mapping_status"}
            fields = {k: v for k, v in obj.items() if k not in server}
            unchanged = before is not None and fields == {k: v for k, v in before.items() if k not in server}
            if split == "holdout" and fields.get("provenance", {}).get("source_type", "").startswith("historical"):
                raise KnowledgeError("Historical development questions cannot populate holdout")
            fields.update(dataset_id=key[0], dataset_version=key[1], split=split,
                          created_at=before["created_at"] if before else now(),
                          updated_at=before["updated_at"] if unchanged else now())
            fields[revision] = before[revision] + int(not unchanged) if before else 1
            fields["review_status"] = before["review_status"] if unchanged else (
                "unreviewed" if collection == "questions" else "proposed_unreviewed")
            body[collection].append(fields)
    question_map = {q["question_id"]: q for q in body["questions"]}
    evidence_map = {}
    for item in body["evidence"]:
        q = question_map.get(item["question_id"])
        if not q or set(item["supports_fact_ids"]) - {f["fact_id"] for f in q["required_facts"]}:
            raise KnowledgeError("Evidence must reference an existing question and facts")
        evidence_map.setdefault(q["question_id"], {})[item["evidence_id"]] = item
    for q in body["questions"]:
        related = evidence_map.get(q["question_id"], {})
        q["gold_evidence_ids"] = list(related)
        for fact in q["required_facts"]:
            for option in fact["evidence_options"]:
                if not option or any(key not in related or fact["fact_id"] not in related[key]["supports_fact_ids"] for key in option):
                    raise KnowledgeError("Fact option references missing or unrelated evidence")
        if any(key not in related for group in q["required_evidence_groups"] for option in group["options"] for key in option):
            raise KnowledgeError("Evidence group references missing evidence")
    return body


def review_entity(cursor, row, collection, identifier, actor, role, action, note):
    key = row["dataset_id"], row["dataset_version"]
    if row["frozen"]:
        raise KnowledgeError("Dataset version frozen", 409)
    record = next((obj for obj in row["body"][collection]
                   if obj["evidence_id" if collection == "evidence" else "question_id"] == identifier), None)
    if record is None:
        raise KnowledgeError("Entity not found", 404)
    status = REVIEW_ACTIONS.get(action, {}).get(role)
    if not status or (action == "adjudicate" and record["review_status"] != "disputed"):
        raise KnowledgeError("Invalid review transition", 409)
    if record["review_status"] == "disputed" and action == "approve":
        raise KnowledgeError("Resolve the dispute by adjudication or return", 409)
    before = dict(record)
    record.update(review_status=status, updated_at=now())
    audit(cursor, *key, identifier, dict(kind=collection, actor=actor, role=role, action=action, note=note,
                                       before=before, after=record))
    cursor.execute("UPDATE rag_eval_center_datasets SET body=%s WHERE dataset_id=%s AND dataset_version=%s",
                   (packed(row["body"]), *key))
    return record


def compare(entries):
    from statistics import median, quantiles

    output = []
    valid_sets = {}
    for record, cases in entries:
        cohort = record["config"]["review_cohort"]
        reviews = [c["answer_review"] for c in cases if c.get("answer_review", {}).get("cohort") == cohort]
        metrics = [c.get("metrics", {}) if record["config"]["mode"] == "retrieval" else
                   (c.get("answer_review", {}).get("metrics", {}) if c.get("answer_review", {}).get("cohort") == cohort else {})
                   for c in cases]
        group_metrics = {field: {
            label: summarize([m for m, c in zip(metrics, cases) if c["question"][field] == label])
            for label in sorted({c["question"][field] for c in cases})
        } for field in ("question_type", "answerability")}
        for c, metrics_row in zip(cases, metrics):
            if c.get("state") != "error" and not c.get("degradation"):
                valid_sets.setdefault(record["run_id"], {})[c["question"]["question_id"]] = metrics_row
        engineering = {}
        for key in ("total_ms", "retrieval_ms", "rerank_ms", "llm_ms", "prompt_tokens", "completion_tokens",
                    "total_tokens", "model_calls", "cost", "hyde_ms", "repair_count", "retrieval_rounds"):
            values = [c["engineering"].get(key) for c in cases if c.get("engineering", {}).get(key) is not None]
            engineering[key] = {"available": len(values), "status": "available" if values else "unavailable",
                                "mean": sum(values) / len(values) if values else None,
                                "p50": median(values) if values else None,
                                "p95": (quantiles(values, n=100, method="inclusive")[94] if len(values) > 1
                                        else next(iter(values), None))}
        engineering["cache_hit_rate"] = {"value": None, "status": "unavailable", "reason": "bypass"}
        reliability = [c["reliability"] for c in cases if "reliability" in c]
        output.append({"run_id": record["run_id"], "mode": record["config"]["mode"], "cohort": cohort,
                       "reliability": {"cases": len(reliability),
                            "mean_rounds": sum(x["rounds"] for x in reliability) / len(reliability),
                            "timeouts": sum(x["timed_out"] for x in reliability),
                            "metrics": summarize([x["metrics"] for x in reliability]),
                            "actions": {a: sum(x["action"] == a for x in reliability)
                                        for a in ("ANSWER", "CLARIFY", "ABSTAIN", "CONFLICT")}} if reliability else None,
                       "total": record["total"], "completed": record["completed"], "metrics": summarize(metrics),
                       "semantic_reviews": len(reviews), "by_type": group_metrics, "engineering": engineering,
                       "technical_errors": sum(c.get("state") == "error" for c in cases),
                       "degraded_cases": sum(bool(c.get("degradation")) for c in cases),
                       "failures": {key: sum(key in c.get("failures", []) for c in cases) for key in FAILURES}})
    if len(entries) < 2:
        return {"runs": output, "paired": None}
    configs = [r["config"] for r, _ in entries]
    shared_fields = ("dataset_id", "dataset_version", "dataset_sha256", "index_snapshot_id", "publish_generation",
                     "corpus_snapshot_id", "mode", "candidate_k", "final_k", "review_cohort", "reliability_enabled",
                     "reliability_validator", "snapshot_mode", "snapshot_manifest_sha256", "semantic_options",
                     "working_code_sha256", "embedding_model_version", "rerank_model_version", "prompt_version")
    compatible = all(all(config.get(k) == configs[0].get(k) for k in shared_fields) for config in configs[1:])
    if not compatible:
        return {"runs": output, "paired": None, "pairing_status": "incompatible frozen inputs or K"}
    samples = [valid_sets.get(r["run_id"], {}) for r, _ in entries]
    common_questions = set.intersection(*(set(s) for s in samples))
    keys = {k for s in samples for q in common_questions for k in s[q]}
    paired = {key: sorted(q for q in common_questions if all(s[q].get(key, {}).get("value") is not None for s in samples))
              for key in keys}
    return {"runs": output, "paired": {"completed_intersection": len(common_questions),
            "metrics": {key: {"count": len(ids), "run_values": {
                record["run_id"]: summarize([{key: s[q][key]} for q in ids]).get(key)
                for (record, _), s in zip(entries, samples)}} for key, ids in paired.items()}}}
