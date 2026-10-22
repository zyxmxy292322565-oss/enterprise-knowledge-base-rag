"""Recompute the source-level retrieval metrics for this frozen experiment."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
DATASET_PATH = PROJECT / "evaluation/datasets/policy_docs_v3_draft/dataset.jsonl"


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def mean_present(values):
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


dataset = load_jsonl(DATASET_PATH)
case_by_id = {case["case_id"]: case for case in dataset}
raw_records = load_jsonl(ROOT / "raw_case_records.jsonl")

rows: list[dict] = []
candidate_signatures: dict[str, dict[str, tuple]] = collections.defaultdict(dict)
rank_movements: list[int] = []

for wrapper in raw_records:
    config = wrapper["config"]
    record = wrapper["record"]
    case = case_by_id[record["case_id"]]
    result = record.get("result", {})
    trace = record.get("trace", {})
    effective = trace.get("effective_parameters", {})
    expected = set(case.get("reference_sources") or [])
    final_items = result.get("evidence") or result.get("sources") or []
    final_sources = [item.get("source", "") for item in final_items]

    # Repeated chunks from one relevant file receive gain only once. The labels
    # identify relevant source files, not exact chunks.
    seen: set[str] = set()
    relevance: list[int] = []
    for source in final_sources:
        relevance.append(int(source in expected and source not in seen))
        seen.add(source)

    hit = int(any(relevance))
    source_recall = len(set(final_sources) & expected) / len(expected) if expected else None
    reciprocal_rank = next((1 / (rank + 1) for rank, value in enumerate(relevance) if value), 0.0)
    dcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(relevance))
    ideal_relevant = min(len(expected), len(final_sources))
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_relevant))
    ndcg = dcg / idcg if idcg else None

    candidates = trace.get("faiss_candidates") or []
    candidate_sources = [item.get("source", "") for item in candidates]
    candidate_hit = int(bool(set(candidate_sources) & expected))
    candidate_recall = len(set(candidate_sources) & expected) / len(expected) if expected else None
    candidate_signature = tuple(
        (item.get("faiss_index"), item.get("chunk_id"), round(float(item.get("faiss_distance", 0)), 7), item.get("source"))
        for item in candidates
    )
    candidate_signatures[case["case_id"]][config] = candidate_signature

    candidate_by_index = {item.get("faiss_index"): item for item in candidates}
    ranked = trace.get("ranked_candidates") or []
    ranked_sources = [candidate_by_index.get(item.get("faiss_index"), {}).get("source", "") for item in ranked]
    first_faiss = next((rank + 1 for rank, source in enumerate(candidate_sources) if source in expected), None)
    first_rerank = next((rank + 1 for rank, source in enumerate(ranked_sources) if source in expected), None)
    if config == "C" and first_faiss is not None and first_rerank is not None:
        rank_movements.append(first_faiss - first_rerank)

    timing = result.get("timing") or {}
    rows.append(
        {
            "config": config,
            "experiment_id": record["experiment_id"],
            "run_id": record["run_id"],
            "case_run_id": record["case_run_id"],
            "case_id": record["case_id"],
            "question": case["question"],
            "category": case.get("category"),
            "primary_question_type": case.get("primary_question_type"),
            "question_types": "|".join(case.get("question_types") or []),
            "difficulty": case.get("difficulty"),
            "answerable": case.get("answerable"),
            "answerability": case.get("answerability"),
            "review_status": case.get("review_status"),
            "candidate_k": effective.get("faiss_k"),
            "final_k": effective.get("top_k"),
            "rerank_enabled": effective.get("rerank_enabled"),
            "min_rerank_score": effective.get("min_rerank_score"),
            "cache_mode": effective.get("cache_mode"),
            "cache_hit": result.get("cache_hit"),
            "status": record.get("status"),
            "refused": bool(result.get("refused")),
            "expected_sources": "|".join(case.get("reference_sources") or []),
            "candidate_sources": "|".join(candidate_sources),
            "ranked_sources": "|".join(ranked_sources),
            "final_sources": "|".join(final_sources),
            "evidence_hit": hit,
            "source_recall": source_recall,
            "mrr": reciprocal_rank,
            "ndcg": ndcg,
            "candidate_hit_10": candidate_hit,
            "candidate_source_recall_10": candidate_recall,
            "first_relevant_faiss_rank": first_faiss,
            "first_relevant_rerank_rank": first_rerank,
            "embedding_ms": timing.get("embedding_ms"),
            "faiss_ms": timing.get("faiss_ms"),
            "rerank_ms": timing.get("rerank_ms"),
            "server_total_ms": timing.get("total_ms"),
            "client_wall_ms": wrapper.get("client_wall_ms"),
            "error": "",
        }
    )

with (ROOT / "experiment_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary: list[dict] = []
for config in "ABCD":
    selected = [row for row in rows if row["config"] == config]
    server_times = [float(row["server_total_ms"]) for row in selected]
    client_times = [float(row["client_wall_ms"]) for row in selected]
    strict_unanswerable = [row for row in selected if row["answerability"] == "unanswerable"]
    nonanswerable = [row for row in selected if row["answerable"] is False]
    answerable = [row for row in selected if row["answerable"] is True]
    summary.append(
        {
            "config": config,
            "experiment_id": selected[0]["experiment_id"],
            "run_id": selected[0]["run_id"],
            "case_count": len(selected),
            "success_count": len(selected),
            "failure_count": 0,
            "candidate_k": selected[0]["candidate_k"],
            "final_k": selected[0]["final_k"],
            "rerank_enabled": selected[0]["rerank_enabled"],
            "min_rerank_score": selected[0]["min_rerank_score"],
            "cache_mode": "bypass",
            "evidence_hit_rate": statistics.mean(row["evidence_hit"] for row in selected),
            "macro_source_recall": mean_present(row["source_recall"] for row in selected),
            "mrr": statistics.mean(row["mrr"] for row in selected),
            "ndcg": mean_present(row["ndcg"] for row in selected),
            "candidate_hit_10": statistics.mean(row["candidate_hit_10"] for row in selected),
            "candidate_macro_source_recall_10": mean_present(row["candidate_source_recall_10"] for row in selected),
            "refusal_rate_all": statistics.mean(row["refused"] for row in selected),
            "strict_unanswerable_refusal_rate": statistics.mean(row["refused"] for row in strict_unanswerable),
            "all_nonanswerable_refusal_rate": statistics.mean(row["refused"] for row in nonanswerable),
            "answerable_wrong_refusal_rate": statistics.mean(row["refused"] for row in answerable),
            "server_avg_ms": statistics.mean(server_times),
            "server_median_ms": statistics.median(server_times),
            "server_p95_ms": percentile(server_times, 0.95),
            "client_avg_ms": statistics.mean(client_times),
            "client_median_ms": statistics.median(client_times),
            "client_p95_ms": percentile(client_times, 0.95),
            "avg_embedding_ms": statistics.mean(float(row["embedding_ms"]) for row in selected),
            "avg_faiss_ms": statistics.mean(float(row["faiss_ms"]) for row in selected),
            "avg_rerank_ms": statistics.mean(float(row["rerank_ms"]) for row in selected),
        }
    )

with (ROOT / "experiment_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

question_type_rows: list[dict] = []
question_types = sorted({question_type for case in dataset for question_type in case.get("question_types", [])})
for config in "ABCD":
    for question_type in question_types:
        selected = [
            row for row in rows
            if row["config"] == config and question_type in row["question_types"].split("|")
        ]
        question_type_rows.append(
            {
                "config": config,
                "question_type": question_type,
                "case_count": len(selected),
                "evidence_hit_rate": statistics.mean(row["evidence_hit"] for row in selected),
                "macro_source_recall": mean_present(row["source_recall"] for row in selected),
                "mrr": statistics.mean(row["mrr"] for row in selected),
                "ndcg": mean_present(row["ndcg"] for row in selected),
            }
        )

with (ROOT / "question_type_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(question_type_rows[0]))
    writer.writeheader()
    writer.writerows(question_type_rows)

candidate_identity_count = sum(
    len(by_config) == 4 and len(set(by_config.values())) == 1
    for by_config in candidate_signatures.values()
)
computed = {
    "record_count": len(rows),
    "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
    "summary": summary,
    "candidate_set_identity": {
        "identical_case_count": candidate_identity_count,
        "total_case_count": len(candidate_signatures),
    },
    "rerank_first_relevant_rank_movement": {
        "eligible_case_count": len(rank_movements),
        "improved_case_count": sum(value > 0 for value in rank_movements),
        "unchanged_case_count": sum(value == 0 for value in rank_movements),
        "worsened_case_count": sum(value < 0 for value in rank_movements),
        "mean_positions_gained": statistics.mean(rank_movements),
        "median_positions_gained": statistics.median(rank_movements),
    },
}
(ROOT / "computed_metrics.json").write_text(
    json.dumps(computed, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(computed, ensure_ascii=False, indent=2))
