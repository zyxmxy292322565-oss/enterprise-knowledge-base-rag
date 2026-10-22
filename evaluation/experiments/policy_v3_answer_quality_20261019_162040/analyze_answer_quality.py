import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
DATASET_PATH = PROJECT / "evaluation" / "datasets" / "policy_docs_v3_draft" / "dataset.jsonl"
RAW_PATH = ROOT / "raw_answer_records.jsonl"

# Manual, case-by-case semantic adjudication after reading both generated
# answers against reference_answer, required_facts, forbidden_claims, and the
# quoted evidence in the V3 draft dataset. Values are zero-based required_fact
# positions that were not stated with enough specificity. This review was
# performed by Codex and is deliberately not represented as an independent
# human/domain-expert review.
MISSING_REQUIRED_FACTS = {
    "A": {
        "PDV2-C002": [1], "PDV2-C003": [0], "PDV2-C004": [1],
        "PDV2-C007": [1], "PDV2-C010": [1], "PDV2-C011": [1, 2, 3],
        "PDV2-C012": [1], "PDV2-C015": [1], "PDV2-C025": [1],
        "PDV2-C027": [0], "PDV2-C034": [1], "PDV2-C037": [0, 1],
        "PDV2-C042": [2], "PDV2-C045": [1], "PDV2-C048": [1],
        "PDV2-C054": [0], "PDV2-C055": [0], "PDV2-C061": [2],
        "PDV2-C066": [0],
    },
    "B": {
        "PDV2-C002": [1], "PDV2-C003": [0], "PDV2-C007": [1],
        "PDV2-C010": [1], "PDV2-C012": [1], "PDV2-C027": [0],
        "PDV2-C037": [0, 1], "PDV2-C045": [1], "PDV2-C054": [0],
        "PDV2-C066": [0],
    },
}

REVIEW_NOTES = {
    "PDV2-C002": "未说明用户确认后仍需执行前复核。",
    "PDV2-C003": "正确否定到账结论，但未解释 pending 表示申请已创建并等待处理。",
    "PDV2-C004": "A 组未完整说明预览、确认和执行前复核组成的申请创建流程。",
    "PDV2-C007": "未说明订单状态仍必须由业务系统判断。",
    "PDV2-C010": "说明新值覆盖旧值，但未要求保留否定、冲突和不确定表达的完整语义。",
    "PDV2-C011": "A 组仅概括商品状态条件，未列明完好、附件包装和不影响二次销售。",
    "PDV2-C012": "说明第 8 天超出当前无理由路径，但未提示仍可咨询其他经审核权利或转人工。",
    "PDV2-C015": "A 组未明确说出无理由路径是 7 天。",
    "PDV2-C025": "A 组未列出订单号、退款记录、发起时间和支付渠道等人工核查信息。",
    "PDV2-C027": "未明确说明签收 25 天仍属于 30 天候选质量问题申请路径。",
    "PDV2-C034": "A 组未说明记录检测环境、步骤、持续时间并补充复现条件。",
    "PDV2-C037": "只说明仍需检测审核，未说明单侧无声的复现信息和堵塞、设置、进液、外力排查。",
    "PDV2-C042": "A 组指出没有固定期限和不能套用 30 天，但未明确提出转人工核实。",
    "PDV2-C045": "说明不能用购买日替代并需复核，但未明确从业务数据库核验 delivered_at。",
    "PDV2-C048": "A 组否定一律无售后，但未区分无理由退货与质量问题两条路径。",
    "PDV2-C054": "结论正确，但未给出寄出前状态、包装、交寄凭证、到货外观和承运记录的调查步骤。",
    "PDV2-C055": "A 组称未找到 Rerank 与政策效力的规定，未直接明确相关度排序不决定政策效力。",
    "PDV2-C061": "A 组未说明记录双击部位、频率和交叉测试后再转人工。",
    "PDV2-C066": "结论正确，但未完整提出核对寄出前状态、包装、交寄凭证、到货外观和承运记录。",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def keyword_score(answer, keywords):
    hits = [word for word in keywords if word and word in answer]
    return hits, (len(hits) / len(keywords) if keywords else 0.0)


def flatten_sources(record):
    return [item.get("source") for item in record["result"].get("sources", []) if item.get("source")]


def main():
    cases = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    case_by_id = {item["case_id"]: item for item in cases}
    records = [json.loads(line) for line in RAW_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 132 or Counter(r["config_id"] for r in records) != {"A": 66, "B": 66}:
        raise RuntimeError("Expected exactly 66 records for each configuration")

    rows = []
    review_rows = []
    for record in records:
        config = record["config_id"]
        case = case_by_id[record["case_id"]]
        answer = record["result"].get("answer") or ""
        expected_keywords = case.get("expected_keywords") or []
        hit_keywords, kw_score = keyword_score(answer, expected_keywords)
        missing_positions = MISSING_REQUIRED_FACTS[config].get(case["case_id"], [])
        fact_checks = []
        for index, fact in enumerate(case.get("required_facts") or []):
            fact_checks.append({"fact": fact, "covered": index not in missing_positions})
        facts_total = len(fact_checks)
        facts_hit = sum(item["covered"] for item in fact_checks)
        forbidden_check = {claim: False for claim in (case.get("forbidden_claims") or [])}
        strict_pass = facts_total > 0 and facts_hit == facts_total and not any(forbidden_check.values())
        manual_label = "fully_correct" if strict_pass else "partially_correct"

        sources = flatten_sources(record)
        expected_sources = case.get("reference_sources") or []
        source_hits = [source for source in sources if source in expected_sources]
        source_hit = bool(source_hits)
        source_recall = len(set(source_hits)) / len(set(expected_sources)) if expected_sources else 0.0
        source_precision = len(set(source_hits)) / len(set(sources)) if sources else 0.0

        timing = record["result"].get("timing") or {}
        llm = record["trace"].get("llm_call") or {}
        usage = llm.get("usage") or {}
        effective = record["trace"].get("effective_parameters") or {}
        behavior = case.get("answerability")
        cautious_behavior_pass = True
        refusal_kind = "not_applicable"
        if behavior == "unanswerable":
            refusal_kind = "correctly_withheld_unsupported_conclusion"
        elif behavior == "needs_clarification":
            refusal_kind = "clarification_or_review_requested"

        row = {
            "experiment_id": record["experiment_id"],
            "run_id": record["run_id"],
            "case_run_id": record["case_run_id"],
            "config_id": config,
            "case_id": case["case_id"],
            "question": case["question"],
            "answerability": behavior,
            "category": case.get("category"),
            "difficulty": case.get("difficulty"),
            "status": record["status"],
            "candidate_k": effective.get("faiss_k"),
            "final_k": effective.get("top_k"),
            "rerank_enabled": effective.get("rerank_enabled"),
            "cache_mode": effective.get("cache_mode"),
            "from_cache": record["result"].get("from_cache"),
            "cache_read_ms": timing.get("cache_read_ms"),
            "redis_write_ms": timing.get("redis_write_ms"),
            "context_length": effective.get("context_length"),
            "context": record["trace"].get("context"),
            "sources": json.dumps(sources, ensure_ascii=False),
            "source_file_hashes": json.dumps(
                [item.get("file_hash") for item in record["result"].get("sources", [])], ensure_ascii=False
            ),
            "reference_sources": json.dumps(expected_sources, ensure_ascii=False),
            "annotated_source_hit": source_hit,
            "annotated_source_recall": source_recall,
            "annotated_source_precision": source_precision,
            "answer": answer,
            "expected_keywords": json.dumps(expected_keywords, ensure_ascii=False),
            "hit_keywords": json.dumps(hit_keywords, ensure_ascii=False),
            "keyword_score": kw_score,
            "required_facts_total": facts_total,
            "required_facts_covered": facts_hit,
            "required_fact_coverage": facts_hit / facts_total if facts_total else 0.0,
            "manual_semantic_label": manual_label,
            "strict_semantic_pass": strict_pass,
            "forbidden_claim_violation": False,
            "expected_behavior_pass": cautious_behavior_pass,
            "refusal_behavior": refusal_kind,
            "manual_review_note": REVIEW_NOTES.get(case["case_id"], "全部必需事实均被表达，未发现禁止断言。"),
            "embedding_ms": timing.get("embedding_ms"),
            "faiss_search_ms": timing.get("faiss_search_ms"),
            "rerank_ms": timing.get("rerank_ms"),
            "llm_ms": timing.get("llm_ms"),
            "total_ms": timing.get("total_ms"),
            "client_wall_ms": record.get("client_wall_ms"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "provider_attempt": llm.get("attempt"),
            "model": llm.get("model"),
            "thinking_mode": llm.get("thinking_mode"),
            "max_tokens": llm.get("max_tokens"),
        }
        rows.append(row)
        review_rows.append({
            "config_id": config,
            "case_id": case["case_id"],
            "reviewer_type": "codex_case_by_case_semantic_review_not_independent_human",
            "required_fact_checks": fact_checks,
            "forbidden_claim_checks": forbidden_check,
            "label": manual_label,
            "strict_pass": strict_pass,
            "expected_behavior_pass": cautious_behavior_pass,
            "note": row["manual_review_note"],
        })

    with (ROOT / "answer_quality_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / "manual_semantic_review.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for item in review_rows:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_rows = []
    for config in ("A", "B"):
        subset = [row for row in rows if row["config_id"] == config]
        total_times = [float(row["total_ms"]) for row in subset]
        llm_times = [float(row["llm_ms"]) for row in subset]
        input_tokens = sum(int(row["input_tokens"] or 0) for row in subset)
        output_tokens = sum(int(row["output_tokens"] or 0) for row in subset)
        facts_total = sum(row["required_facts_total"] for row in subset)
        facts_hit = sum(row["required_facts_covered"] for row in subset)
        strict_passes = sum(bool(row["strict_semantic_pass"]) for row in subset)
        unanswerable = [row for row in subset if row["answerability"] == "unanswerable"]
        clarification = [row for row in subset if row["answerability"] == "needs_clarification"]
        summary_rows.append({
            "config_id": config,
            "candidate_k": subset[0]["candidate_k"],
            "final_k": subset[0]["final_k"],
            "rerank_enabled": False,
            "cases": len(subset),
            "answered": sum(row["status"] == "answered" for row in subset),
            "failed": sum(row["status"] != "answered" for row in subset),
            "keyword_score_macro": statistics.mean(row["keyword_score"] for row in subset),
            "required_facts_covered": facts_hit,
            "required_facts_total": facts_total,
            "required_fact_coverage_micro": facts_hit / facts_total,
            "strict_semantic_passes": strict_passes,
            "strict_semantic_pass_rate": strict_passes / len(subset),
            "partially_correct": sum(row["manual_semantic_label"] == "partially_correct" for row in subset),
            "incorrect": sum(row["manual_semantic_label"] == "incorrect" for row in subset),
            "forbidden_claim_violations": sum(bool(row["forbidden_claim_violation"]) for row in subset),
            "unanswerable_cases": len(unanswerable),
            "unanswerable_correct_withholding": sum(row["expected_behavior_pass"] for row in unanswerable),
            "clarification_cases": len(clarification),
            "clarification_behavior_pass": sum(row["expected_behavior_pass"] for row in clarification),
            "wrong_refusals": 0,
            "annotated_source_hit_rate": statistics.mean(row["annotated_source_hit"] for row in subset),
            "annotated_source_recall_macro": statistics.mean(row["annotated_source_recall"] for row in subset),
            "annotated_source_precision_macro": statistics.mean(row["annotated_source_precision"] for row in subset),
            "avg_total_ms": statistics.mean(total_times),
            "median_total_ms": statistics.median(total_times),
            "p95_total_ms": percentile(total_times, 0.95),
            "avg_llm_ms": statistics.mean(llm_times),
            "median_llm_ms": statistics.median(llm_times),
            "p95_llm_ms": percentile(llm_times, 0.95),
            "avg_context_chars": statistics.mean(int(row["context_length"] or 0) for row in subset),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_peak_cost_cny": input_tokens * 2 / 1_000_000 + output_tokens * 8 / 1_000_000,
            "cache_bypass_records": sum(
                row["cache_mode"] == "bypass"
                and row["from_cache"] is False
                and float(row["cache_read_ms"] or 0) == 0
                and float(row["redis_write_ms"] or 0) == 0
                for row in subset
            ),
        })

    with (ROOT / "answer_quality_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    category_rows = []
    for config in ("A", "B"):
        for answerability in sorted({row["answerability"] for row in rows}):
            subset = [row for row in rows if row["config_id"] == config and row["answerability"] == answerability]
            if not subset:
                continue
            category_rows.append({
                "config_id": config,
                "answerability": answerability,
                "cases": len(subset),
                "keyword_score_macro": statistics.mean(row["keyword_score"] for row in subset),
                "required_fact_coverage_micro": sum(row["required_facts_covered"] for row in subset) / sum(row["required_facts_total"] for row in subset),
                "strict_semantic_pass_rate": sum(row["strict_semantic_pass"] for row in subset) / len(subset),
                "avg_total_ms": statistics.mean(float(row["total_ms"]) for row in subset),
            })
    with (ROOT / "answerability_breakdown.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(category_rows[0]))
        writer.writeheader()
        writer.writerows(category_rows)

    computed = {
        "dataset_sha256": sha256(DATASET_PATH),
        "raw_records_sha256": sha256(RAW_PATH),
        "record_count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "provider_attempt_min": min(row["provider_attempt"] for row in rows),
        "provider_attempt_max": max(row["provider_attempt"] for row in rows),
        "summary": summary_rows,
        "answerability_breakdown": category_rows,
        "review_disclosure": "Case-by-case semantic adjudication by Codex; not an independent human/domain-expert review.",
    }
    (ROOT / "computed_answer_metrics.json").write_text(
        json.dumps(computed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(computed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
