"""S1-A metrics with explicit eligibility and denominators."""
from math import log2
from statistics import mean

TRUSTED = {"developer_self_reviewed", "independently_reviewed", "adjudicated"}
METRICS_VERSION = "rag-v3-metrics-1.2-draft"
FAILURES = ("RETRIEVAL_MISS", "RERANK_LOSS", "EVIDENCE_INSUFFICIENT", "ANSWER_UNSUPPORTED",
            "CITATION_INVALID", "CITATION_UNSUPPORTED", "FALSE_ANSWER", "FALSE_REFUSAL",
            "MISSED_CLARIFICATION", "MISSED_CONFLICT")


def ratio(n=0, d=0, reason="empty denominator"):
    return {"numerator": n, "denominator": d, "value": n / d if d else None,
            "na_reason": None if d else reason}


def ranking(candidate, final, gold, candidate_k, final_k, reason):
    candidate, final = candidate[:candidate_k], final[:final_k]
    hits = len(set(candidate) & gold)
    seen = set()
    ranks = []
    for rank, key in enumerate(final, 1):
        if key in gold and key not in seen:
            ranks.append(rank)
        seen.add(key)
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(final_k, len(gold)) + 1))
    return {
        "hit": ratio(int(bool(hits)), int(bool(gold)), reason),
        "recall": ratio(hits, len(gold), reason),
        "precision": ratio(len(ranks), final_k if gold else 0, reason),
        "final_recall": ratio(len(ranks), len(gold), reason),
        "mrr": ratio(1, ranks[0]) if ranks else ratio(0, int(bool(gold)), reason),
        "ndcg": ratio(sum(1 / log2(rank + 1) for rank in ranks), ideal, reason),
        "repeat_rate": ratio(len(final) - len(seen), len(final)),
    }


def covered_span(start, end, intervals):
    for left, right in sorted(intervals):
        if left > start:
            break
        start = max(start, right)
        if start >= end:
            return True
    return False


def retrieval_metrics(question, evidence, mappings, candidate, final, config):
    eligible = question["answerability"] == "answerable" and question["review_status"] in TRUSTED
    options = [fact["evidence_options"] for fact in question["required_facts"] if fact["purpose"] == "answer"]
    groups = [group["options"] for group in question.get("required_evidence_groups", [])]
    ids = {key for unit in options + groups for option in unit for key in option}
    positives = [e for e in evidence if e["evidence_id"] in ids and e["role"] == "answer_support"]
    reviewed = bool(options) and all(options + groups) and bool(ids) and (
        {e["evidence_id"] for e in positives if e["review_status"] in TRUSTED} == ids)
    files = {(e["source_path"], e["source_content_hash"]) for e in positives} if eligible and reviewed else set()
    mapped = eligible and reviewed and all(mappings.get(key, {}).get("status") == "mapped" for key in ids)
    gold_chunks = {row["child_id"] for key in ids for row in mappings[key]["bindings"]} if mapped else set()
    reason = "ineligible, unreviewed or incomplete mapping"
    result = {}
    for prefix, field, gold in (("file", lambda c: (c["source"], c["source_content_hash"]), files),
                                ("chunk", lambda c: c["child_id"], gold_chunks)):
        values = ranking([field(c) for c in candidate], [field(c) for c in final], gold,
                         config["candidate_k"], config["final_k"], reason)
        result.update({f"{prefix}_{key}": value for key, value in values.items()})
    by_child = {row["child_id"]: row for row in final[:config["final_k"]]}
    satisfied = set()
    if mapped:
        for key in ids:
            binding = mappings[key]
            spans = [(row["clean_start"], min(row["clean_end"], by_child[row["child_id"]]["visible_end"]))
                     for row in binding["bindings"] if row["child_id"] in by_child]
            if covered_span(binding["clean_start"], binding["clean_end"], spans):
                satisfied.add(key)
    for name, units in (("required_fact_evidence_coverage", options), ("evidence_group_coverage", groups)):
        result[name] = ratio(sum(any(option and set(option) <= satisfied for option in unit) for unit in units),
                             len(units) if mapped else 0, reason)
    # Gold-anchor span coverage with verified Parent/Child/version/snapshot
    # identity. This does not claim that an entire Parent was shown to the LLM.
    parent_ready = mapped and all(mappings[key].get("document_version_id")
        and mappings[key].get("index_snapshot_id")
        and all(row.get("parent_id") for row in mappings[key]["bindings"]) for key in ids)
    parent_supported = set()
    if parent_ready:
        for key in ids:
            binding = mappings[key]
            spans = []
            for row in binding["bindings"]:
                actual = by_child.get(row["child_id"], {})
                if (actual.get("parent_id") == row["parent_id"]
                        and actual.get("document_version_id") == binding["document_version_id"]
                        and actual.get("index_snapshot_id") == binding["index_snapshot_id"]):
                    spans.append((row["clean_start"], min(row["clean_end"], actual["visible_end"])))
            if covered_span(binding["clean_start"], binding["clean_end"], spans):
                parent_supported.add(key)
    result["parent_child_evidence_coverage"] = ratio(len(parent_supported), len(ids) if parent_ready else 0,
        "unreviewed Gold or missing Parent/Child/version/snapshot mapping")
    return result


def answer_metrics(question, review):
    """Only submitted semantic labels contribute. None represents unknown."""
    facts = review["facts"]
    claims = review["claims"]
    citations = review["citations"]
    expected, action = question["expected_action"], review["actual_action"]
    def flag(value):
        return ratio(int(value is True), int(value is not None))
    result = {}
    for purpose in ("answer", "rationale"):
        values = [facts.get(fact["fact_id"]) for fact in question["required_facts"] if fact["purpose"] == purpose]
        decided = [v for v in values if v in {"covered", "missing", "incorrect"}]
        result[f"required_fact_coverage_{purpose}"] = ratio(decided.count("covered"), len(decided))
    decided = [c["status"] for c in claims if c["status"] != "uncertain"]
    result["unsupported_claim_rate"] = ratio(sum(v in {"unsupported", "contradicted"} for v in decided), len(decided))
    result["faithfulness"] = ratio(decided.count("supported"), len(decided), "no adjudicated claim labels")
    result["reviewed_action_agreement"] = flag(action == expected if review["action_complete"] is not None else None)
    result["forbidden_claim_rate"] = flag(review["forbidden_violation"] if question["forbidden_claims"] else None)
    result["citation_valid_rate"] = ratio(sum(c["valid"] is True for c in citations),
                                          sum(c["valid"] is not None for c in citations))
    judged = {(c["claim_id"], c["citation_id"]): c for c in citations if c["supported"] is not None or c["valid"] is False}
    result["citation_support_rate"] = ratio(sum(c["valid"] is True and c["supported"] is True for c in judged.values()),
                                            len(judged))
    result["false_answer_rate"] = flag(review["false_answer"] if review["target_conclusion"] else None)
    result["false_refusal_rate"] = flag(action == "abstain" if expected == "answer" else None)
    for key, target in (("correct_clarification_rate", "clarify"), ("conflict_detection_rate", "conflict"),
                        ("correct_abstain_rate", "abstain")):
        result[key] = flag((False if action != target else review["action_complete"]) if expected == target else None)
    return result


def semantic_failures(question, review):
    metrics = answer_metrics(question, review)
    predicates = (
        ("ANSWER_UNSUPPORTED", metrics["unsupported_claim_rate"]["numerator"] > 0),
        ("CITATION_INVALID", any(c["valid"] is False for c in review["citations"])),
        ("CITATION_UNSUPPORTED", any(c["valid"] is True and c["supported"] is False for c in review["citations"])),
        ("FALSE_ANSWER", review["false_answer"] is True),
        ("FALSE_REFUSAL", metrics["false_refusal_rate"]["value"] == 1),
        ("MISSED_CLARIFICATION", metrics["correct_clarification_rate"]["value"] == 0),
        ("MISSED_CONFLICT", metrics["conflict_detection_rate"]["value"] == 0),
    )
    return [key for key, failed in predicates if failed]


def summarize(rows):
    names = {key for row in rows for key in row}
    return {key: {
        "macro": mean(valid) if (valid := [r[key]["value"] for r in rows if key in r and r[key]["value"] is not None]) else None,
        "micro": (ratio(sum(r[key]["numerator"] for r in rows if key in r),
                        sum(r[key]["denominator"] for r in rows if key in r))["value"]
                  if not key.endswith(("_mrr", "_ndcg")) else None),
        "eligible": len(valid), "planned": len(rows),
    } for key in sorted(names)}
