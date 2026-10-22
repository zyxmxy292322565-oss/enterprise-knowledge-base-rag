import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
DATASET_PATH = PROJECT / "evaluation" / "datasets" / "policy_docs_v3_draft" / "dataset.jsonl"
BASE_URL = "http://127.0.0.1:18080"
MAX_CALLS = 132
BUDGET_CNY = 1.50
STOP_THRESHOLD_CNY = 1.48


def request_json(method, path, payload=None, timeout=180):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def main():
    cases = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != 66:
        raise RuntimeError(f"Expected 66 cases, found {len(cases)}")

    probe = json.loads((ROOT / "first_call_probe.json").read_text(encoding="utf-8"))
    experiment_id = probe["experiment_id"]
    run_ids = {"A": probe["run_A_id"]}
    run_ids["B"] = request_json("POST", f"/evaluation/experiments/{experiment_id}/runs")["run_id"]

    raw_path = ROOT / "raw_answer_records.jsonl"
    failures_path = ROOT / "request_failures.jsonl"
    if raw_path.exists() or failures_path.exists():
        raise RuntimeError("Output checkpoint already exists; refusing to overwrite")

    first = probe["record"]
    first["config_id"] = "A"
    append_jsonl(raw_path, first)
    llm = first.get("trace", {}).get("llm_call") or {}
    last_attempt = int(llm.get("attempt", 0))
    cumulative_cost = float(llm.get("cumulative_estimated_cost_cny", 0.0))
    completed = 1

    configs = {
        "A": {"candidate_k": 10, "final_k": 3, "rerank_enabled": False},
        "B": {"candidate_k": 10, "final_k": 5, "rerank_enabled": False},
    }
    started = datetime.now(timezone.utc).isoformat()
    for config_id, params in configs.items():
        for case in cases:
            if config_id == "A" and case["case_id"] == first["case_id"]:
                continue
            if last_attempt >= MAX_CALLS:
                raise RuntimeError(f"Call cap reached at attempt {last_attempt}")
            if cumulative_cost >= STOP_THRESHOLD_CNY:
                raise RuntimeError(f"Budget guard reached at CNY {cumulative_cost:.6f}")
            payload = {
                "case_id": case["case_id"],
                "question": case["question"],
                "chain": "ask",
                "candidate_k": params["candidate_k"],
                "final_k": params["final_k"],
                "faiss_multiplier": 2,
                "min_rerank_score": 0.2,
                "rerank_enabled": params["rerank_enabled"],
                "max_context_chars": 6000,
                "max_chunk_chars": 1200,
                "cache_mode": "bypass",
            }
            before = time.perf_counter()
            try:
                record = request_json("POST", f"/evaluation/runs/{run_ids[config_id]}/cases", payload)
                record["config_id"] = config_id
                record["client_wall_ms"] = round((time.perf_counter() - before) * 1000, 3)
                append_jsonl(raw_path, record)
                llm = record.get("trace", {}).get("llm_call") or {}
                attempt = int(llm.get("attempt", last_attempt + 1))
                if attempt != last_attempt + 1:
                    raise RuntimeError(f"Non-sequential provider attempt: {last_attempt} -> {attempt}")
                last_attempt = attempt
                cumulative_cost = float(llm.get("cumulative_estimated_cost_cny", cumulative_cost))
                completed += 1
            except Exception as exc:
                # No retry. Preserve every failed client request locally; failures
                # still count toward the provider cap if the request reached it.
                failure = {
                    "config_id": config_id,
                    "case_id": case["case_id"],
                    "run_id": run_ids[config_id],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "client_wall_ms": round((time.perf_counter() - before) * 1000, 3),
                    "no_retry": True,
                }
                append_jsonl(failures_path, failure)
                raise
            print(
                f"progress={completed}/{MAX_CALLS} config={config_id} case={case['case_id']} "
                f"attempt={last_attempt} cost_cny={cumulative_cost:.6f}",
                flush=True,
            )

    manifest = {
        "experiment_id": experiment_id,
        "run_ids": run_ids,
        "dataset_path": str(DATASET_PATH.relative_to(PROJECT)).replace("\\", "/"),
        "case_count": len(cases),
        "configs": configs,
        "provider_call_attempts": last_attempt,
        "completed_records": completed,
        "cumulative_estimated_cost_cny": cumulative_cost,
        "budget_cny": BUDGET_CNY,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
