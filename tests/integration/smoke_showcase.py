import hashlib, json, sys, subprocess, time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from urllib.parse import urlencode

root = Path(sys.argv[1]).resolve()
repository = Path(__file__).resolve().parents[2]
ids = json.loads((root / "projects.json").read_text())
private = json.loads((root / "secrets/environment.json").read_text())
if not ids["app"].startswith("ragv3-") or ids["milvus"] != ids["app"] + "-milvus":
    raise SystemExit("Use a fresh isolated deployment created by scripts/bootstrap.py")
if private.get("RAG_V2_ISOLATED_LLM") != "1" or private.get("RAG_LLM_BUDGETED_REAL") == "1":
    raise SystemExit("This smoke check requires the local model fixture")
port = ids["ports"]["api"]
header = {"X-RAG-V2-Token": private["RAG_V2_ADMIN_TOKEN"]}
graph_header = {"X-KG-Reviewer": private["RAG_V3_GRAPH_REVIEW_TOKEN"]}
report = {}

def http(path, payload=None, *, expected=200, extra=None, auth=True):
    headers = dict(header) if auth else {}
    if extra:
        headers.update(extra)
    if payload is not None and not isinstance(payload, bytes):
        payload = json.dumps(payload, ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"
    try:
        with urlopen(Request(f"http://127.0.0.1:{port}{path}", data=payload, headers=headers), timeout=120) as response:
            status, body = response.status, response.read()
    except HTTPError as e:
        status, body = e.code, e.read()
    if status != expected:
        raise AssertionError(f"{path}: HTTP {status} expected {expected}; {body[:300]!r}")
    data = json.loads(body)
    return data

health = http("/system/health")
head = http("/v2/index-snapshots")["head"]
assert health["success"] and not health["services"]["rag"]["ready_for_queries"]
assert head["generation"] == 0 and not head["snapshot_id"]
assert http("/__deployment/identity")["vector_count"] == 0
report["empty"] = {"health": health["success"], "generation": head["generation"]}
with urlopen(f"http://127.0.0.1:{ids['ports']['ui']}/_stcore/health") as res:
    assert res.status == 200 and res.read().strip() == b"ok"
report["streamlit_health"] = 200
doc = "验收政策 ZX-V3-DEMO 允许有凭证的申请人申请退款，人工审核后确定结果。"
kg = {"source": {"type": "Policy", "name": "ZX-V3-DEMO", "scope": "general"},
      "target": {"type": "Action", "name": "申请退款", "scope": "general"},
      "predicate": "ALLOWS", "evidence_quote": doc}
body = (doc + "\nKG " + json.dumps(kg, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
prefix = b'--ragboundary\r\nContent-Disposition: form-data; name="file"; filename="showcase-policy.txt"\r\nContent-Type: text/plain\r\n\r\n'
multipart = prefix + body + b"\r\n--ragboundary--\r\n"
ctype = {"Content-Type": "multipart/form-data; boundary=ragboundary"}
http("/v2/documents", multipart, extra=ctype, expected=403, auth=False)
uploaded = http("/v2/documents", multipart, extra=ctype, expected=201)
assert uploaded["content_hash"] == hashlib.sha256(body).hexdigest()
built = http("/v2/index-snapshots", {}, expected=201)
sid = built["snapshot_id"]
chunk = http(f"/v2/index-snapshots/{sid}/chunks")
assert chunk["children"] and chunk["parents"]
assert chunk["children"][0]["file_hash"] == hashlib.md5(body).hexdigest()
report["uploaded"] = {"document_id": uploaded["document_id"], "snapshot_id": sid, "children": chunk["total"]}
http(f"/v3/graph/{sid}/build", {}, expected=201)
pending = http(f"/v3/graph/overview?snapshot_id={sid}")
assert len(pending["relations"]) == 1 and len(pending["entities"]) == 2
pub = {"expected_generation": 0, "idempotency_key": "release-smoke-first"}
http(f"/v2/index-snapshots/{sid}/publish", pub | {"idempotency_key": "release-smoke-graph-gate"}, expected=409)
for kind, key in [("entity", "entities"), ("relation", "relations")]:
    for record in pending[key]:
        http(f"/v3/graph/{sid}/review", {"kind": kind, "id": record[kind + "_id"],
                                      "status": "APPROVED", "note": "Synthetic smoke review"},
             extra=graph_header)
sealed = http(f"/v3/graph/{sid}/seal", {})
assert sealed["status"] == "SEALED"
publication = http(f"/v2/index-snapshots/{sid}/publish", pub)
assert publication["publish_generation"] == 1
lookup = http("/v3/graph/lookup?" + urlencode({"question": "ZX-V3-DEMO"}))
assert lookup["snapshot_id"] == sid and lookup["paths"], lookup
assert chunk["children"][0]["child_id"] in json.dumps(lookup, ensure_ascii=False)
report["graph"] = {"paths": len(lookup["paths"]), "approved_synthetic_relations": 1, "unsealed_publish": 409}

query = {"question": "ZX-V3-DEMO 如何申请退款？", "retrieval": {"retrieval_mode": "hybrid", "dense_enabled": True,
        "bm25_enabled": True, "graph_enabled": True, "rrf_enabled": True, "parent_child_enabled": True,
        "rerank_enabled": False, "threshold_enabled": False}}

def verify_query():
    result = http("/rag/query", query)
    assert result.get("evidence"), result
    item = result["evidence"][0]
    assert item["index_snapshot_id"] == sid and item["child_id"] == chunk["children"][0]["child_id"], result
    assert item["file_hash"] == hashlib.md5(body).hexdigest()
    assert "人工审核" in item["content"] and item["parent_id"]
    return {k: item[k] for k in ("index_snapshot_id", "child_id", "parent_id", "file_hash")}

report["retrieval"] = verify_query()
identity = http("/__deployment/identity")
assert identity["dense_class"] == "MilvusSnapshotIndex" and identity["vector_count"] > 0
assert identity["redis_identity_verified"] and identity["model_fixture"] and not identity["real_llm"]
assert identity["mysql"]["port"] == 43317 and identity["mysql"]["database"] == private["MYSQL_DATABASE"]
report["runtime"] = {k: identity[k] for k in ("generation", "dense_class", "vector_count", "redis_db", "model_fixture", "real_llm")}
assert http("/v2/documents")["active_members"][0]["source"] == "showcase-policy.txt"
plan = query["retrieval"] | {"rerank_enabled": True}
debug = http("/v2/retrieval/debug", {"question": query["question"], "retrieval": plan})
assert debug["trace"]["retrieval"]["rerank"]["status"] == "used"
assert debug["trace"]["retrieval"]["channels"] == {"dense": "used", "sparse": "used", "graph": "used"}
report["channels"] = debug["trace"]["retrieval"]["channels"] | {"crossencoder": "used"}
reliable = http("/v2/retrieval/reliable", {"question": query["question"], "reliability": {"timeout_budget_ms": 30000}})
trace = http(f"/v2/retrieval/reliable/{reliable['trace_id']}")
assert trace["trace_id"] == reliable["trace_id"] and trace["index_snapshot_id"] == sid
report["trace"] = {"id": trace["trace_id"], "action": trace["answer_action"], "read": 200}

dataset = {"dataset_version": "smoke-1", "split": "dev", "name": "Showcase synthetic Gold",
    "questions": [{"question_id": "Q1", "question": query["question"], "question_type": "fixture",
        "answerability": "answerable", "expected_action": "answer", "difficulty": "easy",
        "required_facts": [{"fact_id": "F1", "text": "退款需人工审核", "purpose": "answer",
            "evidence_options": [["E1"]]}]}],
    "evidence": [{"evidence_id": "E1", "question_id": "Q1", "source_id": "showcase",
        "source_path": "showcase-policy.txt", "source_content_hash": hashlib.sha256(body).hexdigest(),
        "original_text_quote": doc, "source_offset": {"start": 0, "end": len(doc)},
        "role": "answer_support", "necessity": "required", "supports_fact_ids": ["F1"]}]}
created = http("/v2/evaluation/datasets", dataset, expected=201)
dataset_id = created["dataset_id"]
dataset_path = f"/v2/evaluation/datasets/{dataset_id}/smoke-1"
for collection, entity in (("questions", "Q1"), ("evidence", "E1")):
    review = http(dataset_path + "/reviews", {"collection": collection, "entity_id": entity,
                  "action": "approve", "note": "Synthetic developer test review"})
    assert review["review_status"] == "developer_self_reviewed"
assert http(dataset_path + "/freeze", {})["frozen"]
listed = http("/v2/evaluation/datasets")
assert dataset_id in json.dumps(listed)
run = http("/v2/evaluation/runs", {
    "dataset_id": dataset_id, "dataset_version": "smoke-1", "corpus_snapshot_id": debug["corpus_snapshot_id"],
    "index_snapshot_id": sid, "publish_generation": 1, "mode": "retrieval",
    "candidate_k": 6, "final_k": 3, **plan}, expected=201)
run_path = f"/v2/evaluation/runs/{run['run_id']}"
http(run_path + "/start", {})
for _ in range(60):
    evaluated = http(run_path)
    state = evaluated["run"]["state"]
    if state == "completed":
        break
    assert state not in {"failed", "cancelled"}, evaluated
    time.sleep(0.5)
else:
    raise AssertionError("Evaluation did not complete")
assert len(evaluated["cases"]) == 1
report["evaluation"] = {"dataset_id": dataset_id, "frozen": True, "run_id": run["run_id"], "cases": 1, "state": state}
subprocess.run(["docker", "compose", "-p", ids["app"], "--env-file", str(root / "compose.env"),
                "-f", "deployment/compose.yml", "restart", "api"], check=True, cwd=repository)
for _ in range(60):
    try:
        if http("/system/health")["success"]:
            break
    except (OSError, HTTPError, ValueError, AssertionError):
        pass
    time.sleep(2)
else:
    raise AssertionError("API did not recover after restart")
assert http("/v2/index-snapshots")["head"]["snapshot_id"] == sid
assert http("/__deployment/identity")["generation"] == 1
assert verify_query() == report["retrieval"]
report["restart"] = {"same_generation": True, "same_snapshot": True, "same_child_evidence": True}
rollback = http(f"/v2/index-snapshots/{sid}/rollback",
                {"expected_generation": 1, "idempotency_key": "showcase-smoke-rollback"})
assert rollback["publish_generation"] == 2
assert not http("/rag/query", query)["cache_hit"]
assert http("/rag/query", query)["cache_hit"]
report["rollback"] = {"snapshot_unchanged": True, "new_generation": 2, "cache_hit_after_refill": True}
print(json.dumps(report, ensure_ascii=True, indent=2))
