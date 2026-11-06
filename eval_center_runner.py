"""Durable runs around the existing RAG retrieval and answer paths."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import logging
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from eval_center_data import map_anchor
from eval_center_metrics import METRICS_VERSION, retrieval_metrics
from eval_center_store import packed, run
from knowledge_v2 import KnowledgeError
from retrieval_engine import RetrievalConfig, identity as retrieval_identity

logger = logging.getLogger(__name__)
worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evaluation-center")


def runtime_identity(app):
    from isolated_config import verified_model_metadata
    embedding = verified_model_metadata('RAG_V2_EMBED_PATH')
    reranker = verified_model_metadata('RAG_V2_RERANK_PATH')
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL, timeout=4).strip()
    except (OSError, subprocess.SubprocessError):
        commit = os.getenv("RAG_V2_CODE_COMMIT")
    if not commit:
        raise KnowledgeError("Cannot determine Git commit; set RAG_V2_CODE_COMMIT for packaged deployments")
    api = root / "api_app.py"
    source = api.read_text(encoding="utf-8")
    prompt = source.split("# 7. 构建 prompt", 1)[1].split("# 8.", 1)[0]
    fingerprint = sha256()
    for path in sorted(root.glob("*.py")):
        if not path.name.startswith("test_"):
            fingerprint.update(path.name.encode() + path.read_bytes())
    return retrieval_identity() | {
        "agent_router_model": os.getenv("RAG_V3_ROUTER_MODEL"),
        "agent_router_available": os.getenv("RAG_V3_ROUTER_ENABLED") == "1",
        "semantic_generator": os.getenv("RAG_V2_SEMANTIC_GENERATOR"),
        "semantic_judge": os.getenv("RAG_V2_SEMANTIC_JUDGE"),
        "code_commit": commit, "working_code_sha256": fingerprint.hexdigest(),
        "prompt_version": sha256(prompt.encode()).hexdigest(),
        "embedding_model": embedding['name'],
        "embedding_model_version": embedding['revision'],
        "embedding_weights_sha256": embedding['weights_sha256'],
        "embedding_model_location": embedding['location'],
        "rerank_model": reranker['name'],
        "rerank_model_version": reranker['revision'],
        "rerank_weights_sha256": reranker['weights_sha256'],
        "rerank_model_location": reranker['location'],
        "generation_model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "generation_model_revision": os.getenv("RAG_V2_GENERATION_REVISION", "unavailable"),
        "runtime_config": app.state.knowledge_model_config,
    }


def load_evaluation_snapshot(store, config):
    snapshot_id = config["index_snapshot_id"]
    if config.get("snapshot_mode") == "draft_validation":
        if store.snapshot_status(snapshot_id)["manifest_sha"] != config["snapshot_manifest_sha256"]:
            raise KnowledgeError("Draft manifest changed; create a new run", 409)
        return store.load_snapshot(snapshot_id, strict_sparse=True)
    return store.load_snapshot(snapshot_id, config["publish_generation"])


def capture(store, app, options):
    snapshot_id = options["index_snapshot_id"]
    manifest_hash = store.snapshot_status(snapshot_id)["manifest_sha"]
    options = options | {"snapshot_manifest_sha256": manifest_hash}
    draft = options.get("snapshot_mode") == "draft_validation"
    view = load_evaluation_snapshot(store, options)
    if view.corpus_snapshot_id != options["corpus_snapshot_id"]:
        raise KnowledgeError("Corpus snapshot does not match index snapshot", 409)
    if not draft:
        with store.transaction() as cursor:
            cursor.execute("SELECT snapshot_id FROM rag_v2_publications WHERE publish_generation=%s",
                           (options["publish_generation"],))
            row = cursor.fetchone()
            if not row or row["snapshot_id"] != view.snapshot_id:
                raise KnowledgeError("Index and publish generation mismatch", 409)
    return runtime_identity(app) | options | {
        "metrics_version": METRICS_VERSION, "cache_mode": "bypass",
        "vector_backend": "milvus" if getattr(view,"dense_binding",None) else "faiss",
        "dense_collection": (getattr(view,"dense_binding",None) or {}).get("collection"),
        "dense_records_sha256": (getattr(view,"dense_binding",None) or {}).get("records_sha256"),
        "bm25_profile": view.manifest.get("bm25", {}).get("profile"),
        "chunking_config_hash": sha256(packed([m["processing_config"] for m in view.manifest["members"]]).encode()).hexdigest(),
    }


def start_run(store, app, identifier):
    with store.transaction() as cursor:
        record = run(cursor, identifier, lock=True)
        if any(record["config"].get(k) != v for k, v in runtime_identity(app).items()):
            raise KnowledgeError("Runtime drift; create a new experiment", 409)
        if (record["config"]["mode"] == "answer" and not record["config"].get("reliability_enabled")
                and os.getenv("RAG_V2_EVAL_ENABLE_GENERATION") != "1"):
            raise KnowledgeError("Answer generation is disabled in this environment", 409)
        cursor.execute("""UPDATE rag_eval_center_runs SET state='queued',error_message=NULL,finished_at=NULL,
                       heartbeat_at=NOW(6),worker_id=NULL WHERE run_id=%s AND
                       (state IN ('created','failed','cancelled') OR
                        (state IN ('queued','running','cancelling') AND heartbeat_at < NOW(6)-INTERVAL 5 MINUTE))""",
                       (identifier,))
        if cursor.rowcount != 1:
            raise KnowledgeError("Run already active or completed", 409)
    worker.submit(execute, store, identifier)


def annotate_rows(view, rows, visible=None):
    members = {m["document_version_id"]: m for m in view.manifest["members"]}
    enriched = []
    for rank, item in enumerate(rows, 1):
        child = view.manifest["children"][item["faiss_index"]]
        enriched.append(child | item | {
            "rank": rank,
            "index_snapshot_id": view.snapshot_id,
            "source_content_hash": members[child["document_version_id"]]["content_hash"],
            "visible_end": min(child["clean_end"], child["clean_start"] + visible.get(item["faiss_index"], len(child["text"])))
            if visible is not None else child["clean_end"],
        })
    return enriched


def evaluate(store, view, cfg, question, evidence):
    from api_app import AskRequest, PolicyRAGRequest, _ask_question, _query_policy_rag
    if cfg.get("reliability_enabled"):
        from api_app import app
        from reliability import ReliabilityOptions, control
        from retrieval_engine import ChatTurn
        from semantic_reliability import run_semantic
        settings = question.get("reliability_input", {}) | (cfg.get("semantic_options") or {})
        if cfg.get("snapshot_mode") == "draft_validation":
            settings = settings | {"graph_enabled": False}
        args = (store, view, question["question"],
                [ChatTurn.model_validate(t) for t in question.get("chat_history", [])],
                ReliabilityOptions.model_validate(settings), app.state.embed_model)
        semantic = cfg.get("reliability_validator") == "semantic"
        result = (run_semantic(*args, reranker=app.state.rerank_model if cfg["rerank_enabled"] else None)
                  if semantic else control(*args, app.state.rerank_model))
        from eval_center_metrics import ratio
        attempts = result["retrieval_attempts"]
        validation = result["evidence_validation"]
        claims = result["citation_validation"]["claims"]
        cohort = cfg["review_cohort"]
        eligible = question["review_status"] in ({"developer_self_reviewed"} if cohort == "developer" else
                                                    {"independently_reviewed", "adjudicated"})
        metrics = {
            "route_fallback": ratio(int(any(a.get("error") or a.get("retrieval", {}).get("degraded") for a in attempts)), int(bool(attempts))),
            "structural_pass": ratio(int(not validation.get("invalid_rows")), int(any(a.get("candidate_child_ids") for a in attempts))),
            "claim_citation_mapping": ratio(sum(bool(c["citations"]) for c in claims), len(claims)),
            "reviewed_action_match": ratio(int(result["answer_action"].lower() == question["expected_action"]) if eligible else 0,
                                          int(eligible), "question not reviewed for this cohort"),
        }
        router_decision = result.get("retrieval_plan", {}).get("router_decision", {})
        metrics["rule_fallback_rate"] = ratio(int(router_decision.get("status") == "fallback"), int(bool(router_decision)))
        metrics["router_decision_accuracy"] = ratio(reason="Independent route Gold not supplied")
        usage = result["model_usage"] if semantic else []
        engineering = {"total_ms": result["stage_latency"]["total_ms"],
                       "model_calls": len(usage)+int(bool(router_decision.get("attempted"))),
                       "repair_count": result.get("repair_count", 0), "retrieval_rounds": len(attempts)}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            engineering[key] = sum(x["usage"][key] for x in usage) if usage and all(
                x.get("usage", {}).get(key) is not None for x in usage if x.get("usage") is not None
            ) and all(x.get("usage") is not None for x in usage) else None
        return dict(question=question, evidence=evidence, trace=result,
                    result={"answer": result["final_answer"], "answer_action": result["answer_action"]},
                    state="completed", metrics={}, failures=[],
                    engineering=engineering,
                    reliability={"action": result["answer_action"], "rounds": len(result["retrieval_attempts"]),
                                 "structurally_valid": not validation.get("invalid_rows") and validation["status"] != "INVALID",
                                 "timed_out": result.get("budget", {}).get("timed_out", result.get("error") in {
                                     "TimeoutError", "APITimeoutError"}), "metrics": metrics})
    started = time.perf_counter()
    trace = {}
    params = {"question": question["question"], "top_k": cfg["final_k"], "min_rerank_score": cfg["rerank_threshold"],
              "chat_history": question.get("chat_history", [])}
    strategy = RetrievalConfig.model_validate({key: cfg[key] for key in RetrievalConfig.model_fields if key in cfg})
    flags = {"candidate_k": cfg["candidate_k"], "rerank_enabled": cfg["rerank_enabled"], "retrieval": strategy}
    retrieval = cfg["mode"] == "retrieval"
    result = (_query_policy_rag(PolicyRAGRequest(**params), evaluation_options=flags, trace=trace, snapshot=view)
              if retrieval else _ask_question(AskRequest(**params), evaluation_options=flags, trace=trace, snapshot=view))
    total = (time.perf_counter() - started) * 1000
    failed = (result.get("code") in {"RAG_QUERY_FAILED", "RAG_EMPTY", "POLICY_METADATA_CONFLICT"}
              if retrieval else trace.get("status") == "error")
    plan = trace.get("retrieval", {}).get("query", {})
    hyde = plan.get("hyde", {})
    stages = [trace.get("usage") if trace.get("model_calls") else None,
              hyde.get("usage") if hyde.get("model_calls") else None]
    usage = [metadata for metadata, active in zip(stages, [trace.get("model_calls"), hyde.get("model_calls")]) if active]
    def tokens(key):
        if not usage or any(data is None or data.get(key) is None for data in usage):
            return None
        return sum(data[key] for data in usage)
    timing = result.get("timing") or {}
    retrieval_times = [timing.get("embedding_ms"), timing.get("faiss_ms", timing.get("faiss_search_ms"))]
    engineering = {"total_ms": total, "retrieval_ms": sum(retrieval_times) if None not in retrieval_times else None,
                   "rerank_ms": timing.get("rerank_ms"), "llm_ms": None if retrieval else timing.get("llm_ms"),
                   "model_calls": trace.get("model_calls", 0) + hyde.get("model_calls", 0), "prompt_tokens": tokens("input_tokens"),
                   "completion_tokens": tokens("output_tokens"), "total_tokens": tokens("total_tokens"), "cost": None,
                   "hyde_ms": hyde.get("ms")}
    candidates = annotate_rows(view, trace.get("faiss_candidates", []))
    ranking = annotate_rows(view, trace.get("ranked_candidates", []))
    by_index = {row["faiss_index"]: row for row in candidates}
    rank_scores = {row["faiss_index"]: row for row in trace.get("ranked_candidates", [])}
    visible = {row["faiss_index"]: row["visible_chars"] for row in trace.get("selected_evidence", [])}
    final = annotate_rows(view, [by_index[i] | rank_scores.get(i, {}) for i in trace.get("used_indices", [])],
                          visible if not retrieval else None)
    mappings = {e["evidence_id"]: map_anchor(store, view, e) for e in evidence}
    result_body = dict(question=question, evidence=evidence, mappings=mappings, candidates=candidates, ranked=ranking,
                       final=final, trace=trace, result=result, state="error" if failed else "completed",
                       engineering=engineering, metrics={}, failures=[])
    degradation = list(trace.get("retrieval", {}).get("degraded", []))
    degradation.extend(key for key in ("hyde", "rewrite") if plan.get(key, {}).get("status") == "fallback")
    result_body["degradation"] = degradation
    if retrieval and not failed and not degradation:
        cohort = cfg["review_cohort"]
        allowed = {"independent_human": {"independently_reviewed", "adjudicated"},
                   "developer": {"developer_self_reviewed"}}[cohort]
        scored_q = question if question["review_status"] in allowed else question | {"review_status": "unreviewed"}
        scores = retrieval_metrics(scored_q, [e for e in evidence if e["review_status"] in allowed],
                                   mappings, candidates, final, cfg)
        result_body["metrics"] = scores
        approved = [e for e in evidence if e["review_status"] in allowed]
        def coverage(rows, k):
            return retrieval_metrics(scored_q, approved, mappings, candidates, rows, cfg | {"final_k": k})[
                "required_fact_evidence_coverage"]["value"]
        candidate_coverage = coverage(candidates, cfg["candidate_k"])
        raw_top_coverage = coverage(candidates, cfg["final_k"])
        ranked_coverage = coverage(ranking, cfg["final_k"])
        result_body["evidence_coverage_by_stage"] = {"candidate": candidate_coverage,
            "raw_top_k": raw_top_coverage, "ranked_top_k": ranked_coverage,
            "final": scores["required_fact_evidence_coverage"]["value"]}
        if candidate_coverage is not None and candidate_coverage < 1:
            result_body["failures"].append("RETRIEVAL_MISS")
        if cfg["rerank_enabled"] and raw_top_coverage == 1 and ranked_coverage is not None and ranked_coverage < 1:
            result_body["failures"].append("RERANK_LOSS")
        final_coverage = scores["required_fact_evidence_coverage"]["value"]
        if final_coverage is not None and final_coverage < 1:
            result_body["failures"].append("EVIDENCE_INSUFFICIENT")
        result_body["failure_stage"] = ("threshold_or_gate" if ranked_coverage == 1 and final_coverage != 1
                                         else "selection" if result_body["failures"] else None)
    result_body["primary_failure"] = next(iter(result_body["failures"]), None)
    if cfg.get("graph_enabled"):
        from eval_center_metrics import ranking, ratio
        graph = trace.get("retrieval", {}).get("graph", {})
        paths = graph.get("paths", [])
        route = next((r for r in trace.get("retrieval", {}).get("routes", []) if r["id"] == "graph"), {})
        gold = question.get("graph_gold", {})
        cohort_status = ({"developer_self_reviewed"} if cfg["review_cohort"] == "developer"
                         else {"independently_reviewed", "adjudicated"})
        eligible = (question["review_status"] in cohort_status and gold.get("review_status") in cohort_status
                    and gold.get("snapshot_id") == view.snapshot_id and route.get("status") == "used")
        ids = [x["child_id"] for x in route.get("results", [])]
        expected = set(gold.get("child_ids", [])) if eligible else set()
        scores = ranking(ids, ids, expected, cfg["candidate_k"], cfg["final_k"], "No eligible graph Gold")
        entities = set(gold.get("seed_ids", [])) if eligible else set()
        found = set(graph.get("seeds", []))
        complete = [tuple(p) for p in gold.get("required_paths", [])] if eligible else []
        actual = {tuple(p["relation_ids"]) for p in paths}
        audited = set(gold.get("audited_extraction_ids", [])) if eligible else set()
        expected_relations = set(gold.get("extraction_relation_ids", [])) if eligible else set()
        extracted = set()
        records = []
        if audited or (eligible and gold.get("extraction_exhaustive") and expected_relations):
            with store.transaction() as cursor:
                cursor.execute("SELECT relation_id FROM kg_relations WHERE snapshot_id=%s LIMIT 5001", (view.snapshot_id,))
                records = cursor.fetchall()
            if len(records) <= 5000:
                extracted = {r["relation_id"] for r in records}
        result_body["metrics"].update({
            "graph_evidence_recall": scores["final_recall"], "graph_mrr": scores["mrr"], "graph_ndcg": scores["ndcg"],
            "entity_linking_accuracy": ratio(len(found & entities), len(found) if entities else 0),
            "entity_coverage": ratio(len(found & entities), len(entities)),
            "multi_hop_evidence_recall": ratio(sum(p in actual for p in complete if len(p) > 1),
                                               sum(len(p) > 1 for p in complete)),
            "graph_evidence_completeness": ratio(sum(p in actual for p in complete), len(complete)),
            "relation_extraction_precision": ratio(len(extracted & audited & expected_relations), len(extracted & audited)),
            "relation_extraction_recall": ratio(len(extracted & expected_relations),
                len(expected_relations) if eligible and gold.get("extraction_exhaustive") and len(records) <= 5000 else 0),
        })
    return result_body


def execute(store, identifier):
    owner = uuid4().hex
    try:
        with store.transaction() as cursor:
            cursor.execute("""UPDATE rag_eval_center_runs SET state='running',worker_id=%s,
                           started_at=COALESCE(started_at,NOW(6)),heartbeat_at=NOW(6)
                           WHERE run_id=%s AND state='queued'""", (owner, identifier))
            if cursor.rowcount != 1:
                return
            record = run(cursor, identifier)
            cursor.execute("SELECT question_id FROM rag_eval_center_cases WHERE run_id=%s", (identifier,))
            done = {r["question_id"] for r in cursor.fetchall()}
        cfg, inputs = record["config"], record["inputs"]
        view = load_evaluation_snapshot(store, cfg)
        for question in inputs["questions"]:
            qid = question["question_id"]
            if qid in done:
                continue
            with store.transaction() as cursor:
                active = run(cursor, identifier, lock=True)
                if active["state"] != "running" or active["worker_id"] != owner:
                    break
                cursor.execute("UPDATE rag_eval_center_runs SET heartbeat_at=NOW(6) WHERE run_id=%s", (identifier,))
            evidence = [e for e in inputs["evidence"] if e["question_id"] == qid]
            try:
                body = evaluate(store, view, cfg, question, evidence)
            except Exception as exc:
                logger.exception("Evaluation case error: %s/%s", identifier, qid)
                body = dict(question=question, evidence=evidence, state="error", error_type=type(exc).__name__,
                            metrics={}, failures=[])
            with store.transaction() as cursor:
                active = run(cursor, identifier, lock=True)
                if active["worker_id"] != owner or active["state"] not in {"running", "cancelling"}:
                    return
                cursor.execute("INSERT INTO rag_eval_center_cases(run_id,question_id,body) VALUES(%s,%s,%s)",
                               (identifier, qid, packed(body)))
                cursor.execute("""UPDATE rag_eval_center_runs SET completed=completed+1,heartbeat_at=NOW(6)
                               WHERE run_id=%s""", (identifier,))
        with store.transaction() as cursor:
            cursor.execute("""UPDATE rag_eval_center_runs SET state=IF(state='cancelling','cancelled','completed'),
                           finished_at=NOW(6),worker_id=NULL WHERE run_id=%s AND worker_id=%s""", (identifier, owner))
    except Exception as exc:
        logger.exception("Evaluation run error: %s", identifier)
        with store.transaction() as cursor:
            cursor.execute("""UPDATE rag_eval_center_runs SET state='failed',error_message=%s,finished_at=NOW(6),
                           worker_id=NULL WHERE run_id=%s AND worker_id=%s""", (type(exc).__name__, identifier, owner))
