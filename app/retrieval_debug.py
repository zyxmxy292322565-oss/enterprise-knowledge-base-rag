"""Authenticated inspection of the shared, cache-bypassed retrieval path."""
import os
import json
from hashlib import sha256
import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from trace_lifecycle import principal, authenticated_store, Principal
from pydantic import Field, model_validator

from knowledge_v2_api import _store, call
from retrieval_engine import RetrievalConfig, RetrievalRequest, identity
from reliability import ReliabilityOptions, control
from semantic_reliability import run_semantic
from knowledge_v2 import json_bytes, write_file

router = APIRouter(prefix="/v2/retrieval", tags=["retrieval-debug"])


class ReliableRequest(RetrievalRequest):
    snapshot_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    question: str = Field(min_length=1, max_length=4000)
    reliability: ReliabilityOptions = Field(default_factory=ReliabilityOptions)
    validator: Literal["rules", "semantic"] = "rules"


def _pin_reliable_view(store, snapshot_id):
    if snapshot_id is None:
        return store.pin()
    from knowledge_v2_api import published_view
    return published_view(store,snapshot_id)


def _reliable(payload: ReliableRequest, store, owner):
    from api_app import app

    from request_budget import stage, current, BudgetExceeded
    from uuid import uuid4
    view = None
    try:
        with stage("snapshot_pin"):
            view = call(_pin_reliable_view,store,payload.snapshot_id)
        result = (run_semantic(store, view, payload.question, payload.chat_history, payload.reliability, app.state.embed_model,reranker=app.state.rerank_model)
                  if payload.validator == "semantic" else
                  control(store, view, payload.question, payload.chat_history, payload.reliability,
                          app.state.embed_model, app.state.rerank_model))
    except BudgetExceeded as exc:
        result = {"trace_id":uuid4().hex,"original_query":payload.question,
                  "index_snapshot_id":getattr(view,"snapshot_id",None),
                  "corpus_snapshot_id":getattr(view,"corpus_snapshot_id",None),
                  "publish_generation":getattr(view,"generation",None),
                  "retrieval_plan":payload.reliability.model_dump(mode="json"),"retrieval_attempts":[],
                  "answer_action":"ABSTAIN","final_answer":"请求预算已用尽或请求已取消，未继续执行。",
                  "evidence_validation":{"status":"INSUFFICIENT"},"citation_validation":{"claims":[]},
                  "budget":{"execution":current.get().snapshot()},"stage_latency":{},"error":str(exc)}
    if current.get() is not None:
        result.setdefault("stage_latency",{})["request_total_ms"] = current.get().snapshot()["elapsed_ms"]
    def persist():
        attempts = [{k: v for k, v in a.items() if k in {"strategy", "reason", "error", "candidate_child_ids",
                                                         "channels", "degraded", "requested"}}
                    | {"channels": a.get("retrieval", {}).get("channels")} for a in result["retrieval_attempts"]]
        saved = {k: v for k, v in result.items() if k not in {"original_query", "retrieval_attempts", "retrieval_plan"}}
        saved["question_sha256"] = sha256(payload.question.encode()).hexdigest()
        saved["retrieval_attempts"] = attempts
        saved["retrieval_plan"] = {k: v for k, v in result.get("retrieval_plan", {}).items()
                                   if k in {"query_type", "complexity", "strategy", "timeout_budget_ms", "token_budget"}}
        router_trace = result.get("retrieval_plan", {}).get("router_decision", {})
        saved["router_decision"] = {k:v for k,v in router_trace.items() if k in {"status","reason","model","ms","usage"}}
        from trace_lifecycle import save,cleanup
        cleanup(store)
        save(store,saved,owner)
    call(persist)
    return result


@router.post("/reliable")
async def reliable(payload: ReliableRequest, request: Request, store=Depends(authenticated_store),owner=Depends(principal)):
    import asyncio
    from threading import Event
    from starlette.concurrency import run_in_threadpool
    from request_budget import cancel_signal, current, ExecutionBudget
    signal=Event()
    token=cancel_signal.set(signal)
    budget_token=current.set(ExecutionBudget(payload.reliability))

    async def watch_disconnect():
        # FastAPI has already parsed the request body. Wait for the actual ASGI
        # disconnect event instead of polling a pre-cancelled receive call.
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                signal.set()
                return

    watcher = asyncio.create_task(watch_disconnect())
    try:
        return await run_in_threadpool(_reliable,payload,store,owner)
    except asyncio.CancelledError:
        signal.set()
        raise
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        current.reset(budget_token)
        cancel_signal.reset(token)


@router.get("/reliable/{trace_id}")
def reliable_trace(trace_id: str, store=Depends(authenticated_store),owner=Depends(principal)):
    from trace_lifecycle import read
    return call(read,store,trace_id,owner)

@router.post('/reliable-traces/cleanup')
def cleanup_traces(store=Depends(_store)):
    from trace_lifecycle import cleanup
    return call(cleanup,store)


class DebugRequest(RetrievalRequest):
    snapshot_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    mode: Literal["retrieval", "answer"] = "retrieval"
    question: str = Field(min_length=1, max_length=4000)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    candidate_k: int = Field(default=12, ge=1, le=200)
    final_k: int = Field(default=4, ge=1, le=50)
    rerank_threshold: float = Field(default=0.2, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_k(self):
        if self.final_k > self.candidate_k:
            raise ValueError("final_k exceeds candidate_k")
        return self


@router.post("/debug")
def debug(payload: DebugRequest, store=Depends(_store)):
    from api_app import AskRequest, PolicyRAGRequest, _ask_question, _query_policy_rag

    if payload.mode == "answer" and os.getenv("RAG_V2_EVAL_ENABLE_GENERATION") != "1":
        raise HTTPException(409, "Answer generation is disabled in this environment")
    from knowledge_v2_api import published_view
    view = call(published_view, store, payload.snapshot_id)
    trace = {}
    options = {"candidate_k": payload.candidate_k, "retrieval": payload.retrieval}
    params = {"question": payload.question, "top_k": payload.final_k, "min_rerank_score": payload.rerank_threshold,
              "chat_history": payload.chat_history}
    result = (_ask_question(AskRequest(**params), evaluation_options=options, trace=trace, snapshot=view)
              if payload.mode == "answer" else
              _query_policy_rag(PolicyRAGRequest(**params), evaluation_options=options, trace=trace, snapshot=view))
    return {"mode": payload.mode, "snapshot_id": view.snapshot_id, "corpus_snapshot_id": view.corpus_snapshot_id,
            "publish_generation": view.generation, "history": [t.model_dump() for t in payload.chat_history],
            "identity": identity(), "trace": trace, "result": result}
