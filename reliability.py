"""Opt-in, extractive policy controller over the existing snapshot retriever."""
from datetime import date
import logging
import re
from time import perf_counter
from uuid import uuid4
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from knowledge_v2 import KnowledgeError, read_checked
from retrieval_engine import CONTEXTUAL, RetrievalConfig, TOPICS, rescore, retrieve

logger = logging.getLogger(__name__)
VERSION = "extractive-policy-v1"


from request_budget import request_scoped


class ReliabilityOptions(BaseModel):
    model_router: bool = False
    max_tool_calls: int = Field(default=24, ge=0, le=32)
    max_model_calls: int = Field(default=8, ge=0, le=16)
    max_graph_queries: int = Field(default=8, ge=0, le=16)
    max_final_evidence: int = Field(default=10, ge=1, le=30)
    max_sub_queries: int = Field(default=3, ge=1, le=5)
    max_graph_hops: int = Field(default=2, ge=1, le=2)
    graph_timeout_ms: int = Field(default=1000, ge=50, le=5000)
    product: str | None = Field(default=None, max_length=150)
    category: str | None = Field(default=None, max_length=150)
    graph_enabled: bool = False
    model_config = ConfigDict(extra="forbid")
    scope: str | None = Field(default=None, min_length=1, max_length=80)
    facts: list[str] = Field(default_factory=list, max_length=5)
    slots: dict[str, str] = Field(default_factory=dict, max_length=12)
    as_of: date | None = None
    max_retrieval_attempts: int = Field(default=2, ge=1, le=2)
    token_budget: int = Field(default=4096, ge=128, le=65536)
    timeout_budget_ms: int = Field(default=5000, ge=50, le=60000)


class PolicyRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact: str = Field(min_length=1, max_length=80)
    scope: str = Field(min_length=1, max_length=80)
    verdict: str = Field(min_length=1, max_length=160)
    when: dict[str, str] = Field(default_factory=dict, max_length=8)
    unless: dict[str, str] = Field(default_factory=dict, max_length=8)
    starts: date | None = None
    ends: date | None = None
    policy_id: str | None = Field(default=None,max_length=100)
    policy_version: str | None = Field(default=None,max_length=80)
    policy_relation_id: str | None = Field(default=None,pattern=r'^[a-f0-9]{64}$')
    applicable_products: list[str] = Field(default_factory=list,max_length=20)
    applicable_categories: list[str] = Field(default_factory=list,max_length=20)
    effect: Literal['ALLOW','DENY'] | None = None

def rule_binding(store,view,child,rule,options):
    """A RULE never grants its own policy authority."""
    from graph_governance import policy_rows,policy_decisions,assess_policies
    if not (rule.policy_id and rule.policy_version and rule.policy_relation_id and rule.effect
            and (rule.applicable_products or rule.applicable_categories)):return None
    with store.transaction() as c:
        c.execute("SELECT task_status FROM kg_builds WHERE snapshot_id=%s",(view.snapshot_id,));job=c.fetchone()
        if not job or job['task_status']!='SEALED':return None
        policies=policy_rows(c,view.snapshot_id)
        c.execute("SELECT r.*,s.polarity FROM kg_relations r JOIN kg_relation_semantics s ON s.snapshot_id=r.snapshot_id AND s.relation_id=r.relation_id WHERE r.snapshot_id=%s AND r.status='APPROVED'",(view.snapshot_id,))
        edges=c.fetchall()
    p=next((p for p in policies if p['relation_id']==rule.policy_relation_id),None)
    edge=next((e for e in edges if e['relation_id']==rule.policy_relation_id),None)
    if not p or not edge:return None
    meta=p['metadata']
    denied=(edge['relation_type']=='EXCLUDES') != (edge.get('polarity')=='negated')
    if (p['policy_id']!=rule.policy_id or p['policy_version']!=rule.policy_version
        or p['chunk_id']!=child['child_id'] or meta.get('review_status')!='APPROVED' or not meta.get('reviewed_by')
        or edge['relation_type'] not in {'ALLOWS','EXCLUDES'} or (rule.effect=='DENY')!=denied
        or meta['scope']!=rule.scope or set(meta['applicable_products'])!=set(rule.applicable_products)
        or set(meta['applicable_categories'])!=set(rule.applicable_categories)
        or meta['conditions']!=rule.when or meta['exceptions']!=rule.unless
        or meta['valid_from']!=(rule.starts.isoformat() if rule.starts else None)
        or meta['valid_until']!=(rule.ends.isoformat() if rule.ends else None)):return None
    context={'scope':options.scope,'product':options.product,'category':options.category,'slots':options.slots}
    day=(options.as_of or date.today()).isoformat()
    decision=policy_decisions(policies,day,context).get(rule.policy_relation_id)
    if decision in {'SUPERSEDED','WITHDRAWN','TIME_DIFFERENCE','UNRESOLVED'}:return None
    assessment=assess_policies([p for p in policies if p['metadata']['scope']==rule.scope],edges,day,context)
    return {'snapshot_id':view.snapshot_id,'policy_id':p['policy_id'],'policy_version':p['policy_version'],
            'relation_id':p['relation_id'],'reviewed_by':meta['reviewed_by'],'decision':decision,
            'conflicts':assessment['conflicts']}


def route(question, history, options):
    facts = list(dict.fromkeys(f.strip() for f in options.facts if f.strip()))
    if not facts:
        facts = [topic for topic in TOPICS if topic in question][:5]
        if "运费" in facts and "费用" in facts:
            facts.remove("费用")
    rewrite = bool(history and CONTEXTUAL.search(question))
    complex_query = len(facts) > 1 or len(re.findall(r"[？?；;]", question)) > 1
    precise = bool(re.search(r"[0-9]|[A-Z]{2,}-|编号|型号", question))
    use_graph = options.graph_enabled and (complex_query or precise or bool(re.search("关联|适用|例外|条件", question)))
    config = RetrievalConfig(retrieval_mode="hybrid", bm25_enabled=True, rrf_enabled=True,
                             query_rewrite_enabled=rewrite, query_decomposition_enabled=complex_query,parent_child_enabled=True,
                             graph_enabled=use_graph,
                             graph_timeout_ms=options.graph_timeout_ms,
                             bm25_candidate_k=16 if precise else 8, dense_candidate_k=8)
    query_type = ("multi_condition" if complex_query else "contextual" if rewrite else
                  "exact" if precise else "simple")
    return config, {**options.model_dump(mode="json"), "query_type": query_type, "complexity": "high" if complex_query else "low",
                    "routing_reason": "deterministic question signals; conflict and gaps checked after recall",
                    "facts": facts, "strategy": config.model_dump()}


def verified_rules(store, view, row, options=None):
    from request_budget import checkpoint
    checkpoint()
    child = view.manifest["children"][row]
    doc = view.all_docs[row]
    member = next(m for m in view.manifest["members"] if m["document_version_id"] == child["document_version_id"])
    parent = next(p for p in view.manifest["parents"] if p["parent_id"] == child["parent_id"])
    if (doc.page_content != child["text"] or parent["document_version_id"] != child["document_version_id"]
            or member["legacy_file_hash"] != child["file_hash"] or member["source"] != child["source"]
            or member["policy_version"] != child["version"]
            or parent["page"] != child["page"] or
            any(doc.metadata.get(k) != child[k] for k in ("child_id", "chunk_id", "source", "file_hash", "version"))
            or doc.metadata.get("index_snapshot_id") != view.snapshot_id):
        raise ValueError("Child identity mismatch")
    if "RULE " not in child["text"]:
        return []
    version = child["document_version_id"]
    read_checked(store.root / "documents" / version / "source.bin", member["content_hash"])
    artifact = store.document_artifact(version)
    original = next(p["text"] for p in artifact["pages"] if p["page"] == child["page"])
    if not 0 <= child["start"] < child["end"] <= len(original):
        raise ValueError("Child raw coordinates invalid")
    fragment = original[child["start"]:child["end"]]
    original_lines = {}
    offset = child["start"]
    for raw_line in fragment.splitlines(keepends=True):
        original_lines.setdefault(raw_line.rstrip("\r\n"), offset)
        offset += len(raw_line)
    result = []
    for line in child["text"].splitlines():
        checkpoint()
        if not line.startswith("RULE ") or line not in original_lines:
            continue
        try:
            rule = PolicyRule.model_validate_json(line[5:])
        except ValidationError:
            continue
        if rule.starts and rule.ends and rule.starts > rule.ends:
            continue
        binding=rule_binding(store,view,child,rule,options or ReliabilityOptions())
        if not binding:continue
        citation = {k: child[k] for k in ("child_id", "chunk_id", "parent_id", "document_version_id",
                    "source", "file_hash", "version", "page")}
        start = original_lines[line]
        result.append((rule, citation | {"start": start, "end": start + len(line), "quote": line,
                        "index_snapshot_id": view.snapshot_id,'policy_binding':binding}))
    return result


def validate(facts, rules, options):
    missing, conflicts, claims, gaps = [], [], [], []
    for fact in facts:
        matches = [(rule, citation) for rule, citation in rules if rule.fact == fact]
        scoped = [(r, c) for r, c in matches if r.scope == options.scope]
        if matches and not options.scope:
            missing.append("适用商品或政策范围")
        applicable = []
        for rule, citation in scoped:
            if rule.applicable_products:
                if not options.product:missing.append('product');continue
                if options.product not in rule.applicable_products:continue
            if rule.applicable_categories:
                if not options.category:missing.append('category');continue
                if options.category not in rule.applicable_categories:continue
            if citation.get('policy_binding',{}).get('conflicts'):
                conflicts.extend(citation['policy_binding']['conflicts']);continue
            if options.as_of and ((rule.starts and options.as_of < rule.starts) or
                                  (rule.ends and options.as_of > rule.ends)):
                continue
            if any(k in options.slots and options.slots[k] != v for k, v in rule.when.items()):
                continue
            if any(options.slots.get(k) == v for k, v in rule.unless.items()):
                continue
            unknown = sorted((rule.when.keys() | rule.unless.keys()) - options.slots.keys())
            if (rule.starts or rule.ends) and not options.as_of:
                unknown.append("适用日期")
            if unknown:
                missing.extend(unknown)
                continue
            applicable.append((rule, citation))
        verdicts = {r.verdict for r, _ in applicable}
        if len(verdicts) > 1:
            conflicts.append({"fact": fact, "policies": [{"rule": r.model_dump(mode="json"), "citation": c}
                                                          for r, c in applicable]})
        elif applicable:
            citations = {(c["child_id"], c.get("start")): c for _, c in applicable}
            claims.append({"fact": fact, "statement": applicable[0][0].verdict,
                           "rules": [r.model_dump(mode="json") for r, _ in applicable],
                           "citations": list(citations.values())})
        else:
            gaps.append(fact)
    return claims, conflicts, sorted(set(missing)), gaps


def render_claim(claim):
    references = ", ".join(dict.fromkeys(c["child_id"] for c in claim["citations"]))
    conditions = []
    for rule in claim["rules"]:
        parts = [f"范围 {rule['scope']}"]
        parts.extend(f"{key}为{value}" for key, value in rule["when"].items())
        parts.extend(f"排除{key}为{value}的情况" for key, value in rule["unless"].items())
        if rule["starts"] or rule["ends"]:
            parts.append(f"适用日期 {rule['starts'] or '不限'} 至 {rule['ends'] or '不限'}")
        conditions.append("，".join(parts))
    return f"{claim['fact']}：{claim['statement']} [{references}]。适用条件：{'；'.join(dict.fromkeys(conditions))}。"


@request_scoped
def control(store, view, question, history, options, embedder, reranker):
    from agent_router import plan_router
    started = perf_counter()
    from request_budget import current
    deadline = min(current.get().deadline, started + options.timeout_budget_ms / 1000)
    strategy, plan, router_tokens = plan_router(question, history, options, deadline)
    attempts, evidence, invalid = [], {}, []
    claims, conflicts, missing, gaps = [], [], [], plan["facts"]
    policy_blocked = False
    remaining = options.token_budget - router_tokens
    tool_budget = [options.max_tool_calls]
    for index in range(options.max_retrieval_attempts if gaps else 0):
        query = question if not index else " ".join([options.scope or "", *gaps, *options.slots.values()]).strip()
        if (perf_counter() >= deadline or (tool_budget is not None and not tool_budget[0])
                or not query or any(a["query"] == query for a in attempts)):
            break
        # Conservative budget for all auxiliary variants, including replicated history.
        query_bytes = len(query.encode()) + (sum(len(t.content.encode()) for t in history[-2:]) if not index else 0)
        reserve = 8 * (query_bytes + 80) if strategy.query_decomposition_enabled else 2 * (query_bytes + 80)
        if reserve > remaining:
            break
        remaining -= reserve
        config = strategy if not index else RetrievalConfig(retrieval_mode="hybrid", bm25_enabled=True,
                 rrf_enabled=True, bm25_candidate_k=32, dense_candidate_k=16, graph_enabled=strategy.graph_enabled)
        attempt = {"query": query, "strategy": config.model_dump(), "reason": "missing_facts" if index else "initial"}
        attempts.append(attempt)
        try:
            retrieved = retrieve(view, query, history if not index else [], config, 24 if index else 12, embedder,
                                 deadline=deadline, tool_budget=tool_budget,
                                 suggested_subqueries=plan.get("sub_queries") if not index else None,
                                 model_rewrite=plan.get('model_rewrite',{}).get('query') if not index else None,
                                 required_conditions=plan.get('required_conditions'),
                                 policy_context={"scope":options.scope,"slots":options.slots,
                                                 'required_conditions':plan.get('required_conditions',[]),
                                                 "product":options.product,"category":options.category,
                                                 "as_of":options.as_of.isoformat() if options.as_of else None})
            candidates = retrieved.candidates
            attempt["retrieval"] = retrieved.trace
            if any(r.get("family")=="graph" and r.get("status")!="used" for r in retrieved.trace.get("routes",[])):
                policy_blocked = True
                break
            graph = retrieved.trace.get("graph", {})
            graph_parts = [graph, *retrieved.trace.get("graph_subqueries", [])]
            if any(g.get('ambiguous') for g in graph_parts):
                missing=['entity:'+alias for g in graph_parts for alias in g.get('ambiguous',[])];break
            if any(g.get("truncated") or g.get("timed_out") for g in graph_parts):
                policy_blocked = True
                break
            assessments = [g.get("policy_assessment",{}) for g in graph_parts]
            assessment = next((a for a in assessments if a.get("action")=="CONFLICT"),
                              next((a for a in assessments if a.get("action")=="CLARIFY"),
                                   next((a for a in assessments if a.get("policy_count",0) and a.get("action")=="ABSTAIN"),graph.get("policy_assessment",{}))))
            if assessment.get("policy_count",0) and assessment.get("action") == "ABSTAIN":
                policy_blocked = True
                break
            if assessment.get("action") == "CLARIFY":
                missing = assessment["missing_slots"]
                break
            if assessment.get("action") == "CONFLICT":
                conflicts = assessment["conflicts"]
                break
            graph_conflicts = retrieved.trace.get("graph", {}).get("conflicts", [])
            if graph_conflicts:
                from semantic_reliability import evidence_for
                conflicts = [{"relation_ids": conflict["relation_ids"],
                              "evidence": [evidence_for(store, view, row["row"])
                                           for row in conflict["evidence"]]}
                             for conflict in graph_conflicts]
                break
            pairs = [[query, item["doc"].page_content] for item in candidates]
            if pairs and perf_counter() < deadline:
                scores, used = rescore(retrieved, pairs, reranker)
                if used:
                    candidates = [item for _, item in sorted(zip(scores, candidates), key=lambda pair: -pair[0])]
            attempt["candidate_child_ids"] = [item["doc"].metadata.get("child_id") for item in candidates]
            from retrieval_engine import expand_parents
            if hasattr(retrieved,'config') and isinstance(retrieved.config,RetrievalConfig):
                expand_parents(retrieved,candidates[:options.max_final_evidence],512)
            for item in candidates:
                if perf_counter() >= deadline:
                    break
                row = item["faiss_index"]
                if row not in evidence and len(evidence)>=options.max_final_evidence:
                    break
                if row not in evidence:
                    try:
                        evidence[row] = verified_rules(store, view, row, options)
                    except (KnowledgeError, OSError, ValueError, KeyError, StopIteration):
                        invalid.append(row)
                        evidence[row] = []
        except Exception as exc:
            logger.warning("Reliability retrieval failed: %s", type(exc).__name__)
            attempt["error"] = type(exc).__name__
        claims, conflicts, missing, gaps = validate(plan["facts"], [r for group in evidence.values() for r in group], options)
        if conflicts or not gaps or missing:
            break
    timed_out = perf_counter() >= deadline
    from agent_router import required_condition_gaps
    missing=sorted(set(missing)|set(required_condition_gaps(plan,options)))
    sufficient = bool(claims) and not gaps and not missing and not invalid and not timed_out and not policy_blocked
    action = ("ABSTAIN" if timed_out or invalid or policy_blocked else "CONFLICT" if conflicts else
              "CLARIFY" if missing else "ANSWER" if sufficient else "ABSTAIN")
    state = ("INVALID" if invalid else "CONFLICTED" if conflicts else "SUFFICIENT" if sufficient else
             "PARTIALLY_SUPPORTED" if claims else "INSUFFICIENT")
    answer = {"CLARIFY": "请补充：" + "、".join(missing) + "。",
              "ABSTAIN": "当前知识库没有足够依据，不能确认具体结论。",
              "CONFLICT": "当前适用政策条款存在冲突，请核对下列来源后人工裁决。",
              "ANSWER": "\n".join(render_claim(c) for c in claims)}[action]
    return {"trace_id": uuid4().hex, "original_query": question, "retrieval_plan": plan,
            "index_snapshot_id": view.snapshot_id, "corpus_snapshot_id": view.corpus_snapshot_id,
            "publish_generation": view.generation, "retrieval_attempts": attempts,
            "evidence_validation": {"status": state, "invalid_rows": invalid, "missing_facts": gaps,
                                    "missing_slots": missing, "conflicts": conflicts},
            "citation_validation": {"method": VERSION, "claims": claims if action == "ANSWER" else []},
            "answer_action": action, "final_answer": answer,
            "model_usage": {"calls": int(bool(plan.get("router_decision", {}).get("attempted")))},
            "budget": {"estimated_tokens_reserved": options.token_budget - remaining, "timed_out": timed_out,
                       "tool_calls": (options.max_tool_calls-tool_budget[0] if tool_budget is not None else
                                      sum(r.get("status") in {"used","failed"} for a in attempts
                                          for r in a.get("retrieval",{}).get("routes",[])))},
            "stage_latency": {"total_ms": round((perf_counter() - started) * 1000, 2)}}
