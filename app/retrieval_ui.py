"""Shared retrieval controls and the real backend debugging view."""
import json
import os

import requests
import streamlit as st


def strategy_controls():
    dense = st.checkbox("Dense 向量检索", value=True)
    sparse = st.checkbox("BM25")
    graph = st.checkbox("MySQL Graph") if os.getenv("RAG_V3_GRAPH_ENABLED") == "1" else False
    hops = st.selectbox("图最大跳数", [1, 2], index=1) if graph else 2
    rrf = st.checkbox("RRF 融合")
    rerank = st.checkbox("CrossEncoder 重排", value=True)
    gate = st.checkbox("CE 分数门控", value=True)
    rewrite = st.checkbox("查询改写")
    decomposition = st.checkbox("复杂问题拆解")
    expansion = st.checkbox("同义查询扩展")
    hyde = st.checkbox("HyDE")
    parent = st.checkbox("Parent 上下文")
    with st.expander("检索预算"):
        dense_k = st.number_input("Dense 通道 K", 1, 200, 10)
        sparse_k = st.number_input("BM25 通道 K", 1, 200, 10)
        rrf_k = st.number_input("RRF 常数", 1, 500, 60)
        max_subqueries = st.number_input("最多子问题", 2, 5, 3)
        expansions = st.number_input("最多扩展查询", 1, 4, 2)
        parent_budget = st.number_input("Parent 预算（保守 UTF-8 字节上限）", 0, 2048, 512)
    return dict(retrieval_mode="hybrid" if dense and sparse else "sparse" if sparse else "dense" if dense else "graph",
                graph_enabled=graph, graph_max_hops=hops,
                dense_enabled=dense, bm25_enabled=sparse, rrf_enabled=rrf, rrf_k=rrf_k,
                dense_candidate_k=dense_k, bm25_candidate_k=sparse_k, rerank_enabled=rerank,
                threshold_enabled=gate, query_rewrite_enabled=rewrite, query_decomposition_enabled=decomposition,
                query_expansion_enabled=expansion, hyde_enabled=hyde, parent_child_enabled=parent,
                parent_context_budget=parent_budget, max_subqueries=max_subqueries, expansion_query_count=expansions)


def render_debug(base, token):
    st.title("检索调试")
    with st.expander("可靠性分析"):
        with st.form("reliable_debug"):
            reliable_question = st.text_area("可靠性问题")
            reliable_history = st.text_area("可靠性聊天历史 JSON", "[]")
            reliability_options = st.text_area("范围、必要事实和条件 JSON", "{}")
            validator = st.selectbox("验证方式", ["rules", "semantic"])
            run_reliable = st.form_submit_button("执行可靠性分析")
        if run_reliable:
            try:
                response = requests.post(f"{base}/v2/retrieval/reliable", headers={"X-RAG-V2-Token": token},
                    json={"question": reliable_question, "chat_history": json.loads(reliable_history),
                          "reliability": json.loads(reliability_options), "validator": validator}, timeout=30)
                response.raise_for_status()
                result = response.json()
                st.markdown(f"**{result['answer_action']}**")
                st.write(result["final_answer"])
                st.json(result)
            except (ValueError, requests.RequestException) as exc:
                st.error(str(exc))
    with st.form("retrieval_debug"):
        question = st.text_area("原始问题")
        history_text = st.text_area("聊天历史（JSON）", "[]",
                                   help='例如 [{"role":"user","content":"退款有什么条件？"}]')
        mode = st.selectbox("调试链路", ["retrieval", "answer"])
        candidate = st.number_input("融合候选 K", 1, 200, 10)
        final = st.number_input("最终证据 K", 1, 50, 3)
        threshold = st.number_input("重排门槛", value=0.2)
        strategy = strategy_controls()
        submitted = st.form_submit_button("执行真实调试")
    if submitted:
        try:
            history = json.loads(history_text)
            response = requests.post(f"{base}/v2/retrieval/debug", headers={"X-RAG-V2-Token": token}, timeout=120,
                json={"question": question, "chat_history": history, "mode": mode, "retrieval": strategy,
                      "candidate_k": candidate, "final_k": final, "rerank_threshold": threshold})
            response.raise_for_status()
            st.session_state["v2_debug_result"] = response.json()
        except (ValueError, requests.RequestException) as exc:
            st.session_state.pop("v2_debug_result", None)
            st.error(exc.response.text if isinstance(exc, requests.HTTPError) else str(exc))
    value = st.session_state.get("v2_debug_result")
    if not value:
        return
    st.caption(f"快照 {value['snapshot_id']} · 发布代次 {value['publish_generation']}")
    trace = value["trace"]
    retrieval = trace.get("retrieval", {})
    for label, data in (
        ("原问、历史与查询理解", {"history": value["history"], "query": retrieval.get("query")}),
        ("Dense、BM25 与通道排名", retrieval.get("routes")),
        ("RRF 与 CrossEncoder", {"rrf": retrieval.get("rrf"), "ranking": trace.get("ranked_candidates"),
                                "threshold": retrieval.get("threshold")}),
        ("Parent 与最终证据", {"parent": retrieval.get("parent_child"), "gaps": retrieval.get("subquery_gaps"),
                               "final": value["result"].get("evidence", value["result"].get("sources"))}),
        ("回答、耗时、Token 与降级", {"result": value["result"], "usage": trace.get("usage"),
                                   "hyde": retrieval.get("query", {}).get("hyde"), "degraded": retrieval.get("degraded")}),
    ):
        with st.expander(label):
            st.json(data)
