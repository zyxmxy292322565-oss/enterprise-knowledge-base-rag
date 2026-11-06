import os
import json
from uuid import uuid4

import requests
import streamlit as st


def render_graph(base):
    st.title("知识图谱")
    headers = {"X-RAG-V2-Token": os.getenv("RAG_V2_ADMIN_TOKEN", "")}

    def call(path, method="GET", **kwargs):
        try:
            response = requests.request(method, base + "/v3/graph" + path, headers=headers, timeout=45, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            st.error(exc.response.text if exc.response is not None else str(exc))
            return None

    snapshot = st.text_input("快照 ID（留空查看活动快照）")
    data = call("/overview", params={"snapshot_id": snapshot} if snapshot else {})
    if not data:
        return
    st.json(data["active"])
    st.dataframe(data["jobs"])
    st.dataframe(data["review_counts"])
    st.json(data.get("extraction"))
    for field in ("entities", "aliases", "relations", "evidence", "chunks", "reviews", "conflicts"):
        with st.expander(field):
            st.dataframe(data[field])
    mode = st.selectbox("抽取方式", ["rules", "llm"])
    batch = st.number_input("本次处理 Chunk 数", min_value=1, max_value=100, value=4 if mode == "llm" else 100)
    if snapshot and st.button("构建或继续抽取") and call(f"/{snapshot}/build", "POST",
                                                         params={"mode": mode, "batch_size": batch}):
        st.rerun()
    if snapshot and st.button("封存图") and call(f"/{snapshot}/seal", "POST"):
        st.rerun()
    with st.form("kg_review"):
        kind = st.selectbox("对象类型", ["entity", "relation"])
        key = st.text_input("审核对象 ID")
        status = st.selectbox("审核状态", ["APPROVED", "REJECTED", "NEEDS_REVIEW"])
        note = st.text_input("审核意见")
        token = st.text_input("审核凭证", type="password")
        if st.form_submit_button("提交审核") and snapshot:
            headers["X-KG-Reviewer"] = token
            if call(f"/{snapshot}/review", "POST", json={"kind": kind, "id": key, "status": status, "note": note}):
                st.rerun()
    if snapshot:
        with st.expander("实体与政策治理"):
            q = st.text_input("实体候选搜索")
            if q:
                result = call(f"/{snapshot}/candidates", params={"query":q})
                if result is not None:
                    st.json(result)
            credential = st.text_input("治理审核凭证", type="password")
            action = st.selectbox("治理动作", ["merge","undo","policy"])
            operation = st.text_input("操作 ID（撤销必填）")
            sample = {"source_id":"","target_id":"","reason":""} if action=="merge" else {"reason":""} if action=="undo" else {
                "policy_id":"","policy_version":"","policy_entity_id":"","relation_id":"","evidence_id":"",
                "policy_name":"","scope":"","valid_from":"","authority_source":"","reason":""}
            body = st.text_area("治理 JSON", json.dumps(sample, ensure_ascii=False,indent=2), key=f"kg-governance-{action}")
            if st.button("提交治理审核"):
                try:
                    payload = json.loads(body)
                    headers["X-KG-Reviewer"] = credential
                    path = f"/{snapshot}/merges"
                    if action == "merge":
                        payload["operation_id"] = operation or uuid4().hex
                    elif action == "undo":
                        path += f"/{operation}/undo"
                    else:
                        path = f"/{snapshot}/policies"
                    result = call(path,"POST",json=payload)
                    if result is not None:
                        st.json(result)
                except (TypeError,ValueError) as exc:
                    st.error(str(exc))
            day = st.text_input("政策历史日期 YYYY-MM-DD")
            if st.button("查看合并及政策历史"):
                result = call(f"/{snapshot}/governance",params={"as_of":day} if day else {})
                if result is not None:
                    st.json(result)
        with st.form("kg_alias"):
            entity = st.text_input("别名归属实体 ID")
            evidence = st.text_input("别名原文证据 ID")
            alias = st.text_input("确认别名")
            reason = st.text_input("对齐理由")
            credential = st.text_input("对齐审核凭证", type="password")
            if st.form_submit_button("确认别名绑定"):
                headers["X-KG-Reviewer"] = credential
                if call(f"/{snapshot}/aliases", "POST", json={"entity_id": entity, "evidence_id": evidence,
                                                               "alias": alias, "note": reason}):
                    st.rerun()
    question = st.text_input("活动图谱查询")
    hops = st.selectbox("跳数", [1, 2], index=1)
    if st.button("查看路径和原文") and question:
        result = call("/lookup", params={"question": question, "hops": hops})
        if result is not None:
            labels = {n["entity_id"]: n["entity_name"] for n in data["entities"]}
            edges = {}
            for path in result["paths"]:
                for edge in path["relation_path"]:
                    edges[edge["relation_id"]] = edge
            if edges:
                nodes = {e[key] for e in edges.values() for key in ("source_entity_id", "target_entity_id")}
                dot = [f'"{node}" [label={json.dumps(labels.get(node, node[:8]), ensure_ascii=False)}];' for node in nodes]
                dot += [f'"{e["source_entity_id"]}" -> "{e["target_entity_id"]}" [label="{e["relation_type"]}"];'
                        for e in edges.values()]
                st.graphviz_chart("digraph G {" + "".join(dot) + "}")
            st.json(result)
