"""Streamlit view for the real V2 management API."""
import uuid

import requests
import streamlit as st


def render_knowledge_center(base_url: str, token: str) -> None:
    st.title("V2 数据中台")
    headers = {"X-RAG-V2-Token": token}

    def request(method, path, **kwargs):
        try:
            response = requests.request(method, f"{base_url}/v2{path}", headers=headers, timeout=120, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(detail)
            return None

    docs = request("GET", "/documents")
    state = request("GET", "/index-snapshots")
    if state is None:
        return
    head = state["head"]
    st.caption(f"活动快照 {head['snapshot_id'] or '尚未发布'} · 发布代次 {head['generation']}")
    documents = docs["documents"] if docs is not None else []
    if docs is not None:
        st.subheader("文档草稿")
        st.dataframe(documents)
        st.subheader("活动文档与 Chunk 数")
        st.dataframe(docs["active_members"])

    names = {row["document_id"]: row for row in documents}
    selected = st.selectbox("文档", [None, *names], format_func=lambda key: names[key]["source"] if key else "新增文档")
    current = names.get(selected)
    if current:
        versions = request("GET", f"/documents/{selected}/versions")
        if versions is not None:
            st.dataframe(versions)
        if st.button("查看当前版本处理记录"):
            details = request("GET", f"/document-versions/{current['current_version_id']}/artifact")
            if details:
                st.json({"parents": len(details["parents"]), "children": len(details["children"]),
                         "removed": details["removed"], "processing_config": details["processing_config"]})

    with st.form("knowledge_document"):
        file = st.file_uploader("TXT 或 PDF", type=["txt", "pdf"])
        version = st.text_input("政策版本", value=(current or {}).get("policy_version") or "v1")
        category = st.text_input("分类", value=(current or {}).get("category") or "general")
        origin = st.text_input("来源", value=(current or {}).get("origin") or "upload")
        submitted = st.form_submit_button("保存文档草稿")
    if submitted:
        if file is None:
            st.warning("请选择文档")
        else:
            data = {"version": version, "category": category, "origin": origin}
            path = "/documents"
            if current:
                data["expected_version"] = current["current_version_id"]
                path = f"/documents/{selected}/versions"
            if request("POST", path, files={"file": (file.name, file.getvalue(), file.type)}, data=data):
                st.rerun()
    if current and current["document_status"] != "deleted":
        confirmed = st.checkbox("确认在下一次发布中删除此文档", key=f"delete-{selected}")
        if st.button("删除草稿文档", disabled=not confirmed):
            if request("DELETE", f"/documents/{selected}", params={"expected_version": current["current_version_id"]}):
                st.rerun()

    st.subheader("索引快照")
    st.dataframe(state["snapshots"])
    package_documents = st.multiselect("知识包文档（不选则包含全部草稿）", list(names),
                                      format_func=lambda key: names[key]["source"])
    if st.button("构建当前草稿快照"):
        with st.spinner("正在构建并校验快照"):
            kwargs = {"json": {"document_ids": package_documents}} if package_documents else {}
            if request("POST", "/index-snapshots", **kwargs):
                st.rerun()
    ready = [row["snapshot_id"] for row in state["snapshots"] if row["build_status"] == "ready"]
    selected_snapshot = st.selectbox("查看快照", ready)
    if selected_snapshot:
        page = st.number_input("Chunk 偏移", min_value=0, step=100, value=0)
        contents = request("GET", f"/index-snapshots/{selected_snapshot}/chunks", params={"offset": page, "limit": 100})
        if contents:
            st.caption(f"Child 总数 {contents['total']}")
            st.dataframe(contents["children"])
            st.dataframe(contents["parents"])
        operation = st.radio("快照操作", ["publish", "rollback"], horizontal=True)
        op_key = f"{selected_snapshot}:{operation}"
        attempts = st.session_state.setdefault("knowledge_publication_attempts", {})
        pending = attempts.get(op_key)
        if pending and pending["expected_generation"] != head["generation"]:
            st.caption("存在待核对的提交结果。可使用原幂等键重试，或放弃该请求后按当前代次重新发起。")
            if st.button("放弃待核对的发布请求"):
                attempts.pop(op_key, None)
                st.rerun()
        if st.button("发布" if operation == "publish" else "回滚到所选快照"):
            payload = attempts.setdefault(op_key, {"expected_generation": head["generation"],
                                                   "idempotency_key": uuid.uuid4().hex})
            if request("POST", f"/index-snapshots/{selected_snapshot}/{operation}", json=payload):
                attempts.pop(op_key, None)
                st.rerun()
    st.subheader("发布与回滚记录")
    history = request("GET", "/publications")
    if history is not None:
        st.dataframe(history)
    st.subheader("发布任务状态")
    tasks = request("GET", "/publication-tasks")
    if tasks is not None:
        st.dataframe(tasks)
