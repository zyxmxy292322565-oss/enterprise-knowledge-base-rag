"""Evaluation Center views for the real management API."""
import json

import pandas as pd
import requests
import streamlit as st
from retrieval_ui import strategy_controls


def redact_run_export(value, scope=None):
    if isinstance(value, list):
        return [redact_run_export(item, scope) for item in value]
    if not isinstance(value, dict):
        return value
    excluded = {"chat_history", "prompt", "effective_question", "retrieval_text"}
    if scope == "hyde":
        excluded.add("text")
    if scope == "rewrite":
        excluded.add("query")
    return {key: redact_run_export(data, key) for key, data in value.items() if key not in excluded}


def render_evaluation_center(base_url, token):
    st.title("评测中台")
    pages = ["评测总览", "数据集管理", "Gold Evidence 审核", "实验配置与运行", "实验结果分析", "失败案例分析"]
    page = st.sidebar.radio("评测页面", pages)
    headers = {"X-RAG-V2-Token": token}
    with st.sidebar.expander("审核与封存权限"):
        reviewer = st.text_input("审核凭证", type="password")
        holdout = st.text_input("保留集凭证", type="password")
    if reviewer:
        headers["X-Eval-Reviewer"] = reviewer
    if holdout:
        headers["X-Eval-Holdout"] = holdout

    def request(method, path, **kwargs):
        try:
            response = requests.request(method, f"{base_url}/v2{path}", headers=headers, timeout=45, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            st.error(exc.response.text if exc.response is not None else str(exc))
            return None

    def api(path, method="GET", **kwargs):
        return request(method, "/evaluation" + path, **kwargs)

    def download(label, value, filename):
        st.download_button(label, json.dumps(value, ensure_ascii=False, indent=2), filename,
                           mime="application/json")

    def json_text(label, value, key):
        return st.text_area(label, json.dumps(value, ensure_ascii=False, indent=2), height=260, key=key)

    def submit_json(text, operation):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            st.error(str(exc))
            return
        if operation(value) is not None:
            st.rerun()

    def select_dataset():
        selected = st.selectbox("数据集版本", list(range(len(datasets))),
                                format_func=lambda i: f"{datasets[i]['name']} · {datasets[i]['dataset_version']}")
        if selected is None:
            return None, None
        row = datasets[selected]
        path = f"/datasets/{row['dataset_id']}/{row['dataset_version']}"
        return path, api(path)

    datasets, runs = api("/datasets"), api("/runs")
    if datasets is None or runs is None:
        return
    if page == pages[0]:
        cards = st.columns(5)
        for column, title, number in zip(cards, ("数据集版本", "问题总数", "已审核问题", "证据条数", "运行次数"),
                (len(datasets), sum(d["question_count"] for d in datasets),
                 sum(d["reviewed_questions"] for d in datasets), sum(d["evidence_count"] for d in datasets), len(runs))):
            column.metric(title, number)
        st.dataframe([{key: row[key] for key in ("run_id", "state", "completed", "total", "created_at")} for row in runs])
        if st.button("核对历史检索与回答批次"):
            source = api("/historical")
            if source is not None:
                st.json(source)
        return

    if page == pages[1]:
        if st.button("导入历史 V3 开发集"):
            if api("/datasets/import-v3", "POST"):
                st.rerun()
        with st.expander("创建或导入数据集"):
            upload = st.file_uploader("上传 S1-A JSON", type=["json"])
            template = {"dataset_version": "dev-v1", "name": "新评测集", "split": "dev", "questions": [], "evidence": []}
            if upload:
                try:
                    data = json.loads(upload.getvalue())
                    template.update(data.get("body", data))
                except (ValueError, TypeError):
                    st.warning("无法读取数据集 JSON")
            text = json_text("数据集 JSON", template, "dataset_import")
            if st.button("保存为新数据集"):
                submit_json(text, lambda data: api("/datasets", "POST", json=data))
        st.dataframe(datasets)
        path, row = select_dataset()
        if row:
            body = row["body"]
            st.caption("已冻结" if row["frozen"] else "草稿")
            download("导出数据集", body | {"dataset_version": row["dataset_version"], "split": row["split"]},
                     f"{row['dataset_version']}.json")
            st.dataframe(body["questions"], column_order=("question_id", "question", "question_type", "answerability", "review_status"))
            version = st.text_input("新版本号")
            if st.button("派生版本", disabled=not version) and api(path + "/fork", "POST", params={"new_version": version}):
                st.rerun()
            if not row["frozen"]:
                questions = json_text("新增、编辑或删除问题", body["questions"], path + row["sha256"])
                if st.button("保存问题"):
                    def save_questions(values):
                        keys = {q["question_id"] for q in values}
                        return api(path, "PUT", json={"questions": values, "evidence": [
                            e for e in body["evidence"] if e["question_id"] in keys], "expected_sha256": row["sha256"]})
                    submit_json(questions, save_questions)
                if st.button("冻结此版本") and api(path + "/freeze", "POST"):
                    st.rerun()
        return

    if page == pages[2]:
        path, row = select_dataset()
        snapshots = request("GET", "/index-snapshots")
        if not row or not snapshots:
            return
        body = row["body"]
        questions = {q["question_id"]: q for q in body["questions"]}
        qid = st.selectbox("问题", list(questions))
        if not qid:
            return
        st.json(questions[qid])
        gold = [e for e in body["evidence"] if e["question_id"] == qid]
        snapshot = st.selectbox("候选快照", [s["snapshot_id"] for s in snapshots["snapshots"] if s["build_status"] == "ready"])
        if snapshot and st.button("核对原文并映射到 Child"):
            mapped = api(f"{path}/mappings/{snapshot}", params={"question_id": qid})
            if mapped is not None:
                st.json(mapped)
        st.json(gold, expanded=False)
        if not row["frozen"]:
            text = json_text("新增或编辑该题证据", gold, path + qid + row["sha256"])
            if st.button("保存证据"):
                submit_json(text, lambda values: api(path, "PUT", json={"questions": body["questions"],
                    "evidence": [e for e in body["evidence"] if e["question_id"] != qid] + values,
                    "expected_sha256": row["sha256"]}))
            entity = st.selectbox("审核对象", [qid] + [e["evidence_id"] for e in gold])
            action = st.selectbox("审核操作", ["approve", "dispute", "return", "adjudicate"],
                                  format_func={"approve": "通过", "dispute": "标记争议",
                                               "return": "退回", "adjudicate": "裁决"}.get)
            note = st.text_input("审核意见")
            if st.button("提交审核", disabled=not note) and api(path + "/reviews", "POST", json={
                "collection": "questions" if entity == qid else "evidence", "entity_id": entity,
                "action": action, "note": note,
            }):
                st.rerun()
        if st.button("查看审核与修订历史"):
            st.json(api(path + "/audit") or [])
        return

    def select_runs(multiple=False):
        available = {r["run_id"]: r for r in runs}
        title = lambda identifier: f"{available[identifier]['config']['mode']} · {identifier[:8]} · {available[identifier]['state']}"
        return (st.multiselect("实验", list(available), format_func=title, max_selections=10)
                if multiple else st.selectbox("实验", list(available), format_func=title))

    if page == pages[3]:
        _, row = select_dataset()
        snapshots = request("GET", "/index-snapshots")
        history = request("GET", "/publications")
        if row and snapshots and history:
            metadata = {s["snapshot_id"]: s for s in snapshots["snapshots"]}
            publication = st.selectbox("已发布快照与代次", history,
                format_func=lambda h: f"{h['publish_generation']} · {h['snapshot_id']}")
            with st.form("run_config"):
                mode = st.selectbox("实验链路", ["retrieval", "answer"])
                reliability_enabled = st.checkbox("使用规则可靠性控制器（回答链路）")
                reliability_validator = st.selectbox("可靠性验证方式", ["rules", "semantic"])
                candidate = st.number_input("Candidate K", 1, 200, 10)
                final = st.number_input("Final K", 1, 50, 3)
                strategy = strategy_controls()
                threshold = st.number_input("重排阈值", value=0.2)
                cohort = st.selectbox("审核口径", ["developer", "independent_human"])
                submitted = st.form_submit_button("创建实验", disabled=not row["frozen"])
            if submitted and api("/runs", "POST", json={
                "dataset_id": row["dataset_id"], "dataset_version": row["dataset_version"], "mode": mode,
                "candidate_k": candidate, "final_k": final, "rerank_threshold": threshold,
                "reliability_enabled": reliability_enabled, "reliability_validator": reliability_validator, **strategy,
                "review_cohort": cohort, "index_snapshot_id": publication["snapshot_id"],
                "corpus_snapshot_id": metadata[publication["snapshot_id"]]["corpus_snapshot_id"],
                "publish_generation": publication["publish_generation"],
            }):
                st.rerun()
        identifier = select_runs()
        if identifier:
            selected = next(r for r in runs if r["run_id"] == identifier)
            st.progress(selected["completed"] / selected["total"], text=selected["state"])
            st.json(selected["config"], expanded=False)
            if selected.get("error_message"):
                st.error(selected["error_message"])
            a, b, c = st.columns(3)
            if a.button("启动或续跑") and api(f"/runs/{identifier}/start", "POST"):
                st.rerun()
            if b.button("取消") and api(f"/runs/{identifier}/cancel", "POST"):
                st.rerun()
            if c.button("刷新进度"):
                st.rerun()
        return

    if page == pages[4]:
        ids = select_runs(multiple=True)
        if not ids:
            return
        report = api(f"/runs/{ids[0]}") if len(ids) == 1 else api("/compare", params={"ids": ",".join(ids)})
        if not report:
            return
        summary = report.get("summary", report)
        for mode in ("retrieval", "answer"):
            subset = [r for r in summary["runs"] if r["mode"] == mode]
            if not subset:
                continue
            st.subheader("检索指标" if mode == "retrieval" else "回答指标")
            for run_summary in subset:
                if run_summary.get("reliability"):
                    with st.expander("可靠性汇总 " + run_summary["run_id"][:8]):
                        st.json(run_summary["reliability"])
            metric_rows = [{"run": r["run_id"][:8], "metric": key, **value} for r in subset
                           for key, value in r["metrics"].items()]
            if metric_rows:
                columns = st.columns(min(4, len(metric_rows)))
                for column, entry in zip(columns, metric_rows):
                    value = entry["macro"]
                    column.metric(entry["metric"], "N/A" if value is None else f"{value:.3f}")
                frame = pd.DataFrame(metric_rows)
                st.dataframe(frame)
                values = frame.dropna(subset=["macro"])
                if not values.empty:
                    st.line_chart(values.pivot(index="run", columns="metric", values="macro"))
            st.dataframe([{"run": r["run_id"][:8], "metric": key, **value} for r in subset
                          for key, value in r["engineering"].items()])
            failure = pd.DataFrame([{"run": r["run_id"][:8], **r["failures"]} for r in subset]).set_index("run")
            st.bar_chart(failure)
        st.json(summary.get("paired"), expanded=False)
        download("导出报告", redact_run_export(report), "evaluation-report.json")
        return

    identifier = select_runs()
    if identifier:
        report = api(f"/runs/{identifier}")
        if not report:
            return
        cases = {row["question"]["question_id"]: row for row in report["cases"]}
        only_failed = st.checkbox("仅查看失败或执行异常")
        qid = st.selectbox("逐题执行", [qid for qid, row in cases.items()
                                       if not only_failed or row.get("failures") or row["state"] == "error"])
        if qid:
            item = cases[qid]
            st.json(item, expanded=False)
            labels = {"note": "", "failures": item.get("failures", []), "primary_failure": item.get("primary_failure")}
            if report["run"]["config"]["mode"] == "answer":
                labels.update(item.get("answer_review", {}).get("labels", {
                    "actual_action": None, "facts": {}, "claims": [], "citations": [], "forbidden_violation": None,
                    "target_conclusion": None, "false_answer": None, "action_complete": None,
                }))
            text = json_text("修改失败归因或提交回答人审", labels, identifier + qid)
            if st.button("保存审核"):
                submit_json(text, lambda data: api(f"/runs/{identifier}/cases/{qid}/review", "POST", json=data))
            st.json([e for e in report["audit"] if e["entity_id"] == qid], expanded=False)
