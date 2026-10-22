import streamlit as st
import requests
import pandas as pd
import altair as alt
import json
import os
from dotenv import load_dotenv
# ===============================
# 基础配置
# ===============================

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RAG 企业级知识库问答系统",
    layout="wide"
)

# ------------------------------
# session_state 初始化
# ------------------------------
session_defaults = {
    "last_rag_question": "",
    "last_rag_answer": "",
    "last_rag_sources": "",

    "last_rag_top_k": 0,
    "last_rag_faiss_k": 0,
    "last_rag_top_n": 0,
    "last_rag_rerank_threshold": 0.2,
    "last_rag_use_rerank": 1,

    "eval_logs": []
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "last_rag_top_k" not in st.session_state:
    st.session_state.last_rag_top_k = 0

if "last_rag_faiss_k" not in st.session_state:
    st.session_state.last_rag_faiss_k = 0

if "last_rag_top_n" not in st.session_state:
    st.session_state.last_rag_top_n = 0

if "last_rag_rerank_threshold" not in st.session_state:
    st.session_state.last_rag_rerank_threshold = 0.2

if "last_rag_use_rerank" not in st.session_state:
    st.session_state.last_rag_use_rerank = 1

st.title("企业知识库 RAG + Agent 智能问答系统")
st.caption("Streamlit 前端 + FastAPI 后端 + FAISS 向量索引 + MySQL 数据库")


# ===============================
# 通用请求函数
# ===============================

def safe_get(endpoint: str, params: dict | None = None):
    """
    调用 FastAPI GET 接口。
    endpoint 示例：/qa_logs、/documents、/stats
    """
    try:
        response = requests.get(
            f"{API_BASE}{endpoint}",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return None, str(e)


def safe_post(path: str, json_data: dict = None, timeout: int = 60):
    """
    安全 POST 请求封装。

    path: 接口路径，例如 /rag/eval/run_all
    json_data: POST 请求体
    timeout: 请求超时时间，单位秒
    """
    try:
        resp = requests.post(
            f"{API_BASE}{path}",
            json=json_data or {},
            timeout=timeout
        )
        return resp.json(), None

    except Exception as e:
        return None, str(e)

def safe_delete(endpoint: str):
    """
    调用 FastAPI DELETE 接口。
    endpoint 示例：/qa_logs/1
    """
    try:
        response = requests.delete(
            f"{API_BASE}{endpoint}",
            timeout=30
        )
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return None, str(e)

def safe_put(endpoint: str, json: dict | None = None):
    """
    调用 FastAPI PUT 接口。
    endpoint 示例：/qa_logs/1/note
    """
    try:
        response = requests.put(
            f"{API_BASE}{endpoint}",
            json=json,
            timeout=30
        )
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return None, str(e)

def safe_parse_json(text):
    """
    安全解析 JSON 字符串。
    如果解析失败，就返回原始文本。
    """
    if not text:
        return {}

    if isinstance(text, dict):
        return text

    try:
        return json.loads(text)
    except Exception:
        return text

def format_sources_for_eval(sources):
    if not sources:
        return ""

    formatted_sources = []

    for source in sources:
        if isinstance(source, dict):
            source_file = source.get("source") or source.get("来源文件") or source.get("file_name") or "未知文件"
            chunk_id = source.get("chunk_id") or source.get("文本块编号") or "未知chunk"
            rerank_score = source.get("rerank_score") or source.get("Rerank分数") or ""

            if rerank_score != "":
                formatted_sources.append(
                    f"{source_file} | chunk_id: {chunk_id} | rerank_score: {rerank_score}"
                )
            else:
                formatted_sources.append(
                    f"{source_file} | chunk_id: {chunk_id}"
                )
        else:
            formatted_sources.append(str(source))

    return "\n".join(formatted_sources)

# ===============================
# 1. 知识库状态
# ===============================

st.header("1. 知识库状态")

status_data, status_error = safe_get("/rag_status")

if status_error:
    st.error(f"无法连接 FastAPI 后端：{status_error}")
    st.info("请确认已经启动后端：uvicorn api_app:app")
else:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("文本块数量", status_data.get("doc_count", 0))
    col2.metric("索引数量", status_data.get("index_count", 0))
    col3.metric("已处理文件数", status_data.get("processed_file_count", 0))
    col4.metric("向量状态", "已生成" if status_data.get("has_vectors") else "未生成")

    if status_data.get("has_index"):
        st.success("知识库索引已建立，可以开始提问。")
    else:
        st.warning("知识库索引尚未建立，请先上传文档。")

    with st.expander("查看模型与缓存状态"):
        st.write("Embedding 模型：", "已加载" if status_data.get("has_embed_model") else "未加载")
        st.write("Rerank 模型：", "已加载" if status_data.get("has_rerank_model") else "未加载")
        st.write("LLM 大模型：", "已加载" if status_data.get("has_chat_model") else "未加载")
        st.write("向量缓存文件：", status_data.get("vectors_cache_file"))
        st.write("文本块缓存文件：", status_data.get("all_docs_cache_file"))
        st.write("FAISS 索引缓存文件：", status_data.get("index_cache_file"))

st.divider()

st.subheader("Redis 缓存监控")

try:
    stats_resp = requests.get(
        f"{API_BASE}/redis/rag_cache/stats",
        timeout=10
    )

    stats_data = stats_resp.json()

    if stats_data.get("success"):
        cache_stats = stats_data.get("data", {})

        hit = cache_stats.get("hit", 0)
        miss = cache_stats.get("miss", 0)
        total = cache_stats.get("total", 0)
        hit_rate = cache_stats.get("hit_rate", 0)
        cache_count = cache_stats.get("cache_count", 0)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("缓存命中次数", hit)

        with col2:
            st.metric("缓存未命中次数", miss)

        with col3:
            st.metric("缓存总请求数", total)

        with col4:
            st.metric("当前缓存数量", cache_count)

        st.progress(min(hit_rate, 1.0))
        st.caption(f"Redis 缓存命中率：{hit_rate:.2%}")

    else:
        st.warning(stats_data.get("message", "Redis 缓存统计获取失败"))

except Exception as e:
    st.error(f"Redis 缓存统计接口请求失败：{e}")

if st.button("清空 Redis RAG 问答缓存"):
    try:
        clear_resp = requests.delete(
            f"{API_BASE}/redis/rag_cache",
            timeout=10
        )

        clear_data = clear_resp.json()

        if clear_data.get("success"):
            st.success(
                f"缓存清理成功，删除 {clear_data.get('deleted_count', 0)} 条缓存"
            )
            st.rerun()
        else:
            st.error(clear_data.get("message", "缓存清理失败"))

    except Exception as e:
        st.error(f"清理缓存请求失败：{e}")

if st.button("清空 Redis 缓存统计计数"):
    try:
        clear_stats_resp = requests.delete(
            f"{API_BASE}/redis/rag_cache/stats",
            timeout=10
        )

        clear_stats_data = clear_stats_resp.json()

        if clear_stats_data.get("success"):
            st.success(
                f"缓存统计清理成功，删除 {clear_stats_data.get('deleted_count', 0)} 个统计 key"
            )
            st.rerun()
        else:
            st.error(clear_stats_data.get("message", "缓存统计清理失败"))

    except Exception as e:
        st.error(f"清理缓存统计请求失败：{e}")

if st.button("一键清空 Redis RAG 缓存和统计"):
    try:
        clear_all_resp = requests.delete(
            f"{API_BASE}/redis/rag_cache/all",
            timeout=10
        )

        clear_all_data = clear_all_resp.json()

        if clear_all_data.get("success"):
            data = clear_all_data.get("data", {})
            st.success(
                f"清理完成：问答缓存 {data.get('answer_deleted_count', 0)} 条，统计 key {data.get('stats_deleted_count', 0)} 个"
            )
            st.rerun()
        else:
            st.error(clear_all_data.get("message", "清理失败"))

    except Exception as e:
        st.error(f"一键清理请求失败：{e}")

st.subheader("系统健康检查")

if st.button("检查系统健康状态"):
    health_data, health_error = safe_get("/system/health")

    if health_error:
        st.error(f"系统健康检查失败：{health_error}")
    else:
        status = health_data.get("status", "unknown")
        message = health_data.get("message", "")

        if status == "healthy":
            st.success(message)
        else:
            st.warning(message)

        services = health_data.get("services", {})

        mysql_status = services.get("mysql", {})
        redis_status = services.get("redis", {})
        rag_status = services.get("rag", {})

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "MySQL",
                "正常" if mysql_status.get("ok") else "异常"
            )
            st.caption(mysql_status.get("message", ""))

        with col2:
            st.metric(
                "Redis",
                "正常" if redis_status.get("ok") else "异常"
            )
            st.caption(redis_status.get("message", ""))

        with col3:
            st.metric(
                "RAG",
                "正常" if rag_status.get("ok") else "异常"
            )
            st.caption(f"文档数：{rag_status.get('doc_count', 0)}，索引数：{rag_status.get('index_count', 0)}")

        st.write("RAG 详细状态")

        rag_col1, rag_col2, rag_col3 = st.columns(3)

        with rag_col1:
            st.metric(
                "Embedding 模型",
                "已加载" if rag_status.get("has_embed_model") else "未加载"
            )

        with rag_col2:
            st.metric(
                "Rerank 模型",
                "已加载" if rag_status.get("has_rerank_model") else "未加载"
            )

        with rag_col3:
            st.metric(
                "LLM 模型",
                "已加载" if rag_status.get("has_chat_model") else "未加载"
            )

        with st.expander("查看系统健康检查完整返回"):
            st.json(health_data)

st.subheader("RAG 性能监控")

if st.button("刷新 RAG 性能统计"):
    perf_stats_data, perf_stats_error = safe_get("/rag/perf/stats")

    if perf_stats_error:
        st.error(f"性能统计查询失败：{perf_stats_error}")
    else:
        stats = perf_stats_data.get("data", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("请求总数", stats.get("total_count", 0))

        with col2:
            st.metric("平均总耗时", f"{stats.get('avg_total_ms', 0) / 1000:.2f} 秒")

        with col3:
            st.metric("平均 LLM 耗时", f"{stats.get('avg_llm_ms', 0) / 1000:.2f} 秒")

        with col4:
            st.metric("缓存命中率", f"{stats.get('cache_hit_rate', 0):.2%}")

        with st.expander("查看完整性能统计"):
            st.json(stats)

if st.button("查看最近 RAG 请求"):
    recent_data, recent_error = safe_get("/rag/perf/recent?limit=20")

    if recent_error:
        st.error(f"最近请求查询失败：{recent_error}")
    else:
        recent_logs = recent_data.get("data", [])

        if recent_logs:
            st.dataframe(pd.DataFrame(recent_logs), use_container_width=True)
        else:
            st.info("暂无请求日志。")

if st.button("查看最慢 RAG 请求 Top10"):
    slow_data, slow_error = safe_get("/rag/perf/slow?limit=10")

    if slow_error:
        st.error(f"慢请求查询失败：{slow_error}")
    else:
        slow_logs = slow_data.get("data", [])

        if slow_logs:
            st.dataframe(pd.DataFrame(slow_logs), use_container_width=True)
        else:
            st.info("暂无慢请求日志。")

st.subheader("RAG 评测集")

st.caption(
    "质量评测固定使用缓存旁路：不读取或写入回答缓存，"
    "也不计入正常缓存命中/未命中统计。"
)

with st.expander("新增评测用例"):
    eval_question = st.text_area(
        "标准问题",
        placeholder="例如：Redis 在本项目中主要用来做什么？"
    )

    expected_answer = st.text_area(
        "标准答案",
        placeholder="例如：Redis 主要用于缓存高频问答结果，减少重复大模型调用，提高响应速度。"
    )

    expected_keywords = st.text_input(
        "期望关键词，用逗号分隔",
        placeholder="例如：Redis,缓存,问答结果,大模型,响应速度"
    )

    if st.button("保存评测用例"):
        data, error = safe_post(
            "/rag/eval/cases",
            {
                "question": eval_question,
                "expected_answer": expected_answer,
                "expected_keywords": expected_keywords
            }
        )

        if error:
            st.error(f"保存评测用例失败：{error}")
        else:
            if data.get("success"):
                st.success(data.get("message", "保存成功"))
            else:
                st.error(data.get("message", "保存失败"))

st.write("评测统计")

eval_stats_data, eval_stats_error = safe_get("/rag/eval/stats")

if eval_stats_error:
    st.warning(f"评测统计获取失败：{eval_stats_error}")
else:
    stats = eval_stats_data.get("data", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("评测次数", stats.get("total_count", 0))

    with col2:
        st.metric("平均关键词得分", f"{stats.get('avg_keyword_score', 0):.2%}")

    with col3:
        st.metric("平均总耗时", f"{stats.get('avg_total_ms', 0) / 1000:.2f} 秒")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("普通缓存命中率", f"{stats.get('cache_hit_rate', 0):.2%}")

    with col5:
        st.metric("质量评测旁路次数", stats.get("cache_bypass_count", 0))

    with col6:
        st.metric("历史缓存状态未知", stats.get("cache_unknown_count", 0))

    st.caption(
        "普通缓存命中率只使用明确的 hit 与 miss 作为分母；"
        "bypass 和历史 unknown 不进入该比率。"
    )

if st.button("刷新评测用例"):
    cases_data, cases_error = safe_get("/rag/eval/cases")

    if cases_error:
        st.error(f"查询评测用例失败：{cases_error}")
    else:
        st.session_state.rag_eval_cases = cases_data.get("data", [])

if "rag_eval_cases" not in st.session_state:
    st.session_state.rag_eval_cases = []

cases = st.session_state.rag_eval_cases

if cases:
    st.write("当前评测用例")

    st.dataframe(pd.DataFrame(cases), use_container_width=True)

    for case in cases:
        case_id = case.get("ID")
        question = case.get("问题")
        keywords = case.get("期望关键词")

        with st.expander(f"评测用例 #{case_id}：{question}"):
            st.write(f"问题：{question}")
            st.write(f"期望关键词：{keywords}")

            eval_top_k = st.number_input(
                f"top_k - 用例 {case_id}",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                key=f"eval_top_k_{case_id}"
            )

            col_run, col_delete = st.columns(2)

            with col_run:
                if st.button("运行该评测", key=f"run_eval_{case_id}"):
                    run_data, run_error = safe_post(
                        f"/rag/eval/run/{case_id}",
                        {"top_k": eval_top_k},
                        timeout=120
                    )

                    if run_error:
                        st.error(f"运行评测失败：{run_error}")
                    else:
                        if run_data.get("success"):
                            st.success(run_data.get("message", "评测成功"))

                            eval_result = run_data.get("eval_result", {})
                            ask_result = run_data.get("ask_result", {})

                            st.metric(
                                "关键词得分",
                                f"{eval_result.get('keyword_score', 0):.2%}"
                            )

                            st.write("命中关键词：")
                            st.write(eval_result.get("hit_keywords", []))

                            st.write("系统回答：")
                            st.write(ask_result.get("answer", ""))

                            st.write(
                                "缓存状态：",
                                run_data.get("cache_status", "未知")
                            )

                            st.write(
                                "追踪标识：",
                                {
                                    "experiment_id": run_data.get("experiment_id"),
                                    "run_id": run_data.get("run_id"),
                                    "case_run_id": run_data.get("case_run_id"),
                                }
                            )

                            with st.expander("查看实际生效参数与检索摘要"):
                                st.json({
                                    "effective_parameters": run_data.get("effective_parameters"),
                                    "trace_summary": run_data.get("trace_summary"),
                                })

                            with st.expander("查看完整评测返回"):
                                st.json(run_data)
                        else:
                            st.error(run_data.get("message", "评测失败"))

            with col_delete:
                confirm_delete = st.checkbox(
                    f"确认删除用例 {case_id}",
                    key=f"confirm_delete_eval_{case_id}"
                )

                if st.button("删除该评测用例", key=f"delete_eval_{case_id}"):
                    if not confirm_delete:
                        st.warning("请先勾选确认删除。")
                    else:
                        delete_data, delete_error = safe_delete(
                            f"/rag/eval/cases/{case_id}"
                        )

                        if delete_error:
                            st.error(f"删除失败：{delete_error}")
                        else:
                            if delete_data.get("success"):
                                st.success(delete_data.get("message", "删除成功"))
                            else:
                                st.warning(delete_data.get("message", "删除失败"))

                            st.rerun()
else:
    st.info("暂无评测用例，请先新增或点击刷新。")

st.write("最近评测结果")

if st.button("刷新最近评测结果"):
    results_data, results_error = safe_get("/rag/eval/results?limit=50")

    if results_error:
        st.error(f"查询评测结果失败：{results_error}")
    else:
        results = results_data.get("data", [])

        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("暂无评测结果。")

st.subheader("RAG 批量评测与结果分析")

if "last_run_all_data" not in st.session_state:
    st.session_state.last_run_all_data = None

experiment_name = st.text_input(
    "实验名称",
    value="默认参数实验",
    placeholder="例如：top_k3_faiss2_rerank0.2"
)

experiment_goal = st.text_input(
    "实验目标",
    value="比较指定 RAG 参数对回答质量的影响",
    key="batch_experiment_goal"
)

comparison_variable = st.text_input(
    "本次主要对比变量",
    value="top_k",
    key="batch_comparison_variable"
)

dataset_version = st.text_input(
    "数据集版本",
    value="unversioned",
    help="正式参数实验前应替换为经过审核的数据集版本。",
    key="batch_dataset_version"
)

reuse_experiment_id = st.number_input(
    "复用已有 Experiment ID（0 表示新建）",
    min_value=0,
    value=0,
    step=1,
    key="batch_reuse_experiment_id"
)

batch_top_k = st.number_input(
    "批量评测 top_k",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    key="batch_eval_top_k"
)

with st.expander("批量评测高级 RAG 参数"):
    batch_faiss_multiplier = st.number_input(
        "批量评测 FAISS 初筛倍数",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        key="batch_faiss_multiplier"
    )

    batch_min_rerank_score = st.slider(
        "批量评测 Rerank 最低分数阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        key="batch_min_rerank_score"
    )

    batch_max_context_chars = st.number_input(
        "批量评测最大上下文长度",
        min_value=1000,
        max_value=12000,
        value=6000,
        step=500,
        key="batch_max_context_chars"
    )

    batch_max_chunk_chars = st.number_input(
        "批量评测单 chunk 最大截取长度",
        min_value=300,
        max_value=3000,
        value=1200,
        step=100,
        key="batch_max_chunk_chars"
    )

    batch_low_score_threshold = st.slider(
        "低分统计阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        key="batch_low_score_threshold"
    )

if st.button("批量运行所有评测用例"):
    run_all_data, run_all_error = safe_post(
        "/rag/eval/run_all",
        {
            "experiment_name": experiment_name,
            "experiment_id": int(reuse_experiment_id) if reuse_experiment_id else None,
            "experiment_goal": experiment_goal,
            "comparison_variable": comparison_variable,
            "dataset_version": dataset_version,
            "top_k": batch_top_k,
            "faiss_multiplier": batch_faiss_multiplier,
            "min_rerank_score": batch_min_rerank_score,
            "max_context_chars": batch_max_context_chars,
            "max_chunk_chars": batch_max_chunk_chars,
            "low_score_threshold": batch_low_score_threshold
        },
        timeout=300
    )

    if run_all_error:
        st.error(f"批量评测失败：{run_all_error}")
    else:
        st.session_state.last_run_all_data = run_all_data


run_all_data = st.session_state.last_run_all_data

if run_all_data:
    if run_all_data.get("success"):
        st.success(run_all_data.get("message", "批量评测完成"))

        experiment_summary = run_all_data.get("experiment_summary", {})

        if experiment_summary:
            st.write("本次实验汇总")

            st.code(
                f"Experiment ID: {experiment_summary.get('experiment_id')}\n"
                f"Run ID: {experiment_summary.get('run_id')}\n"
                f"Run 状态: {experiment_summary.get('run_status')}"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "平均关键词得分",
                    f"{experiment_summary.get('avg_keyword_score', 0):.2%}"
                )

            with col2:
                st.metric(
                    "平均总耗时",
                    f"{experiment_summary.get('avg_total_ms', 0) / 1000:.2f} 秒"
                )

            with col3:
                st.metric(
                    "低分数量",
                    experiment_summary.get("low_score_count", 0)
                )

            with col4:
                st.metric(
                    "缓存模式",
                    "旁路" if experiment_summary.get("cache_mode") == "bypass" else "普通"
                )

            with st.expander("查看实验参数与汇总"):
                st.json(experiment_summary)

        results = run_all_data.get("results", [])

        if results:
            st.write("本次批量评测明细")
            st.dataframe(pd.DataFrame(results), use_container_width=True)

            case_run_ids = [
                item.get("case_run_id") for item in results if item.get("case_run_id")
            ]
            if case_run_ids:
                selected_case_run_id = st.selectbox(
                    "选择 Case Run 查看详细 Trace",
                    case_run_ids,
                    key="batch_trace_case_run_id"
                )
                if st.button("加载 Case Run Trace", key="load_case_run_trace"):
                    trace_data, trace_error = safe_get(
                        f"/rag/eval/case_runs/{selected_case_run_id}?include_artifacts=true"
                    )
                    if trace_error:
                        st.error(f"加载 Trace 失败：{trace_error}")
                    elif trace_data.get("success"):
                        detail = trace_data.get("data", {})
                        st.json({
                            key: value for key, value in detail.items()
                            if key != "artifacts"
                        })
                        with st.expander("查看最终 Context 与 Prompt"):
                            artifacts = detail.get("artifacts", {})
                            st.write("Context SHA-256：", artifacts.get("context_sha256"))
                            st.text_area(
                                "最终 Context",
                                value=artifacts.get("context_text") or "",
                                height=240,
                                disabled=True,
                            )
                            st.write("Prompt SHA-256：", artifacts.get("prompt_sha256"))
                            st.text_area(
                                "最终 Prompt",
                                value=artifacts.get("prompt_text") or "",
                                height=240,
                                disabled=True,
                            )
                    else:
                        st.warning(trace_data.get("message", "未找到 Trace"))

        with st.expander("查看批量评测完整返回"):
            st.json(run_all_data)

    else:
        st.warning(run_all_data.get("message", "批量评测失败"))
        failed_summary = run_all_data.get("experiment_summary", {})
        if failed_summary:
            st.write("本次 Run 状态")
            st.json(failed_summary)
        failed_results = run_all_data.get("results", [])
        if failed_results:
            st.write("已保存的逐题运行明细")
            st.dataframe(pd.DataFrame(failed_results), use_container_width=True)
else:
    st.info("还没有运行批量评测。")

st.write("最近实验 Run")
if st.button("刷新最近 Run", key="refresh_rag_eval_runs"):
    runs_data, runs_error = safe_get("/rag/eval/runs?limit=50")
    if runs_error:
        st.error(f"查询 Run 失败：{runs_error}")
    elif runs_data.get("success"):
        recent_runs = runs_data.get("data", [])
        if recent_runs:
            st.dataframe(pd.DataFrame(recent_runs), use_container_width=True)
        else:
            st.info("暂无 v2 Run 记录。")
    else:
        st.warning(runs_data.get("message", "Run 查询失败"))

st.write("评测用例平均得分分析")

if st.button("刷新用例得分统计"):
    case_stats_data, case_stats_error = safe_get("/rag/eval/case_stats")

    if case_stats_error:
        st.error(f"用例得分统计查询失败：{case_stats_error}")
    else:
        case_stats = case_stats_data.get("data", [])

        if case_stats:
            df_case_stats = pd.DataFrame(case_stats)

            st.dataframe(df_case_stats, use_container_width=True)

            chart_df = df_case_stats[["问题", "平均关键词得分"]].copy()
            chart_df = chart_df.set_index("问题")

            st.bar_chart(chart_df)
        else:
            st.info("暂无用例得分统计。")


st.write("低分评测结果")

low_score_threshold = st.slider(
    "低分阈值",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05
)

if st.button("查看低分评测结果"):
    low_score_data, low_score_error = safe_get(
        f"/rag/eval/low_score?threshold={low_score_threshold}&limit=20"
    )

    if low_score_error:
        st.error(f"低分评测结果查询失败：{low_score_error}")
    else:
        low_score_results = low_score_data.get("data", [])

        if low_score_results:
            st.warning(f"发现 {len(low_score_results)} 条低分评测结果")
            st.dataframe(pd.DataFrame(low_score_results), use_container_width=True)
        else:
            st.success("暂无低分评测结果。")

experiment_summary = {}

if run_all_data:
    experiment_summary = run_all_data.get("experiment_summary", {})

if experiment_summary:
    st.write("本次实验汇总")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "平均关键词得分",
            f"{experiment_summary.get('avg_keyword_score', 0):.2%}"
        )

    with col2:
        st.metric(
            "平均总耗时",
            f"{experiment_summary.get('avg_total_ms', 0) / 1000:.2f} 秒"
        )

    with col3:
        st.metric(
            "低分数量",
            experiment_summary.get("low_score_count", 0)
        )

    with col4:
        st.metric(
            "缓存模式",
            "旁路" if experiment_summary.get("cache_mode") == "bypass" else "普通"
        )

    with st.expander("查看实验参数与汇总"):
        st.json(experiment_summary)

# ===============================
# 2. 上传文档
# ===============================

st.header("2. 上传文档")

uploaded_file = st.file_uploader(
    "上传知识库文档",
    type=["txt", "pdf"]
)

if uploaded_file is not None:
    st.write("当前选择文件：", uploaded_file.name)
    st.write("文件类型：", uploaded_file.type)
    st.write("文件大小：", f"{uploaded_file.size / 1024:.2f} KB")

    if st.button("上传并构建知识库"):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream"
                )
            }

            with st.spinner("正在上传、解析、切块、向量化并更新索引..."):
                resp = requests.post(
                    f"{API_BASE}/upload",
                    files=files,
                    timeout=120
                )

            upload_data = resp.json()

            # 后端明确返回 success=True
            if upload_data.get("success"):
                if upload_data.get("is_duplicate"):
                    st.warning(upload_data.get("message", "文件已存在，跳过处理"))
                else:
                    st.success(upload_data.get("message", "上传成功"))

                if "deleted_cache_count" in upload_data:
                    st.info(
                        f"已清理 Redis 旧问答缓存：{upload_data.get('deleted_cache_count', 0)} 条"
                    )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("文本长度", upload_data.get("text_length", 0))

                with col2:
                    st.metric("新增文本块", upload_data.get("chunk_count", 0))

                with col3:
                    st.metric("总文本块", upload_data.get("total_doc_count", 0))

                with col4:
                    st.metric("索引数量", upload_data.get("index_count", 0))

                with st.expander("查看文档清洗与切分信息"):
                    st.write("原始文本长度：", upload_data.get("raw_text_length", 0))
                    st.write("清洗后文本长度：", upload_data.get("cleaned_text_length", 0))
                    st.write("清洗掉字符数：", upload_data.get("removed_chars", 0))
                    st.write("清洗比例：", f"{upload_data.get('clean_rate', 0):.2%}")
                    st.write("切分策略：", upload_data.get("split_strategy", "未知"))
                    st.write("缓存版本：", upload_data.get("cache_version", "未知"))

                if upload_data.get("file_ext"):
                    st.caption(f"文件后缀：{upload_data.get('file_ext')}")

                if upload_data.get("file_hash"):
                    st.caption(f"文件 Hash：{upload_data.get('file_hash')}")

            # 后端返回 success=False
            else:
                st.error(upload_data.get("message", "上传失败"))

                if upload_data.get("file_ext"):
                    st.warning(f"当前文件后缀：{upload_data.get('file_ext')}")

                if upload_data.get("file_size") is not None:
                    st.warning(
                        f"当前文件大小：{upload_data.get('file_size', 0) / 1024:.2f} KB"
                    )

            with st.expander("查看上传返回详情"):
                st.json(upload_data)

            st.info("上传完成后，如果顶部知识库状态没有立即变化，可以刷新页面状态。")

            if st.button("刷新页面状态"):
                st.rerun()

        except Exception as e:
            st.error(f"上传请求失败：{e}")

st.divider()


# ===============================
# 3. 提问问答
# ===============================

st.header("3. 知识库问答")

question = st.text_area(
    "请输入你的问题",
    placeholder="例如：这个文档主要讲了什么？"
)

st.subheader("RAG 参数设置")

top_k = st.number_input(
    "最终使用文本块数量 top_k",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

with st.expander("高级 RAG 参数"):
    faiss_multiplier = st.number_input(
        "FAISS 初筛倍数 faiss_multiplier",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="例如 top_k=3，faiss_multiplier=2，则 FAISS 先检索 6 个候选，再交给 Rerank 精排。"
    )

    min_rerank_score = st.slider(
        "Rerank 最低分数阈值 min_rerank_score",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="分数低于该阈值时，系统会认为没有足够相关内容。"
    )

    max_context_chars = st.number_input(
        "最大上下文长度 max_context_chars",
        min_value=1000,
        max_value=12000,
        value=6000,
        step=500
    )

    max_chunk_chars = st.number_input(
        "单个文本块最大截取长度 max_chunk_chars",
        min_value=300,
        max_value=3000,
        value=1200,
        step=100
    )

if st.button("开始提问"):
    if not question.strip():
        st.warning("请先输入问题。")
    else:
        payload = {
            "question": question.strip(),
            "top_k": int(top_k)
        }

        with st.spinner("正在检索知识库、Rerank 精排并调用大模型..."):
            ask_data, ask_error = safe_post(
                "/ask",
                {
                    "question": question,
                    "top_k": top_k,
                    "faiss_multiplier": faiss_multiplier,
                    "min_rerank_score": min_rerank_score,
                    "max_context_chars": max_context_chars,
                    "max_chunk_chars": max_chunk_chars
                },
                timeout=120
            )

        if ask_error:
            st.error(f"问答失败：{ask_error}")
        else:
            st.subheader("回答")
            st.write(ask_data.get("answer", "无答案"))

            security = ask_data.get("security", {})

            if security:
                warning_count = security.get("prompt_injection_warning_count", 0)

                if warning_count > 0:
                    st.warning(f"检测到 {warning_count} 个疑似 Prompt Injection 风险片段")

                    with st.expander("查看 Prompt Injection 风险详情"):
                        st.json(security.get("warnings", []))
                else:
                    st.success("未检测到明显 Prompt Injection 风险")

            rag_params = ask_data.get("rag_params", {})

            if rag_params:
                with st.expander("查看本次 RAG 参数"):
                    st.json(rag_params)

            timing = ask_data.get("timing", {})

            if timing:
                st.subheader("本次问答耗时分析")

                total_ms = timing.get("total_ms", 0)
                cache_read_ms = timing.get("cache_read_ms", 0)
                embedding_ms = timing.get("embedding_ms", 0)
                faiss_search_ms = timing.get("faiss_search_ms", 0)
                rerank_ms = timing.get("rerank_ms", 0)
                llm_ms = timing.get("llm_ms", 0)
                redis_write_ms = timing.get("redis_write_ms", 0)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("总耗时", f"{total_ms / 1000:.2f} 秒")
                    st.metric("Redis 读缓存", f"{cache_read_ms:.2f} ms")

                with col2:
                    st.metric("Embedding", f"{embedding_ms:.2f} ms")
                    st.metric("FAISS 检索", f"{faiss_search_ms:.2f} ms")

                with col3:
                    st.metric("Rerank", f"{rerank_ms:.2f} ms")
                    st.metric("LLM 调用", f"{llm_ms / 1000:.2f} 秒")

                st.caption(f"Redis 写缓存耗时：{redis_write_ms:.2f} ms")

                with st.expander("查看完整 timing 数据"):
                    st.json(timing)

            if ask_data.get("from_cache"):
                st.success("本次回答来自 Redis 缓存")
            else:
                st.info("本次回答为实时 RAG 生成")

            sources = ask_data.get("sources", [])

            if sources:
                st.subheader("参考来源")
                df_sources = pd.DataFrame(sources)
                st.dataframe(df_sources, use_container_width=True)
            else:
                st.info("暂无参考来源。")

            with st.expander("查看问答接口完整返回"):
                st.json(ask_data)

            rag_answer_text = ask_data.get("answer", "")
            rag_sources_text = format_sources_for_eval(ask_data.get("sources", []))

            st.session_state.last_rag_question = question.strip()
            st.session_state.last_rag_answer = rag_answer_text
            st.session_state.last_rag_sources = rag_sources_text

            # 保存本次 RAG 参数
            st.session_state.last_rag_top_k = top_k
            st.session_state.last_rag_faiss_k = 0
            st.session_state.last_rag_top_n = top_k
            st.session_state.last_rag_rerank_threshold = 0.2
            st.session_state.last_rag_use_rerank = 1

st.subheader("将本次 RAG 回答保存为评估记录")

if not st.session_state.last_rag_question:
    st.info("还没有 RAG 问答结果。请先在上方进行一次知识库问答。")
else:
    st.write("**最近一次问题：**")
    st.write(st.session_state.last_rag_question)

    st.write("**最近一次 RAG 回答：**")
    st.write(st.session_state.last_rag_answer)

    with st.expander("查看最近一次来源信息"):
        st.write(st.session_state.last_rag_sources)

    quick_expected_answer = st.text_area(
        "标准答案",
        placeholder="填写你认为这个问题的理想答案",
        key="quick_eval_expected_answer"
    )

    quick_score = st.slider(
        "人工评分",
        min_value=0,
        max_value=5,
        value=3,
        step=1,
        key="quick_eval_score"
    )

    quick_note = st.text_area(
        "评估备注",
        placeholder="例如：回答基本准确，但缺少某个关键点",
        key="quick_eval_note"
    )

    if st.button("保存本次 RAG 回答为评估记录"):
        payload = {
            "question": st.session_state.last_rag_question,
            "expected_answer": quick_expected_answer.strip(),
            "rag_answer": st.session_state.last_rag_answer,
            "sources": st.session_state.last_rag_sources,
            "score": quick_score,
            "note": quick_note.strip(),

            # RAG 参数
            "top_k": st.session_state.last_rag_top_k,
            "faiss_k": st.session_state.last_rag_faiss_k,
            "top_n": st.session_state.last_rag_top_n,
            "rerank_threshold": st.session_state.last_rag_rerank_threshold,
            "use_rerank": st.session_state.last_rag_use_rerank
        }

        st.write("**本次 RAG 参数：**")
        st.json({
            "top_k": st.session_state.last_rag_top_k,
            "faiss_k": st.session_state.last_rag_faiss_k,
            "top_n": st.session_state.last_rag_top_n,
            "rerank_threshold": st.session_state.last_rag_rerank_threshold,
            "use_rerank": st.session_state.last_rag_use_rerank
        })

        save_eval_data, save_eval_error = safe_post("/save_eval", json=payload)

        if save_eval_error:
            st.error(f"保存评估记录失败：{save_eval_error}")
        else:
            if save_eval_data.get("success"):
                st.success(save_eval_data.get("message", "保存成功"))

                # 保存成功后顺便刷新评估历史
                eval_logs_data, eval_logs_error = safe_get(
                    "/eval_logs",
                    params={"limit": st.session_state.get("eval_log_limit", 20)}
                )

                if not eval_logs_error:
                    st.session_state.eval_logs = eval_logs_data.get("data", [])
            else:
                st.warning(save_eval_data.get("message", "保存失败"))


st.divider()

# ===============================
# 3.1 FAISS 检索调试
# ===============================

st.header("3.1 FAISS 检索调试")

search_question = st.text_area(
    "请输入 FAISS 检索问题",
    placeholder="例如：这个文档主要讲了什么？",
    key="search_question"
)

search_top_k = st.number_input(
    "FAISS 返回数量 top_k",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    key="search_top_k"
)

if st.button("开始 FAISS 检索调试"):
    if not search_question.strip():
        st.warning("请先输入检索问题。")
    else:
        payload = {
            "question": search_question.strip(),
            "top_k": int(search_top_k)
        }

        with st.spinner("正在执行 FAISS 向量检索..."):
            search_data, search_error = safe_post(
                "/search",
                json=payload
            )

        st.session_state["faiss_search_data"] = search_data
        st.session_state["faiss_search_error"] = search_error


if st.session_state.get("faiss_search_error"):
    st.error(f"FAISS 检索失败：{st.session_state['faiss_search_error']}")

if st.session_state.get("faiss_search_data"):
    search_data = st.session_state["faiss_search_data"]

    st.success(search_data.get("message", "FAISS 检索完成"))

    results = search_data.get("results", [])

    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("暂无 FAISS 检索结果。")

    with st.expander("查看 FAISS 检索完整返回"):
        st.json(search_data)

if st.button("清空 FAISS 调试结果"):
    st.session_state.pop("faiss_search_data", None)
    st.session_state.pop("faiss_search_error", None)
    st.rerun()
# ===============================
# 3.2 Rerank 检索调试
# ===============================

st.header("3.2 Rerank 检索调试")

debug_question = st.text_area(
    "请输入调试问题",
    placeholder="例如：这个文档主要讲了什么？",
    key="debug_question"
)

debug_col1, debug_col2 = st.columns(2)

with debug_col1:
    faiss_k = st.number_input(
        "FAISS 初筛数量 faiss_k",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )

with debug_col2:
    top_n = st.number_input(
        "Rerank 最终返回数量 top_n",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

if st.button("开始调试 Rerank"):
    if not debug_question.strip():
        st.warning("请先输入调试问题。")
    else:
        payload = {
            "question": debug_question.strip(),
            "faiss_k": int(faiss_k),
            "top_n": int(top_n)
        }

        with st.spinner("正在执行 FAISS 初筛 + Rerank 精排..."):
            rerank_data, rerank_error = safe_post(
                "/search_rerank",
                json=payload
            )

        st.session_state["rerank_data"] = rerank_data
        st.session_state["rerank_error"] = rerank_error

if st.session_state.get("rerank_error"):
    st.error(f"Rerank 调试失败：{st.session_state['rerank_error']}")

if st.session_state.get("rerank_data"):
    rerank_data = st.session_state["rerank_data"]

    st.success(rerank_data.get("message", "Rerank 调试完成"))

    results = rerank_data.get("results", [])

    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("暂无 Rerank 检索结果。")

    with st.expander("查看 Rerank 接口完整返回"):
        st.json(rerank_data)

if st.button("清空 Rerank 调试结果"):
    st.session_state.pop("rerank_data", None)
    st.session_state.pop("rerank_error", None)
    st.rerun()


# ===============================
# 3.3 Agent 智能助手
# ===============================

st.header("3.3 Agent 智能助手")

agent_question = st.text_area(
    "请输入你想让 Agent 处理的问题",
    placeholder="例如：当前知识库有哪些文档？最近 5 条问答记录是什么？哪个文件被引用最多？",
    key="agent_question"
)

if st.button("发送给 Agent"):
    if not agent_question.strip():
        st.warning("请先输入 Agent 问题。")
    else:
        payload = {
            "question": agent_question.strip()
        }

        with st.spinner("Agent 正在判断任务类型并调用工具..."):
            agent_data, agent_error = safe_post(
                "/agent_chat",
                json=payload
            )

        if agent_error:
            st.error(f"Agent 调用失败：{agent_error}")
        else:
            st.subheader("Agent 回答")

            tool_name = agent_data.get("tool_name", "")
            if tool_name:
                st.info(f"Agent 选择的工具：{tool_name}")

            args = agent_data.get("args", {})
            if args:
                st.write("工具参数：")
                st.json(args)

            # 最终回答：给用户看的自然语言回答
            st.write(agent_data.get("answer", "无回答"))

            # 工具原始结果：给开发者调试看的
            with st.expander("查看工具原始结果"):
                st.write(agent_data.get("tool_result", "无工具结果"))

            # 接口完整返回：方便调试
            with st.expander("查看 Agent 接口完整返回"):
                st.json(agent_data)

# ------------------------------
# 3.3.1 LangGraph Agent 智能助手
# ------------------------------
st.header("3.3.1 LangGraph Agent 智能助手")

st.write("新版 Agent 使用 LangGraph 实现，将路由、工具调用、结果总结和日志保存拆分为多个节点。")

graph_agent_question = st.text_area(
    "请输入要发送给 LangGraph Agent 的问题",
    placeholder="例如：当前知识库有哪些文档？",
    key="graph_agent_question"
)

graph_thread_id = st.text_input(
    "LangGraph 会话 ID thread_id",
    value="default",
    key="graph_thread_id"
)

if st.button("发送给 LangGraph Agent"):
    if not graph_agent_question.strip():
        st.warning("问题不能为空。")
    else:
        payload = {
            "question": graph_agent_question.strip(),
            "thread_id": graph_thread_id.strip() or "default"
        }

        graph_agent_data, graph_agent_error = safe_post(
            "/agent_chat_graph",
            json=payload
        )

        if graph_agent_error:
            st.error(f"LangGraph Agent 调用失败：{graph_agent_error}")
        else:
            st.subheader("LangGraph Agent 回答")
            st.caption(f"当前 thread_id：{graph_agent_data.get('thread_id', 'default')}")
            st.write("**最终回答：**")
            st.write(graph_agent_data.get("answer", "无回答"))

            tool_name = graph_agent_data.get("tool_name", "")
            if tool_name:
                st.info(f"LangGraph Agent 选择的工具：{tool_name}")

            args = graph_agent_data.get("args", {})
            if args:
                st.write("**工具参数：**")
                st.json(args)

            log_saved = graph_agent_data.get("log_saved", False)
            error = graph_agent_data.get("error", "")

            if log_saved:
                st.success("Agent 调用日志已保存。")
            else:
                st.warning("Agent 调用日志未保存。")

            if error:
                st.error(f"错误信息：{error}")

            with st.expander("查看当前会话 conversation_history"):
                history = graph_agent_data.get("conversation_history", [])

                if history:
                    for msg in history:
                        role = msg.get("role", "")
                        content = msg.get("content", "")

                        if role == "user":
                            st.write(f"**用户：** {content}")
                        elif role == "assistant":
                            st.write(f"**助手：** {content}")
                        else:
                            st.write(f"**{role}：** {content}")
                else:
                    st.write("当前会话暂无历史。")

            with st.expander("查看工具原始结果"):
                st.write(graph_agent_data.get("tool_result", "无工具结果"))

            with st.expander("查看 Router 调试信息"):
                st.write("**Router 原始输出：**")
                st.code(graph_agent_data.get("router_raw", ""))

                st.write("**Router 校验结果：**")

                router_validated = graph_agent_data.get("router_validated", {})

                if isinstance(router_validated, dict):
                    st.json(router_validated)
                else:
                    router_validated = safe_parse_json(router_validated)

                    if isinstance(router_validated, dict):
                        st.json(router_validated)
                    else:
                        st.write(router_validated)

            with st.expander("查看 LangGraph Agent 接口完整返回"):
                st.json(graph_agent_data)

st.header("3.3.2 LangGraph 多轮记忆管理")

memory_limit = st.number_input(
    "会话列表显示数量",
    min_value=1,
    max_value=100,
    value=20,
    step=1,
    key="memory_thread_limit"
)

if st.button("查询所有 thread_id"):
    threads_data, threads_error = safe_get(
        "/conversation_threads",
        params={"limit": memory_limit}
    )

    if threads_error:
        st.error(f"查询会话列表失败：{threads_error}")
    else:
        threads = threads_data.get("threads", [])

        if not threads:
            st.info("当前还没有会话历史。")
        else:
            st.write(f"共查询到 {len(threads)} 个会话：")
            st.dataframe(threads, use_container_width=True)

query_thread_id = st.text_input(
    "输入要查看的 thread_id",
    value="default",
    key="query_thread_id"
)

history_limit = st.number_input(
    "历史消息数量",
    min_value=1,
    max_value=100,
    value=20,
    step=1,
    key="query_history_limit"
)

if st.button("查询该 thread_id 的历史"):
    history_data, history_error = safe_get(
        f"/conversation_history/{query_thread_id}",
        params={"limit": history_limit}
    )

    if history_error:
        st.error(f"查询会话历史失败：{history_error}")
    else:
        history = history_data.get("history", [])

        st.write(f"thread_id：{history_data.get('thread_id', query_thread_id)}")
        st.write(f"消息数量：{history_data.get('count', 0)}")

        if not history:
            st.info("该 thread_id 暂无历史。")
        else:
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                created_at = msg.get("created_at", "")

                if role == "user":
                    st.markdown(f"**用户：** {content}")
                elif role == "assistant":
                    st.markdown(f"**助手：** {content}")
                else:
                    st.markdown(f"**{role}：** {content}")

                if created_at:
                    st.caption(created_at)

                st.divider()

st.subheader("清空指定会话历史")

clear_thread_id = st.text_input(
    "输入要清空的 thread_id",
    value="",
    key="clear_thread_id"
)

confirm_clear_memory = st.checkbox(
    "我确认要清空这个 thread_id 的历史",
    key="confirm_clear_memory"
)

if st.button("清空该 thread_id 历史"):
    if not clear_thread_id.strip():
        st.warning("thread_id 不能为空。")
    elif not confirm_clear_memory:
        st.warning("请先勾选确认框。")
    else:
        clear_data, clear_error = safe_delete(
            f"/conversation_history/{clear_thread_id.strip()}"
        )

        if clear_error:
            st.error(f"清空会话历史失败：{clear_error}")
        else:
            st.success(
                f"已清空 thread_id={clear_data.get('thread_id')} 的历史，删除 {clear_data.get('deleted_count', 0)} 条消息。"
            )

st.divider()

# ===============================
# 3.4 Agent 调用历史
# ===============================

st.header("3.4 Agent 调用历史")

agent_log_limit = st.number_input(
    "查询 Agent 日志条数",
    min_value=1,
    max_value=100,
    value=20,
    step=1
)

agent_tool_filter = st.selectbox(
    "按工具筛选",
    [
        "全部",
        "get_documents_tool",
        "get_qa_logs_tool",
        "get_stats_tool",
        "rag_qa_tool"
    ]
)

agent_log_keyword = st.text_input(
    "按关键词搜索 Agent 日志",
    placeholder="例如：RAG、文档、最近 3 条、统计"
)

if st.button("查询 Agent 调用历史"):
    params = {
        "limit": agent_log_limit
    }

    if agent_tool_filter != "全部":
        params["tool_name"] = agent_tool_filter

    if agent_log_keyword.strip():
        params["keyword"] = agent_log_keyword.strip()

    with st.spinner("正在查询 Agent 调用历史..."):
        agent_logs_data, agent_logs_error = safe_get(
            "/agent_logs",
            params=params
        )

    if agent_logs_error:
        st.error(f"查询 Agent 调用历史失败：{agent_logs_error}")
    else:
        logs = agent_logs_data.get("data", [])
        count = agent_logs_data.get("count", 0)

        st.success(f"查询成功，共返回 {count} 条记录")

        if not logs:
            st.info("暂无 Agent 调用历史。")
        else:
            df_agent_logs = pd.DataFrame(logs)

            csv_data = df_agent_logs.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="导出当前 Agent 日志为 CSV",
                data=csv_data,
                file_name="agent_logs.csv",
                mime="text/csv"
            )

            for log in logs:
                log_id = log.get("ID", "未知ID")
                question = log.get("问题", "未知问题")
                tool_name = log.get("工具名称", "未知工具")
                created_at = log.get("创建时间", "未知时间")
                answer = log.get("最终回答", "")
                args = log.get("工具参数", "")
                tool_result = log.get("工具原始结果", "")

                note = log.get("备注", "") or ""

                with st.expander(f"#{log_id} | {tool_name} | {question}"):
                    st.write(f"**创建时间：** {created_at}")
                    st.write(f"**用户问题：** {question}")
                    st.write(f"**调用工具：** {tool_name}")

                    st.write("**最终回答：**")
                    st.write(answer)

                    st.write("**工具参数：**")
                    parsed_args = safe_parse_json(args)
                    if isinstance(parsed_args, dict):
                        st.json(parsed_args)
                    else:
                        st.code(str(parsed_args), language="json")

                    st.write("**工具原始结果：**")
                    st.write(tool_result)

                    st.write("**备注：**")
                    new_note = st.text_area(
                        "编辑备注",
                        value=note,
                        key=f"agent_note_{log_id}"
                    )

                    if st.button(
                            "保存备注",
                            key=f"save_agent_note_{log_id}"
                    ):
                        update_data, update_error = safe_put(
                            f"/agent_logs/{log_id}/note",
                            json={"note": new_note}
                        )

                        if update_error:
                            st.error(f"备注保存失败：{update_error}")
                        else:
                            st.success("备注保存成功。")
                            st.rerun()

                    if st.button(
                            "删除这条 Agent 日志",
                            key=f"delete_agent_log_{log_id}"
                    ):
                        delete_data, delete_error = safe_delete(f"/agent_logs/{log_id}")

                        if delete_error:
                            st.error(f"删除失败：{delete_error}")
                        else:
                            st.success("删除成功，请重新查询 Agent 调用历史。")
                            st.rerun()

st.subheader("Agent 日志管理")

confirm_clear_agent_logs = st.checkbox("我确认要清空全部 Agent 日志")

if st.button("清空全部 Agent 日志"):
    if not confirm_clear_agent_logs:
        st.warning("请先勾选确认框。")
    else:
        clear_data, clear_error = safe_delete("/agent_logs")

        if clear_error:
            st.error(f"清空失败：{clear_error}")
        else:
            st.success("所有 Agent 日志已清空。")
            st.rerun()

st.divider()

# ===============================
# 3.5 Agent 调用统计
# ===============================

st.header("3.5 Agent 调用统计")

if st.button("刷新 Agent 调用统计"):
    with st.spinner("正在加载 Agent 调用统计..."):
        agent_stats_data, agent_stats_error = safe_get("/agent_logs/stats")

    if agent_stats_error:
        st.error(f"加载 Agent 调用统计失败：{agent_stats_error}")
    else:
        stats = agent_stats_data.get("data", {})

        tool_name_map = {
            "get_documents_tool": "文档列表工具",
            "get_qa_logs_tool": "问答历史工具",
            "get_stats_tool": "统计信息工具",
            "rag_qa_tool": "知识库问答工具"
        }

        tool_short_name_map = {
            "get_documents_tool": "文档",
            "get_qa_logs_tool": "历史",
            "get_stats_tool": "统计",
            "rag_qa_tool": "问答"
        }

        total_count = stats.get("Agent调用总数", 0)
        latest_time = stats.get("最近调用时间", "暂无")
        most_used_tool = stats.get("最常用工具", "暂无")
        tool_stats = stats.get("工具调用统计", [])

        most_used_tool_display = tool_name_map.get(most_used_tool, most_used_tool)

        col1, col2, col3 = st.columns(3)

        col1.metric("Agent 调用总数", total_count)
        col2.metric("最常用工具", most_used_tool_display)
        col3.metric("最近调用时间", latest_time if latest_time else "暂无")

        if tool_stats:
            st.subheader("各工具调用次数")

            df_tool_stats = pd.DataFrame(tool_stats)

            df_tool_stats["工具显示名"] = df_tool_stats["工具名称"].map(
                lambda x: tool_name_map.get(x, x)
            )

            df_tool_stats["图表显示名"] = df_tool_stats["工具名称"].map(
                lambda x: tool_short_name_map.get(x, x)
            )

            st.dataframe(df_tool_stats, use_container_width=True)

            st.subheader("工具调用次数图表")

            chart_height = max(200, len(df_tool_stats) * 60)

            chart_height = max(200, len(df_tool_stats) * 60)

            chart = (
                alt.Chart(df_tool_stats)
                .mark_bar()
                .encode(
                    x=alt.X("调用次数:Q", title="调用次数"),
                    y=alt.Y(
                        "图表显示名:N",
                        title="工具",
                        sort="-x",
                        axis=alt.Axis(labelAngle=0)
                    ),
                    tooltip=[
                        alt.Tooltip("工具显示名:N", title="工具"),
                        alt.Tooltip("调用次数:Q", title="调用次数")
                    ]
                )
                .properties(
                    height=chart_height
                )
            )

            st.altair_chart(chart, use_container_width=True)

        else:
            st.info("暂无工具调用统计数据。")


st.divider()

# ------------------------------
# RAG 评估历史
# ------------------------------
st.subheader("RAG 评估历史")

if "eval_logs" not in st.session_state:
    st.session_state.eval_logs = []

eval_log_limit = st.number_input(
    "查询最近多少条评估记录",
    min_value=1,
    max_value=100,
    value=20,
    step=1,
    key="eval_log_limit"
)

if st.button("查询 RAG 评估历史"):
    eval_logs_data, eval_logs_error = safe_get(
        "/eval_logs",
        params={"limit": eval_log_limit}
    )

    if eval_logs_error:
        st.error(f"查询评估历史失败：{eval_logs_error}")
    else:
        st.session_state.eval_logs = eval_logs_data.get("data", [])
        st.success(f"查询成功，共 {len(st.session_state.eval_logs)} 条记录")


logs = st.session_state.eval_logs

if not logs:
    st.info("暂无 RAG 评估记录，或还没有点击查询。")
else:
    st.write(f"共查询到 {len(logs)} 条评估记录")

    df_eval_logs = pd.DataFrame(logs)

    st.dataframe(df_eval_logs, use_container_width=True)

    csv_data = df_eval_logs.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="导出 RAG 评估记录为 CSV",
        data=csv_data,
        file_name="rag_eval_logs.csv",
        mime="text/csv"
    )

    for log in logs:
        log_id = log.get("ID")
        question = log.get("问题", "未知问题")
        score_value = log.get("评分", "未知评分")
        created_at = log.get("创建时间", "未知时间")

        with st.expander(f"#{log_id} | 评分：{score_value} | {question}"):
            st.write(f"**创建时间：** {created_at}")

            st.write("**问题：**")
            st.write(log.get("问题", ""))

            st.write("**标准答案：**")
            st.write(log.get("标准答案", ""))

            st.write("**RAG 回答：**")
            st.write(log.get("RAG回答", ""))

            st.write("**来源：**")
            st.write(log.get("来源", ""))

            st.write("**评分：**")
            st.write(log.get("评分", ""))

            st.write("**备注：**")

            current_note = log.get("备注", "") or ""

            new_eval_note = st.text_area(
                "编辑评估备注",
                value=current_note,
                key=f"eval_note_{log_id}"
            )

            if st.button("保存评估备注", key=f"save_eval_note_{log_id}"):
                update_data, update_error = safe_put(
                    f"/eval_logs/{log_id}/note",
                    json={"note": new_eval_note}
                )

                if update_error:
                    st.error(f"保存评估备注失败：{update_error}")
                else:
                    st.success(update_data.get("message", "评估备注保存成功"))

                    eval_logs_data, eval_logs_error = safe_get(
                        "/eval_logs",
                        params={"limit": eval_log_limit}
                    )

                    if not eval_logs_error:
                        st.session_state.eval_logs = eval_logs_data.get("data", [])

                    st.rerun()

            if st.button("删除这条评估记录", key=f"delete_eval_log_{log_id}"):
                delete_data, delete_error = safe_delete(f"/eval_logs/{log_id}")

                if delete_error:
                    st.error(f"删除评估记录失败：{delete_error}")
                else:
                    st.success(delete_data.get("message", "评估记录删除成功"))

                    eval_logs_data, eval_logs_error = safe_get(
                        "/eval_logs",
                        params={"limit": eval_log_limit}
                    )

                    if not eval_logs_error:
                        st.session_state.eval_logs = eval_logs_data.get("data", [])

                    st.rerun()

st.subheader("RAG 评估统计")

if st.button("刷新 RAG 评估统计"):
    eval_stats_data, eval_stats_error = safe_get("/eval_logs/stats")

    if eval_stats_error:
        st.error(f"查询 RAG 评估统计失败：{eval_stats_error}")
    else:
        stats = eval_stats_data.get("data", {})

        total_count = stats.get("评估总数", 0)
        avg_score = stats.get("平均分", 0)
        max_score = stats.get("最高分", 0)
        min_score = stats.get("最低分", 0)
        score_distribution = stats.get("评分分布", [])

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("评估总数", total_count)
        col2.metric("平均分", avg_score)
        col3.metric("最高分", max_score)
        col4.metric("最低分", min_score)

        if not score_distribution:
            st.info("暂无评分统计数据。")
        else:
            df_score_dist = pd.DataFrame(score_distribution)

            st.write("评分分布表：")
            st.dataframe(df_score_dist, use_container_width=True)

            chart = (
                alt.Chart(df_score_dist)
                .mark_bar()
                .encode(
                    x=alt.X("评分:N", title="评分"),
                    y=alt.Y("数量:Q", title="数量"),
                    tooltip=[
                        alt.Tooltip("评分:N", title="评分"),
                        alt.Tooltip("数量:Q", title="数量")
                    ]
                )
                .properties(height=300)
            )

            st.altair_chart(chart, use_container_width=True)

st.subheader("按 top_k 分组统计")

if st.button("刷新 top_k 评分统计"):
    top_k_stats_data, top_k_stats_error = safe_get("/eval_logs/top_k_stats")

    if top_k_stats_error:
        st.error(f"查询 top_k 统计失败：{top_k_stats_error}")
    else:
        top_k_stats = top_k_stats_data.get("data", [])

        if not top_k_stats:
            st.info("暂无 top_k 统计数据。请先保存带有 top_k 参数的评估记录。")
        else:
            df_top_k_stats = pd.DataFrame(top_k_stats)

            st.write("不同 top_k 的评分统计：")
            st.dataframe(df_top_k_stats, use_container_width=True)

            chart = (
                alt.Chart(df_top_k_stats)
                .mark_bar()
                .encode(
                    x=alt.X("top_k:N", title="top_k"),
                    y=alt.Y("平均分:Q", title="平均分"),
                    tooltip=[
                        alt.Tooltip("top_k:N", title="top_k"),
                        alt.Tooltip("评估数量:Q", title="评估数量"),
                        alt.Tooltip("平均分:Q", title="平均分"),
                        alt.Tooltip("最高分:Q", title="最高分"),
                        alt.Tooltip("最低分:Q", title="最低分")
                    ]
                )
                .properties(height=300)
            )

            st.altair_chart(chart, use_container_width=True)

st.divider()

# ===============================
# 4. 数据库统计信息
# ===============================

st.header("4. 数据库统计信息")

stats_data, stats_error = safe_get("/stats")

if stats_error:
    st.error(f"加载统计信息失败：{stats_error}")
else:
    stats = stats_data.get("data", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("问答记录总数", stats.get("问答记录总数", 0))
    col2.metric("文档记录总数", stats.get("文档记录总数", 0))

    latest_time = stats.get("最近问答时间")
    col3.metric("最近问答时间", latest_time if latest_time else "暂无")

    with st.expander("来源文件引用次数"):
        source_stats = stats.get("来源文件统计", [])

        if source_stats:
            st.dataframe(pd.DataFrame(source_stats), use_container_width=True)
        else:
            st.info("暂无来源文件统计。")


st.subheader("知识库文档管理")

if st.button("刷新文档列表"):
    docs_data, docs_error = safe_get("/documents")

    if docs_error:
        st.error(f"查询文档失败：{docs_error}")
    else:
        st.session_state.documents = docs_data.get("data", [])

if "documents" not in st.session_state:
    st.session_state.documents = []

documents = st.session_state.documents

if not documents:
    st.info("暂无文档记录，或还没有点击刷新。")
else:
    st.write(f"当前文档数量：{len(documents)}")

    df_docs = pd.DataFrame(documents)
    st.dataframe(df_docs, use_container_width=True)

    for doc in documents:
        file_id = doc.get("ID")
        file_name = doc.get("文件名")
        file_hash = doc.get("文件hash")
        chunk_count = doc.get("文本块数量")

        with st.expander(f"文档 #{file_id}：{file_name}"):
            st.write(f"文件名：{file_name}")
            st.write(f"文件 hash：{file_hash}")
            st.write(f"文本块数量：{chunk_count}")

            confirm_delete = st.checkbox(
                f"确认删除 {file_name}",
                key=f"confirm_delete_doc_{file_hash}"
            )

            if st.button(
                "删除该文档并重建索引",
                key=f"delete_doc_{file_hash}"
            ):
                if not confirm_delete:
                    st.warning("请先勾选确认删除。")
                else:
                    delete_data, delete_error = safe_delete(
                        f"/documents/{file_hash}"
                    )

                    if delete_error:
                        st.error(f"删除文档失败：{delete_error}")
                    else:
                        if delete_data.get("success"):
                            st.success(delete_data.get("message", "删除成功"))
                        else:
                            st.warning(delete_data.get("message", "删除失败"))

                        st.json(delete_data)

                        docs_data, docs_error = safe_get("/documents")

                        if not docs_error:
                            st.session_state.documents = docs_data.get("data", [])

                        st.rerun()

st.divider()


# ===============================
# 5. 问答历史记录
# ===============================

st.header("5. 问答历史记录")

history_col1, history_col2, history_col3 = st.columns([1, 1, 2])

with history_col1:
    log_limit = st.number_input(
        "显示最近多少条",
        min_value=5,
        max_value=200,
        value=20,
        step=5
    )

with history_col2:
    log_source_file = st.text_input("来源文件筛选")

with history_col3:
    log_keyword = st.text_input("问题关键词搜索")

params = {
    "limit": int(log_limit)
}

if log_keyword.strip():
    params["keyword"] = log_keyword.strip()

if log_source_file.strip():
    params["source_file"] = log_source_file.strip()

qa_data, qa_error = safe_get("/qa_logs", params=params)

if qa_error:
    st.error(f"加载问答历史失败：{qa_error}")
else:
    qa_logs = qa_data.get("data", [])

    if qa_logs:
        df_logs = pd.DataFrame(qa_logs)
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("暂无符合条件的问答记录。")

st.subheader("删除问答记录")

delete_id = st.number_input(
    "请输入要删除的问答记录 ID",
    min_value=1,
    step=1
)

if st.button("删除这条记录"):
    delete_data, delete_error = safe_delete(f"/qa_logs/{int(delete_id)}")

    if delete_error:
        st.error(f"删除失败：{delete_error}")
    else:
        st.success(delete_data.get("message", "删除成功"))
        st.rerun()

st.subheader("更新问答备注")

note_log_id = st.number_input(
    "请输入要更新备注的问答记录 ID",
    min_value=1,
    step=1,
    key="note_log_id"
)

note_text = st.text_area(
    "请输入备注内容",
    placeholder="例如：这条回答质量较好，可以作为简历项目介绍参考",
    key="note_text"
)

if st.button("更新备注"):
    payload = {
        "note": note_text.strip()
    }

    update_data, update_error = safe_put(
        f"/qa_logs/{int(note_log_id)}/note",
        json=payload
    )

    if update_error:
        st.error(f"更新备注失败：{update_error}")
    else:
        st.success(update_data.get("message", "备注更新成功"))
        st.rerun()

st.divider()


# ===============================
# 6. 文档列表
# ===============================

st.header("6. 已上传文档列表")

doc_data, doc_error = safe_get("/documents")

if doc_error:
    st.error(f"加载文档列表失败：{doc_error}")
else:
    documents = doc_data.get("data", [])

    if documents:
        df_documents = pd.DataFrame(documents)
        st.dataframe(df_documents, use_container_width=True)
    else:
        st.info("暂无文档记录。")


# ===============================
# 7. 使用说明
# ===============================

with st.expander("使用说明"):
    st.markdown(
        """
        1. 先启动 FastAPI 后端：`uvicorn api_app:app`
        2. 再启动 Streamlit 前端：`streamlit run main.py`
        3. 先上传 PDF / TXT 文档，等待知识库构建完成。
        4. 再输入问题，系统会从知识库中检索相关文本并调用大模型回答。
        5. 问答结果会自动保存到 MySQL 的 `qa_logs` 表。
        6. 上传文档记录会保存到 MySQL 的 `documents` 表。
        """
    )
