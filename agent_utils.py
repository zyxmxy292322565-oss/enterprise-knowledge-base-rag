import os
import requests
import re
import json

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from db_utils import (
    load_document_records,
    load_qa_logs,
    load_dashboard_stats,
    save_agent_log
)

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

def load_agent_llm():
    """
    只加载 Agent 路由用的大模型。
    不从 models.py 导入，避免触发 sentence_transformers / scipy 加载。
    """
    load_dotenv()

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_API_BASE"),
        temperature=0
    )

@tool
def get_documents_tool() -> str:
    """
    查询当前知识库中已经上传的文档列表。
    当用户询问“有哪些文档”、“上传过哪些文件”、“知识库里有什么文件”时，使用这个工具。
    """
    documents = load_document_records()

    if not documents:
        return "当前知识库中还没有上传任何文档。"

    lines = ["当前知识库中的文档如下："]

    for i, doc in enumerate(documents, start=1):
        lines.append(
            f"{i}. 文件名：{doc.get('文件名', '未知')}，"
            f"文件类型：{doc.get('文件类型', '未知')}，"
            f"文本块数量：{doc.get('文本块数量', '未知')}，"
            f"上传时间：{doc.get('上传时间', '未知')}"
        )

    return "\n".join(lines)

@tool
def get_qa_logs_tool(
    limit: int = 10,
    keyword: str = "",
    source_file: str = ""
) -> str:
    """
    查询问答历史记录。

用途：
- 查询最近问答历史记录
- 查询某个关键词相关的问答历史记录
- 查询某个来源文件相关的问答历史记录
- 适合问题：最近问过什么、查看问答历史、最近几条问答记录、关于 RAG 的问答、有没有问过 Agent 相关问题、某个文件相关的问答历史
参数：
- limit：整数，表示查询多少条记录；如果用户没有说明数量，默认 5
- keyword：字符串，表示关键词；如果用户没有说明关键词，使用空字符串
- source_file：字符串，表示来源文件名；如果用户没有说明来源文件，使用空字符串
    """

    keyword = keyword.strip() if keyword else ""
    source_file = source_file.strip() if source_file else ""

    logs = load_qa_logs(
        limit=limit,
        source_file=source_file if source_file else "",
        keyword=keyword if keyword else None
    )

    if not logs:
        conditions = []

        if keyword:
            conditions.append(f"关键词「{keyword}」")

        if source_file:
            conditions.append(f"来源文件「{source_file}」")

        if conditions:
            return f"没有找到符合 {'，'.join(conditions)} 的问答历史记录。"

        return "当前还没有任何问答历史记录。"

    title_parts = [f"最近 {len(logs)} 条"]

    if keyword:
        title_parts.append(f"关于「{keyword}」")

    if source_file:
        title_parts.append(f"来源文件为「{source_file}」")

    title = "".join(title_parts) + "的问答记录如下："

    lines = [title]

    for i, log in enumerate(logs, start=1):
        question = log.get("问题") or log.get("question") or "未知问题"
        answer = log.get("回答") or log.get("answer") or "未知回答"
        current_source_file = (
                log.get("来源文件")
                or log.get("source_file")
                or "未知来源"
        )
        created_at = (
                log.get("创建时间")
                or log.get("created_at")
                or log.get("时间")
                or "未知时间"
        )

        answer_preview = answer[:150] + "..." if len(answer) > 150 else answer

        lines.append(
            f"{i}. 问题：{question}\n"
            f"   回答摘要：{answer_preview}\n"
            f"   来源文件：{current_source_file}\n"
            f"   创建时间：{created_at}"
        )

    return "\n\n".join(lines)

@tool
def get_stats_tool() -> str:
    """
    查询当前知识库系统的统计信息。
    当用户询问“有多少问答记录”、“有多少文档”、“哪个文件被引用最多”、“系统统计信息”时，使用这个工具。
    """
    stats = load_dashboard_stats()

    if not stats:
        return "当前没有可用的统计信息。"

    qa_count = stats.get("问答记录总数", 0)
    document_count = stats.get("文档记录总数", 0)
    latest_qa_time = stats.get("最近问答时间", "暂无")
    source_stats = stats.get("来源文件统计", [])

    lines = [
        "当前知识库系统统计信息如下：",
        f"1. 问答记录总数：{qa_count}",
        f"2. 文档记录总数：{document_count}",
        f"3. 最近问答时间：{latest_qa_time if latest_qa_time else '暂无'}"
    ]

    if source_stats:
        lines.append("4. 来源文件引用次数：")

        for i, item in enumerate(source_stats, start=1):
            source_file = item.get("来源文件", "未知文件")
            count = item.get("引用次数", 0)
            lines.append(f"   {i}. {source_file}：{count} 次")
    else:
        lines.append("4. 暂无来源文件引用统计。")

    return "\n".join(lines)

@tool
def rag_qa_tool(question: str) -> str:
    """
    根据当前已上传知识库文档回答用户问题。
    当用户询问文档内容、总结文档、根据资料回答问题、分析文档内容时，使用这个工具。
    question 表示用户要问知识库的问题。
    """
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            json={
                "question": question,
                "top_k": 3
            },
            timeout=300
        )

        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "未获得回答。")
        sources = data.get("sources", [])

        if sources:
            source_lines = ["", "参考来源："]

            for i, item in enumerate(sources, start=1):
                source_lines.append(
                    f"{i}. 来源文件：{item.get('source', '未知文件')}，"
                    f"文本块编号：{item.get('chunk_id', '未知')}"
                )

            return answer + "\n".join(source_lines)

        return answer

    except Exception as e:
        return f"调用 RAG 问答工具失败：{e}"

def extract_limit_from_text(text: str, default: int = 5, max_limit: int = 50) -> int:
    """
    从用户输入中提取数量。
    例如：
    “最近 5 条问答记录” → 5
    “查一下最近20条历史” → 20
    如果没有数字，就返回默认值。
    """
    match = re.search(r"\d+", text)

    if not match:
        return default

    limit = int(match.group())

    if limit <= 0:
        return default

    return min(limit, max_limit)

def parse_router_json(text: str) -> dict:
    """
    尝试从 LLM 输出中解析 JSON。
    期望格式：
    {
        "tool_name": "...",
        "args": {...}
    }
    """
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start:end + 1]
        try:
            return json.loads(json_text)
        except Exception:
            pass

    return {
        "tool_name": "rag_qa_tool",
        "args": {
            "question": text
        }
    }

def summarize_tool_result(user_input: str, tool_name: str, args: dict, tool_result: str) -> str:
    """
    让 LLM 根据工具返回结果，生成更自然的最终回答。
    """

    llm = load_agent_llm()

    prompt = f"""
你是一个企业知识库智能助手。

用户原始问题：
{user_input}

Agent 调用的工具：
{tool_name}

工具参数：
{args}

工具返回结果：
{tool_result}

请你根据工具返回结果，用自然、清晰的中文回答用户。

要求：
1. 不要编造工具结果里没有的信息
2. 如果工具结果显示没有找到记录，就明确告诉用户没有找到
3. 如果工具结果里有多条记录，可以适当总结重点
4. 回答要简洁，不要重复太多无关字段
"""

    response = llm.invoke(prompt)

    return response.content

def simple_agent_chat(user_input: str) -> dict:
    """
    简化版 Agent：
    1. 先让 LLM 输出 JSON，判断要调用哪个工具和参数
    2. Python 根据 JSON 调用对应工具
    3. 返回工具名、参数和工具结果
    """

    llm = load_agent_llm()

    json_example_1 = """
    {
      "tool_name": "get_documents_tool",
      "args": {}
    }
    """

    json_example_2 = """
    {
      "tool_name": "get_qa_logs_tool",
      "args": {
        "limit": 3,
        "keyword": "",
        "source_file": ""
      }
    }
    """

    json_example_3 = """
    {
      "tool_name": "get_stats_tool",
      "args": {}
    }
    """

    json_example_4 = """
    {
      "tool_name": "rag_qa_tool",
      "args": {
        "question": "这个文档主要讲了什么？"
      }
    }
    """

    json_example_qa_source = """
    {
      "tool_name": "get_qa_logs_tool",
      "args": {
        "limit": 5,
        "keyword": "",
        "source_file": "调用tools和多agent遇到的问题.txt"
      }
    }
    """

    router_prompt = f"""
    你是一个企业知识库 Agent 路由器。

    你需要根据用户问题，从下面 4 个工具中选择一个最合适的工具，并输出 JSON。

    可选工具：

    1. get_documents_tool
    用途：
    - 查询当前知识库中已经上传的文档列表
    - 适合问题：当前知识库有哪些文档、上传过哪些文件、系统里有什么文件
    参数：
    - 无参数

    2. get_qa_logs_tool
    用途：
    - 查询最近问答历史记录
    - 查询某个关键词相关的问答历史记录
    - 查询某个来源文件相关的问答历史记录
    - 适合问题：最近问过什么、查看问答历史、最近几条问答记录、关于 RAG 的问答、有没有问过 Agent 相关问题、某个文件相关的问答历史
    参数：
    - limit：整数，表示查询多少条记录；如果用户没有说明数量，默认 5
    - keyword：字符串，表示关键词；如果用户没有说明关键词，使用空字符串
    source_file：字符串，表示来源文件名；如果用户没有说明来源文件，使用空字符串

    3. get_stats_tool
    用途：
    - 查询系统统计信息
    - 适合问题：有多少问答记录、有多少文档、哪个文件被引用最多、系统统计信息
    参数：
    - 无参数

    4. rag_qa_tool
    用途：
    - 根据已上传知识库文档回答问题
    - 适合问题：总结文档、询问文档内容、根据资料回答问题、分析文档内容
    参数：
    - question：用户原始问题

    用户问题：
    {user_input}

    请严格只输出 JSON，不要输出解释文字。

    JSON 格式必须包含：
    - tool_name
    - args

    示例 1：
    用户问：当前知识库有哪些文档？
    输出：
    {json_example_1}

    示例 2：
    用户问：最近 5 条关于 RAG 的问答记录
    输出：
    {json_example_2}

    示例 3：
    用户问：当前系统有多少文档？
    输出：
    {json_example_3}

    示例 4：
    用户问：这个文档主要讲了什么？
    输出：
    {json_example_4}

    示例 5：
    用户问：查一下 调用tools和多agent遇到的问题.txt 的问答历史
    输出：
    {json_example_qa_source}
    """

    response = llm.invoke(router_prompt)

    router_result = parse_router_json(response.content)

    tool_name = router_result.get("tool_name", "rag_qa_tool")
    args = router_result.get("args", {})

    print("Agent 路由结果：", router_result)

    try:
        if tool_name == "get_documents_tool":
            tool_result = get_documents_tool.invoke({})

        elif tool_name == "get_qa_logs_tool":
            limit = args.get("limit", 5)
            keyword = args.get("keyword", "")
            source_file = args.get("source_file", "")

            try:
                limit = int(limit)
            except Exception:
                limit = 5

            limit = max(1, min(limit, 50))

            if keyword is None:
                keyword = ""

            if source_file is None:
                source_file = ""

            tool_result = get_qa_logs_tool.invoke({
                "limit": limit,
                "keyword": keyword,
                "source_file": source_file
            })

        elif tool_name == "get_stats_tool":
            tool_result = get_stats_tool.invoke({})

        elif tool_name == "rag_qa_tool":
            question = args.get("question", user_input)

            tool_result = rag_qa_tool.invoke({
                "question": question
            })

        else:
            tool_result = f"Agent 没有识别出合适工具，模型输出为：{response.content}"

    except Exception as e:
        tool_result = f"工具调用失败：{e}"

    try:
        answer = summarize_tool_result(
            user_input=user_input,
            tool_name=tool_name,
            args=args,
            tool_result=tool_result
        )
    except Exception as e:
        answer = f"工具已经返回结果，但生成最终回答时失败：{e}\n\n工具原始结果：\n{tool_result}"

    try:
        save_agent_log(
            question=user_input,
            tool_name=tool_name,
            args=args,
            tool_result=tool_result,
            answer=answer
        )
    except Exception as e:
        print("保存 Agent 日志失败：", e)

    return {
        "tool_name": tool_name,
        "args": args,
        "tool_result": tool_result,
        "answer": answer
    }



if __name__ == "__main__":
    print("开始测试简化版 Agent...")

    questions = [
        "当前知识库有哪些文档？",
        "最近 5 条问答记录是什么？",
        "当前系统有多少问答记录和文档？",
        "这个文档主要讲了什么？"
    ]

    for q in questions:
        print("\n==============================")
        print("用户问题：", q)

        result = simple_agent_chat(q)

        print("Agent回答：")
        print(result)