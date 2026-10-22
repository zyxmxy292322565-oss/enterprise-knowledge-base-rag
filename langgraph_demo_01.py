from typing import TypedDict, Dict, Any
import json

from langgraph.graph import StateGraph, START, END
from db_utils import save_agent_log

from agent_utils import (
    load_agent_llm,
    get_documents_tool,
    get_qa_logs_tool,
    get_stats_tool,
    rag_qa_tool,
    summarize_tool_result
)


# ------------------------------
# 1. 加载 LLM
# ------------------------------
llm = load_agent_llm()


# ------------------------------
# 2. 定义 State
# ------------------------------
class AgentState(TypedDict):
    user_input: str
    tool_name: str
    args: Dict[str, Any]
    tool_result: str
    answer: str
    log_saved: bool
    error: str


# ------------------------------
# 3. JSON 解析函数
# ------------------------------
def parse_router_json(text: str) -> Dict[str, Any]:
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


# ------------------------------
# 4. 真实工具包装函数
# ------------------------------
def call_get_documents_tool(args: Dict[str, Any]) -> str:
    return get_documents_tool.invoke({})


def call_get_qa_logs_tool(args: Dict[str, Any]) -> str:
    limit = args.get("limit", 3)
    keyword = args.get("keyword", "")
    source_file = args.get("source_file", "")

    try:
        limit = int(limit)
    except Exception:
        limit = 3

    limit = max(1, min(limit, 50))

    return get_qa_logs_tool.invoke({
        "limit": limit,
        "keyword": keyword,
        "source_file": source_file
    })


def call_get_stats_tool(args: Dict[str, Any]) -> str:
    return get_stats_tool.invoke({})


def call_rag_qa_tool(args: Dict[str, Any]) -> str:
    question = args.get("question", "")

    return rag_qa_tool.invoke({
        "question": question
    })


# ------------------------------
# 5. 工具映射表
# ------------------------------
tool_map = {
    "get_documents_tool": call_get_documents_tool,
    "get_qa_logs_tool": call_get_qa_logs_tool,
    "get_stats_tool": call_get_stats_tool,
    "rag_qa_tool": call_rag_qa_tool
}


# ------------------------------
# 6. router_node：真实 LLM Router
# ------------------------------
def router_node(state: AgentState):
    user_input = state["user_input"]

    router_prompt = f"""
你是一个企业知识库系统的 Agent Router。

你需要根据用户问题，从下面工具中选择一个最合适的工具。

可用工具：

1. get_documents_tool
作用：查询当前知识库有哪些文档。
适合问题：
- 当前知识库有哪些文档？
- 文档列表是什么？
- 上传了哪些文件？

2. get_qa_logs_tool
作用：查询历史问答记录。
参数：
- limit：查询最近多少条，默认 3
- keyword：关键词筛选，默认空字符串
- source_file：来源文件筛选，默认空字符串
适合问题：
- 最近 3 条问答记录是什么？
- 历史问答有哪些？
- 最近问了什么？
- 查询和 RAG 相关的问答记录

3. get_stats_tool
作用：查询系统统计信息。
适合问题：
- 当前系统有多少文档？
- 有多少条问答记录？
- 系统统计信息是什么？

4. rag_qa_tool
作用：基于知识库内容回答问题。
参数：
- question：用户原始问题。
适合问题：
- 总结这个文档
- 文档主要讲了什么？
- 根据知识库回答某个具体问题

用户问题：
{user_input}

请你只输出 JSON，不要输出解释文字。

JSON 格式必须是：

{{
  "tool_name": "工具名称",
  "args": {{
  }}
}}

示例 1：

{{
  "tool_name": "get_documents_tool",
  "args": {{}}
}}

示例 2：

{{
  "tool_name": "get_qa_logs_tool",
  "args": {{
    "limit": 3,
    "keyword": "",
    "source_file": ""
  }}
}}

示例 3：

{{
  "tool_name": "rag_qa_tool",
  "args": {{
    "question": "帮我总结一下这个文档"
  }}
}}
"""

    response = llm.invoke(router_prompt)

    router_result = parse_router_json(response.content)

    tool_name = router_result.get("tool_name", "rag_qa_tool")
    args = router_result.get("args", {})

    if tool_name == "rag_qa_tool" and "question" not in args:
        args["question"] = user_input

    return {
        "tool_name": tool_name,
        "args": args
    }


# ------------------------------
# 7. tool_node：调用真实工具
# ------------------------------
def tool_node(state: AgentState):
    tool_name = state["tool_name"]
    args = state["args"]

    tool_func = tool_map.get(tool_name)

    if tool_func is None:
        return {
            "tool_result": f"没有找到工具：{tool_name}"
        }

    try:
        tool_result = tool_func(args)
    except Exception as e:
        tool_result = f"工具调用失败：{e}"

    return {
        "tool_result": tool_result
    }


# ------------------------------
# 8. summary_node：总结工具结果
# ------------------------------
def summary_node(state: AgentState):
    user_input = state["user_input"]
    tool_name = state["tool_name"]
    args = state["args"]
    tool_result = state["tool_result"]

    try:
        answer = summarize_tool_result(
            user_input=user_input,
            tool_name=tool_name,
            args=args,
            tool_result=tool_result
        )
    except Exception as e:
        answer = f"工具结果如下，但生成最终回答失败：{e}\n\n工具原始结果：\n{tool_result}"

    return {
        "answer": answer
    }

# ------------------------------
# 9. save_log_node：保存 Agent 调用日志
# ------------------------------
def save_log_node(state: AgentState):
    user_input = state["user_input"]
    tool_name = state["tool_name"]
    args = state["args"]
    tool_result = state["tool_result"]
    answer = state["answer"]

    try:
        save_agent_log(
            question=user_input,
            tool_name=tool_name,
            args=args,
            tool_result=tool_result,
            answer=answer
        )

        return {
            "log_saved": True,
            "error": ""
        }

    except Exception as e:
        return {
            "log_saved": False,
            "error": f"保存 Agent 日志失败：{e}"
        }

# ------------------------------
# 10. 构建 LangGraph
# ------------------------------
graph_builder = StateGraph(AgentState)

graph_builder.add_node("router_node", router_node)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_node("summary_node", summary_node)
graph_builder.add_node("save_log_node", save_log_node)

graph_builder.add_edge(START, "router_node")
graph_builder.add_edge("router_node", "tool_node")
graph_builder.add_edge("tool_node", "summary_node")
graph_builder.add_edge("summary_node", "save_log_node")
graph_builder.add_edge("save_log_node", END)

graph = graph_builder.compile()


# ------------------------------
# 10. 测试
# ------------------------------
if __name__ == "__main__":
    result = graph.invoke({
        "user_input": "当前知识库有哪些文档？",
        "tool_name": "",
        "args": {},
        "tool_result": "",
        "answer": "",
        "log_saved": False,
        "error": ""
    })

    print(result)