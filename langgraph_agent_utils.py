from typing import TypedDict, Dict, Any
import json

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from db_utils import (
    save_agent_log,
    save_conversation_message,
    load_conversation_history
)

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

    # Router 调试信息
    router_raw: str
    router_validated: Dict[str, Any]
    # 多轮对话历史
    conversation_history: list
    thread_id: str


# ------------------------------
# 3. JSON 解析函数
# ------------------------------
def parse_router_json(text: str) -> Dict[str, Any]:
    """
    把 LLM 输出的 JSON 字符串解析成 Python dict。
    如果解析失败，返回空字典，由 validate_router_result 统一兜底。
    """

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

    return {}


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

# 参数转换函数
def safe_int(value, default: int = 3) -> int:
    try:
        return int(value)
    except Exception:
        return default

# Router 校验函数
def validate_router_result(router_result: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    """
    校验 LLM Router 的输出结果。
    目标：
    1. tool_name 必须是已注册工具
    2. args 必须是 dict
    3. 不同工具补齐默认参数
    4. 出错时兜底到 rag_qa_tool
    """

    if not isinstance(router_result, dict):
        return {
            "tool_name": "rag_qa_tool",
            "args": {
                "question": user_input
            }
        }

    tool_name = router_result.get("tool_name", "")
    args = router_result.get("args", {})

    if not isinstance(args, dict):
        args = {}

    if tool_name not in tool_map:
        return {
            "tool_name": "rag_qa_tool",
            "args": {
                "question": user_input
            }
        }

    if tool_name == "get_documents_tool":
        return {
            "tool_name": "get_documents_tool",
            "args": {}
        }

    if tool_name == "get_stats_tool":
        return {
            "tool_name": "get_stats_tool",
            "args": {}
        }

    if tool_name == "get_qa_logs_tool":
        limit = safe_int(args.get("limit", 3), 3)
        limit = max(1, min(limit, 50))

        keyword = args.get("keyword", "")
        source_file = args.get("source_file", "")

        return {
            "tool_name": "get_qa_logs_tool",
            "args": {
                "limit": limit,
                "keyword": str(keyword),
                "source_file": str(source_file)
            }
        }

    if tool_name == "rag_qa_tool":
        question = args.get("question", user_input)

        if not question:
            question = user_input

        return {
            "tool_name": "rag_qa_tool",
            "args": {
                "question": str(question)
            }
        }

    return {
        "tool_name": "rag_qa_tool",
        "args": {
            "question": user_input
        }
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

    router_raw = response.content

    raw_router_result = parse_router_json(router_raw)

    router_result = validate_router_result(
        router_result=raw_router_result,
        user_input=user_input
    )

    tool_name = router_result["tool_name"]
    args = router_result["args"]

    return {
        "tool_name": tool_name,
        "args": args,
        "router_raw": router_raw,
        "router_validated": router_result
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
            "tool_result": "",
            "error": f"没有找到工具：{tool_name}"
        }

    try:
        tool_result = tool_func(args)

        return {
            "tool_result": tool_result,
            "error": ""
        }

    except Exception as e:
        return {
            "tool_result": "",
            "error": f"工具调用失败：{e}"
        }

def route_after_tool(state: AgentState) -> str:
    if state.get("error"):
        return "error_node"

    return "summary_node"

# ------------------------------
# 8. summary_node：总结工具结果
# ------------------------------
def summary_node(state: AgentState):
    user_input = state["user_input"]
    tool_name = state["tool_name"]
    args = state["args"]
    tool_result = state["tool_result"]

    history = state.get("conversation_history", [])

    history_text = ""
    for msg in history[-6:]:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            history_text += f"用户：{content}\n"
        elif role == "assistant":
            history_text += f"助手：{content}\n"

    try:
        summary_prompt = f"""
你是一个企业知识库智能助手。

以下是当前会话的最近历史：
{history_text}

用户本轮问题：
{user_input}

Agent 选择的工具：
{tool_name}

工具参数：
{args}

工具返回结果：
{tool_result}

请根据工具返回结果，并结合必要的会话历史，用简洁自然的中文回答用户。

要求：
1. 优先依据工具返回结果回答
2. 不要编造工具结果里没有的信息
3. 如果用户的问题涉及“刚才”“上一个”“继续”等上下文，可以参考会话历史
4. 回答要简洁清楚
"""

        response = llm.invoke(summary_prompt)

        return {
            "answer": response.content
        }

    except Exception as e:
        return {
            "answer": f"生成最终回答失败：{e}"
        }

def error_node(state: AgentState):
    user_input = state["user_input"]
    tool_name = state["tool_name"]
    args = state["args"]
    error = state["error"]

    answer = f"""
本次 Agent 调用失败。

用户问题：
{user_input}

尝试调用的工具：
{tool_name}

工具参数：
{args}

失败原因：
{error}
"""

    return {
        "answer": answer
    }

def memory_node(state: AgentState):
    thread_id = state.get("thread_id", "default")
    user_input = state["user_input"]
    answer = state.get("answer", "")

    history = state.get("conversation_history", [])

    new_history = history + [
        {
            "role": "user",
            "content": user_input
        },
        {
            "role": "assistant",
            "content": answer
        }
    ]

    new_history = new_history[-10:]

    try:
        save_conversation_message(
            thread_id=thread_id,
            role="user",
            content=user_input
        )

        save_conversation_message(
            thread_id=thread_id,
            role="assistant",
            content=answer
        )

        return {
            "conversation_history": new_history
        }

    except Exception as e:
        old_error = state.get("error", "")
        memory_error = f"保存 conversation_history 失败：{e}"

        if old_error:
            error = old_error + "\n" + memory_error
        else:
            error = memory_error

        return {
            "conversation_history": new_history,
            "error": error
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
    router_raw = state.get("router_raw", "")
    router_validated = state.get("router_validated", {})

    try:
        save_agent_log(
            question=user_input,
            tool_name=tool_name,
            args=args,
            tool_result=tool_result,
            answer=answer,
            router_raw=router_raw,
            router_validated=router_validated
        )

        return {
            "log_saved": True
        }

    except Exception as e:
        old_error = state.get("error", "")
        save_error = f"保存 Agent 日志失败：{e}"

        if old_error:
            error = old_error + "\n" + save_error
        else:
            error = save_error

        return {
            "log_saved": False,
            "error": error
        }

# ------------------------------
# 10. 构建 LangGraph
# ------------------------------
graph_builder = StateGraph(AgentState)

graph_builder.add_node("router_node", router_node)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_node("summary_node", summary_node)
graph_builder.add_node("error_node", error_node)
graph_builder.add_node("memory_node", memory_node)
graph_builder.add_node("save_log_node", save_log_node)

graph_builder.add_edge(START, "router_node")
graph_builder.add_edge("router_node", "tool_node")

graph_builder.add_conditional_edges(
    "tool_node",
    route_after_tool,
    {
        "summary_node": "summary_node",
        "error_node": "error_node"
    }
)

graph_builder.add_edge("summary_node", "memory_node")
graph_builder.add_edge("error_node", "memory_node")
graph_builder.add_edge("memory_node", "save_log_node")
graph_builder.add_edge("save_log_node", END)

checkpointer = InMemorySaver()

graph = graph_builder.compile(checkpointer=checkpointer)

# ------------------------------
# 10. 测试
# 对外暴露的函数
# ------------------------------
def graph_agent_chat(user_input: str, thread_id: str = "default") -> dict:
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    history = load_conversation_history(thread_id=thread_id, limit=10)

    result = graph.invoke(
        {
            "user_input": user_input,
            "thread_id": thread_id,
            "tool_name": "",
            "args": {},
            "tool_result": "",
            "answer": "",
            "log_saved": False,
            "error": "",
            "router_raw": "",
            "router_validated": {},
            "conversation_history": history
        },
        config=config
    )

    return result

if __name__ == "__main__":
    result = graph_agent_chat("当前知识库有哪些文档？")
    print(result)