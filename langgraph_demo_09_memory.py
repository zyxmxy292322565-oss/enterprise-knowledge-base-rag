from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# ------------------------------
# 1. 定义 State
# ------------------------------
class ChatState(TypedDict, total=False):
    user_input: str
    answer: str
    conversation_history: List[Dict[str, str]]


# ------------------------------
# 2. 对话节点
# ------------------------------
def chat_node(state: ChatState):
    user_input = state["user_input"]

    # 读取历史，如果没有历史，就用空列表
    history = state.get("conversation_history", [])

    # 这里先不用真实 LLM，先做一个简单演示
    if "我刚才说我叫什么" in user_input:
        answer = "我从历史里看到，你刚才说你叫周宇轩。"
    else:
        answer = f"我收到你的问题了：{user_input}"

    # 把本轮用户问题和助手回答追加进历史
    new_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer}
    ]

    return {
        "answer": answer,
        "conversation_history": new_history
    }


# ------------------------------
# 3. 构建 Graph
# ------------------------------
graph_builder = StateGraph(ChatState)

graph_builder.add_node("chat_node", chat_node)

graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)


# ------------------------------
# 4. 添加 checkpointer
# ------------------------------
checkpointer = InMemorySaver()

graph = graph_builder.compile(checkpointer=checkpointer)


# ------------------------------
# 5. 测试
# ------------------------------
if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "user_1"
        }
    }

    result1 = graph.invoke(
        {
            "user_input": "我叫周宇轩"
        },
        config=config
    )

    print("第一次结果：")
    print(result1)

    result2 = graph.invoke(
        {
            "user_input": "我刚才说我叫什么？"
        },
        config=config
    )

    print("\n第二次结果：")
    print(result2)