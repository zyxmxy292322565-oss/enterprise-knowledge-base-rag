# RAG 检索与证据链路

1. 请求从活动指针取得一个快照。普通查询默认向量召回，显式配置可开启 BM25、RRF、Graph、Rewrite、Decomposition、Expansion 或 HyDE。
2. 查询理解保留原问题。规则改写从用户原话取上下文，拆解将复合问题变成独立检索任务，固定词表生成扩展。HyDE 输出只用于独立向量召回。
3. BGE 编码后在 Milvus 快照集合做 `FLAT / L2` 搜索。BM25 用 Child 文本倒排，Graph 从审核关系的证据回到 Child。
4. RRF 按通道族排名融合，同一 Child 行去重。CrossEncoder 对「问题、Child 文本」打分，随后按入口选择阈值与最终证据。
5. 高级 `/ask` 可在 Child 之后补父块邻文，按 Parent ID 去重。每个窗口左右最多各 180 字符，总预算默认 512 UTF-8 字节。`/ask` 的上下文字符上限与可靠性控制器的估算 Token 预算是不同机制。
6. `/rag/query` 返回 Child 证据和范围判断。普通 `/ask` 生成回答和来源列表。`/v2/retrieval/reliable` 的规则分支从核验过的规则渲染答案，语义分支用原文摘录核对声明与引用。

Agentic Router 在可靠性接口选择路径，模型建议需满足用户原话和能力开关约束。请求级工具、模型、图查询和时间配额可触发拒答或降级。Graph 的关键证据缺失或截断时，可靠性流程有额外门禁。

[检索源码](../app/retrieval_engine.py) · [策略](../app/agent_router.py) · [请求预算](../app/request_budget.py) · [语义校验](../app/semantic_reliability.py)
