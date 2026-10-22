def append_results(
    results: list,
    query: str,
    reranked_results: list,
    answer: str,
    search_scope: str
) -> list:
    """
    作用：
    把当前问题的检索结果、评分、来源文件和模型回答整理成表格数据。

    输入：
    results：
        总实验结果列表。

    query：
        当前用户问题。

    reranked_results：
        Rerank 排序后的检索结果列表。

    answer：
        大模型根据检索内容生成的回答。

    处理流程：
    1. 遍历 reranked_results。
    2. 取出每条检索结果中的预览文本、Rerank 分数、余弦相似度、FAISS 距离、来源文件。
    3. 把这些信息和当前问题、模型回答一起保存到 results。
    4. 增加 是否Top1 和 回答长度，方便实验分析。

    输出：
    results：
        更新后的实验结果列表。
    """
    for rank, item in enumerate(reranked_results, start=1):
        results.append({
            "问题": query,
            "检索范围": search_scope,
            "检索序号": rank,
            "是否Top1": "是" if rank == 1 else "否",
            "来源文件": item[6],
            "文本块编号": item[7],
            "文本内容": item[1],
            "FAISS原始距离": float(item[5]),
            "Rerank评分": float(item[3]),
            "余弦相似度评分": float(item[4]),
            "回答": answer,
            "回答长度": len(answer)
        })

    return results