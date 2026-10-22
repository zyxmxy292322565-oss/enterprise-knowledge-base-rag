# RAG 知识库问答与评测

基于 FastAPI、Streamlit、FAISS、BGE embedding 和可选 CrossEncoder 的文档问答系统。服务提供文档解析、向量召回、重排、上下文构造和回答生成，并通过 MySQL、Redis 保存记录及缓存。`/ask` 用于生成回答；`/rag/query` 面向政策检索，返回来源证据供业务服务校验，不直接执行业务写操作。另有独立智能客服项目可通过 HTTP 调用该接口。

## 评测与实验

评测通过内部旁路避开回答缓存，记录参数、候选、最终证据和耗时。一次离线实验使用 50 份政策草案、66 道题和四组参数：候选固定 10 条、关闭 Rerank 时，最终证据从 3 增至 5，来源文件级 Evidence Hit 由 55/66（83.33%）增至 60/66（90.91%），宏平均来源 Recall 由 65.40% 增至 75.51%。这些是本次草案样本上的结果，不是生产效果或段落级准确率。重排和阈值组合在该样本没有改善整体命中。

相同题目进行两组各 66 次真实模型回答记录：关键词宏均分 58.08% → 63.51%，总 Token 77,789 → 114,149。关键词分只统计字符串命中，不是准确率；事实覆盖和严格语义审阅依赖 Codex 产生的初步标签，尚未独立人工复核。评分脚本对禁止断言及部分行为使用预设值，不能视为独立验证。仅一次实验，不证明普遍因果。汇总见 `evaluation/experiments/`；原始逐题记录、政策快照及数据集因隐私和材料授权未核验而不纳入本仓库，分析脚本因此不能仅凭仓库文件重算历史结果。

## 本地运行

Python 依赖见 `requirements.txt`；复制 `.env.example` 为本地环境文件并自行设置模型、MySQL、Redis 配置。`uvicorn api_app:app --host 127.0.0.1 --port 8000` 启动 API，另行 `streamlit run main.py` 启动页面。启动涉及外部服务和模型下载，模型调用可能产生费用。当前 `docker-compose.yml` 引用未提交的 `.env.compose` 和 `.env.frontend`，须自行准备；不可直接复用他人密钥。`evaluation/docker-compose.yml` 定义隔离评测服务，重跑需要自行准备经授权语料、数据集、索引和凭据并承担模型调用费用，不会自动重现历史原始记录。

测试文件包括 `test_eval_cache_bypass.py`、`test_policy_rag_contract.py`、`evaluation/test_b3.py`；请在隔离环境阅读其依赖及副作用后执行，例如 `python -m unittest test_eval_cache_bypass test_policy_rag_contract`。测试是否通过取决于环境，本仓库不声称当前已运行。

当前限制：文档与评测集为待审核草案；来源级指标不能替代语义判断；模型权重版本未完全固定；数据与索引未公开。相关历史实验报告中对费用或审核的描述应结合上述限制理解。
