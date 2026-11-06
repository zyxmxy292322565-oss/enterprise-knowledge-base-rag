# 架构与源码导航

| 目录 | 内容 |
| --- | --- |
| [app](../app) | FastAPI/Streamlit、知识和图谱服务、检索、可靠性与评测 |
| [deployment](../deployment) | CPU Docker 镜像、应用 Compose、Milvus Compose、入口与初始化 |
| [scripts](../scripts) | 从空资源目录启动的命令 |
| [migrations](../migrations) | 新库初始化仍依赖的 SQL |
| [tests](../tests) | 单元、回归、独立资源集成 |
| [examples](../examples) | 可上传的合成政策 |

FastAPI 提供数据与查询接口，Streamlit 调用这些接口展示中台。`KnowledgeStore` 在 MySQL 记录版本和发布状态，在文件系统保存原文、Parent/Child 清单和 BM25 索引，在 Milvus 保存每个快照的向量集合。Redis 按快照和发布代次隔离回答或证据缓存。知识图谱及审核记录随快照 ID 存在 MySQL。

| 源码 | 职责 |
| --- | --- |
| [api_app.py](../app/api_app.py) | 应用生命周期、普通问答与政策证据 API |
| [knowledge_v2.py](../app/knowledge_v2.py) | 解析、切分、索引构建、快照与发布 |
| [milvus_store.py](../app/milvus_store.py) | Milvus 集合绑定、行校验和向量检索 |
| [retrieval_engine.py](../app/retrieval_engine.py) | 查询规划、通道执行、RRF、重排和邻文 |
| [agent_router.py](../app/agent_router.py) · [request_budget.py](../app/request_budget.py) | 规则及模型建议，请求期限和阶段配额 |
| [knowledge_graph.py](../app/knowledge_graph.py) | 图谱证据、审核门禁、图路径查询 |
| [semantic_reliability.py](../app/semantic_reliability.py) | 逐声明证据判断与引用核验 |
| [eval_center_runner.py](../app/eval_center_runner.py) | 固定快照评测、候选与最终证据记录 |

业务模块集中在 `app/`，容器通过 `PYTHONPATH=/app/app` 保持模块导入兼容。镜像由仓库根目录作为构建上下文，使用 [专用白名单](../deployment/Dockerfile.dockerignore) 控制内容。
