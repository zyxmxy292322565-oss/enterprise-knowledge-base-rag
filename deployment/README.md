# 部署文件

从仓库根目录使用 [统一启动指南](../docs/deployment.md)。[compose.yml](compose.yml) 管理 FastAPI、Streamlit、MySQL 和 Redis，[compose.milvus.yml](compose.milvus.yml) 管理隔离的 Milvus、etcd 和 MinIO。应用和页面共用一个 [CPU 镜像](Dockerfile)。

[init_schema.py](init_schema.py) 在新建库显式执行 SQL。恢复既有数据时，由操作者核对快照制品、向量集合回执、数据库与发布代次后再挂载。凭据和模型文件均从部署方控制的路径注入。历史评测文件可放在私有资源下的 `evaluation-history/`。

启用真实 DeepSeek 时，按 [模型入口](../app/models.py) 设置私有运行配置中的模型名、服务地址和凭据。若使用 [预算客户端](../app/llm_budget.py)，还需启用预算模式并提供私有费用账本。图谱抽取、Router 和语义校验分别有独立模型配置。默认启动使用本地替身。
