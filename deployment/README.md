# RAG V3 隔离部署

## 从空环境启动

需要 Git、Python 3.11+、Docker Compose v2（支持 `up --wait`）以及本机可用的 CPU、磁盘与内存。首次构建下载 Python 依赖和服务镜像；模型目录也需要提前准备。使用可信来源的模型版本，每个目录只保留一份权重（`model.safetensors` 或 `pytorch_model.bin`），以及完整的 tokenizer 和配置。不要传入由缓存符号链接组成的目录。

例如，可使用 Hugging Face CLI 把两套模型下载到仓库外：

```bash
python -m pip install huggingface_hub
hf download BAAI/bge-small-zh-v1.5 --local-dir ../rag-models/embed --include '*.json' '*.txt' 'model.safetensors'
hf download BAAI/bge-reranker-base --local-dir ../rag-models/rerank --include '*.json' '*.txt' 'model.safetensors'
python deployment/bootstrap.py --resource ../rag-v3-runtime --embed ../rag-models/embed --rerank ../rag-models/rerank
```

对照实验可在下载时通过 `--revision` 固定 Hugging Face 提交。启动器记录实际文件哈希，挂载模型目录为只读。它拒绝已有资源目录和占用端口；可使用 `--api-port`、`--ui-port`、`--mysql-port`、`--redis-port`、`--milvus-port` 和 `--milvus-health-port` 指定其他本机端口。运行清单和随机凭据写入新建私有目录，不需要任何已有数据库或索引。初始化仅在已核验的空 MySQL 库中显式执行一次；API 重启不执行迁移。初始化中断时，保留现场供排查，并用新资源目录重试。

1. 打开 `http://127.0.0.1:18632`，在数据中台上传一份自己的 TXT/PDF，构建快照，再发布。空库下 `/system/health` 显示服务健康、`ready_for_queries=false`；首个快照发布后可查询。
2. 在 API 文档 `http://127.0.0.1:18631/docs` 查接口输入输出。管理调用使用 `X-RAG-V2-Token`，从资源目录下的私有 `secrets/environment.json` 取值。验证时不要将凭据复制到命令历史或报告。
3. 发布后使用 `POST /rag/query` 验证检索、快照和 Child 证据；`POST /ask` 默认使用本地回答替身，输出不代表模型质量。图谱需要构建并审核，空图谱不会返回图路径。若需审核图谱，启动时传入 `--reviewer <本地审核身份>`，在图谱页面使用私有清单生成的 `RAG_V3_GRAPH_REVIEW_TOKEN`。评测中台初始为空，可导入自己的 Gold 数据；审核身份使用私有清单中的 `RAG_V2_EVAL_DEVELOPER` 或 `RAG_V2_EVAL_REVIEWERS`。

读取资源目录中 `projects.json` 的 `app`、`milvus` 项目名。以下示例从仓库根目录运行，`APP_PROJECT` 和 `MILVUS_PROJECT` 替换成实际项目名；将 `compose.env` 的路径替换为私有资源目录中的文件：

```bash
docker compose -p APP_PROJECT --env-file ../rag-v3-runtime/compose.env -f compose.full-isolated.yml ps
docker compose -p APP_PROJECT --env-file ../rag-v3-runtime/compose.env -f compose.full-isolated.yml stop
docker compose -p MILVUS_PROJECT --env-file ../rag-v3-runtime/compose.env -f compose.milvus-isolated.yml stop
# 重启时先恢复 Milvus，再恢复应用；复用现有配置和卷。
docker compose -p MILVUS_PROJECT --env-file ../rag-v3-runtime/compose.env -f compose.milvus-isolated.yml up -d --wait
docker compose -p APP_PROJECT --env-file ../rag-v3-runtime/compose.env -f compose.full-isolated.yml up -d --no-build --wait
```

在部署的 API 容器里可离线执行 `python -m unittest test_milvus_store test_model_fingerprint test_llm_budget test_container_identity`。带 `_integration` 的测试需先核对各自的独立实例、目录、认证和写入条件，避免指向正在使用的环境。历史评测原始资产不公开；验证其汇总或哈希需要自行提供获授权的对应版本。

## 既有数据恢复或真实模型

此节描述手工准备资源的部署方式，适用于已有数据；从空环境启动不需要这些步骤。

应用项目使用 `ragv3app20261106`。API 与 Streamlit 分别监听本机 18631 和 18632。独立 MySQL 与 Redis 通过命名卷持久化。Milvus 通过外部 Docker 网络 `ragv3milvus20261105_default` 接入已有的 `standalone:19530`，并使用该项目的 MinIO 与 etcd。

1. 先准备 Milvus 网络和后端制品。新环境可使用 `compose.milvus-isolated.yml`，通过单独的安全环境文件注入 MinIO 凭据。确认集合中的历史记录与 MySQL 快照绑定和制品回执一致。
2. 按 `full-isolated.env.example` 填写资源与模型路径。资源目录需包含 `runtime/store`、`runtime/HF_HOME`、`evaluation-history` 和受保护的 `secrets`。将历史评测批次按原路径放入 `evaluation-history`，供历史接口进行 Hash 核验。
3. 先启动专用 MySQL 与 Redis。仅向明确身份的新数据库恢复独立备份，核对表结构、UTC 下的行内容、快照与发布代次。运行时沿用已有 schema。将实测 MySQL UUID、Redis 实例标记、模型文件 SHA-256 和对应历史模型身份写入只读挂载的运行清单。实例校验要求见 `container_identity.py`。MySQL 和 Redis 的密码分别通过单独挂载的文件传入。
4. 使用 `Dockerfile.full` 和锁定依赖构建 CPU 镜像。BGE 与 CrossEncoder 通过只读挂载加载，使用受控的模型运行缓存。真实模型仅在操作者主动配置后使用：在私有运行清单中设置 `RAG_LLM_BUDGETED_REAL=1`、`RAG_V2_ISOLATED_LLM=0`、`DEEPSEEK_MODEL=deepseek-flash`、`DEEPSEEK_API_BASE=https://api.deepseek.com`、自己的 `DEEPSEEK_API_KEY`，并将 `RAG_LLM_COST_LEDGER` 指向挂载的 `/cost` 下的共享 JSON 账本（初始字段为 `currency: "CNY"`、`limit` 不高于 20、`requests: []`）。路由、图谱抽取和语义校验的独立配置分别见 `agent_router.py`、`graph_extraction.py` 和 `semantic_reliability.py`。额度和模型能力受提供方约束；上线前应重新核对费率与每条路径的配置。
5. 运行 `deployment/stack.ps1 -Action Start -EnvFile <本机配置路径>`，使用 `Status` 检查服务。停止时使用 `Stop`，该脚本保留持久卷和外部 Milvus 服务。

容器启动日志和 `/system/health` 可用于定位模型、数据库或快照初始化问题。管理员可通过 `/__deployment/identity` 核对实际实例和活动知识包。迁移及恢复前保存目标的数据库备份与原始快照制品，核对完整性后再切换发布指针。
