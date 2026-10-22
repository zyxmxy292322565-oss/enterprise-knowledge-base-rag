# 隔离 RAG 评测基础设施

本目录是独立评测配置，不代表现有 `cache/` 与智能客服知识库属于不同数据。正式 Policy RAG 当前使用项目 `cache/`；不要把它挂载或复制到评测实例。评测环境初始索引为空，须显式选择文档来源。

## 启动前检查

1. 使用 `docker compose -p rag-evaluation -f evaluation/docker-compose.yml config` 检查项目名、挂载和目标；检查输出可能包含密钥，不要发布完整输出。
2. 确认 `/app/cache` 是 `rag-evaluation` 项目专用的 `eval_index` 命名卷并设置 `nocopy`；MySQL、Redis、模型缓存均是专用卷；数据库与 Redis 无宿主端口。
3. 确认评测端口未被占用；未配置的必需凭据将阻止启动。凭据使用独立环境变量 `EVAL_MYSQL_ROOT_PASSWORD`、`EVAL_MYSQL_PASSWORD`、`DEEPSEEK_API_KEY`，不要写入源码或清单。
4. 只在确认隔离后使用明确的 `-p rag-evaluation -f evaluation/docker-compose.yml` 构建及启动。不要对正式项目执行 down、重建或卷删除。

镜像使用 Dockerfile 专用 ignore 文件排除现有缓存与环境文件；评测入口在导入业务代码前检查环境及路径。路径和主机检查不能单独证明 Docker 挂载来源，启动前仍必须检查实际 Compose 配置和卷。

## 文档快照

`python evaluation/snapshot_documents.py <明确选择的文档目录> <快照ID>` 将 TXT/PDF 复制到 `evaluation/snapshots/documents/<快照ID>`，记录文件 MD5 和 SHA-256。ID 不可覆盖；不可使用正式 `cache` 作为来源。请明确核实语料归属，不要默认独立 RAG 和 Policy RAG 知识库完全相同。此命令不会上传文档、建立索引或生成评测集。

评测服务的 `POST /evaluation/documents` 仅上传到评测实例。实际索引由该实例的现有上传流程在专用缓存卷建立。上传之前应核验快照清单；当前还没有自动将快照清单与上传结果绑定的功能。

## 评测接口

- `POST /evaluation/experiments`：提供 `dataset_version`，返回实验 ID。
- `POST /evaluation/experiments/{experiment_id}/runs`：创建关联 run。
- `POST /evaluation/runs/{run_id}/cases`：提供 `case_id`、问题、`chain` (`ask` 或 `policy`)、参数；默认缓存模式 `bypass`，保存逐题记录。
- `GET /evaluation/cases/{case_run_id}`：读取逐题记录。

`candidate_k` 与 `final_k` 独立；关闭 rerank 时使用 FAISS 原顺序，且跳过 rerank 分数阈值，不应把此结果当成同一阈值的对照。评测入口只支持 `bypass`，不会访问 Redis 缓存或正式问答日志；缓存命中性能实验尚未实现。记录包含候选、排序、Context、响应和耗时，内容可能敏感，应限制数据库访问。`dataset_version` 是调用方提供的标签，当前不会自动验证其对应的冻结数据集。模型权重 revision 当前标记为未验证。Policy 链路不生成回答，Context/Prompt 不适用。

不要将本基础设施输出或关键词得分当作准确率；人工审核和正式数据集属于后续阶段。
