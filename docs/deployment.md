# Docker 本地部署与恢复

需要 Git、Python 3.11+、Docker Compose v2（支持 `up --wait`），以及足够的模型和镜像空间。模型通过只读挂载加载。每个目录提供配置、tokenizer 和一份权重文件，避免指向缓存的符号链接。以下示例从仓库根目录运行。

```bash
python -m pip install huggingface_hub
hf download BAAI/bge-small-zh-v1.5 --local-dir ../rag-models/embed --include "*.json" "*.txt" "model.safetensors"
hf download BAAI/bge-reranker-base --local-dir ../rag-models/rerank --include "*.json" "*.txt" "model.safetensors"
cp .env.example .env
python scripts/bootstrap.py --env-file .env --reviewer demo-reviewer
```

PowerShell 可用 `Copy-Item .env.example .env`。模板只包含路径。启动脚本读取指定配置文件，并生成私有运行清单、随机凭据和项目名。它要求运行目录为空的新路径。也可用 `--resource`、`--embed`、`--rerank` 指定路径。若端口冲突，用 `--api-port`、`--ui-port`、`--mysql-port`、`--redis-port`、`--milvus-port` 和 `--milvus-health-port` 指定空闲端口。

默认访问 [Streamlit](http://127.0.0.1:18632) 和 [API Swagger](http://127.0.0.1:18631/docs)。在数据中台上传 [合成政策](../examples/policy-demo.txt)，构建并发布首个快照。可以用 `ZX-V3-DEMO 退款需要什么？` 测试证据召回。空库时健康检查允许启动，发布后才有检索证据。默认生成是本地替身。

## 停启和持久化

从运行目录的 `projects.json` 获取两个实际项目名，以下用 `APP_PROJECT` 和 `MILVUS_PROJECT` 表示。先停应用，再停 Milvus。恢复时先启动 Milvus，再启动应用。两份 Compose 文件由同一个 bootstrap 管理。

```bash
docker compose -p APP_PROJECT --env-file ../rag-v3-runtime/compose.env -f deployment/compose.yml stop
docker compose -p MILVUS_PROJECT --env-file ../rag-v3-runtime/compose.env -f deployment/compose.milvus.yml stop
docker compose -p MILVUS_PROJECT --env-file ../rag-v3-runtime/compose.env -f deployment/compose.milvus.yml up -d --wait
docker compose -p APP_PROJECT --env-file ../rag-v3-runtime/compose.env -f deployment/compose.yml up -d --no-build --wait
```

MySQL、Redis、Milvus、etcd 和 MinIO 使用各自卷，原文与快照制品位于资源目录。已有数据恢复需要对齐 MySQL 发布指针、知识制品、Milvus 集合与模型摘要。初始化只对经身份验证的空库执行。正常重启复用现有目录和卷。具体运行边界见 [部署目录说明](../deployment/README.md)。

## 回归验证

```bash
docker build -f deployment/Dockerfile -t rag-v3-check .
docker run --rm --network none --mount "type=bind,source=${PWD}/tests,target=/app/tests,readonly" rag-v3-check python tests/run_offline.py
```

离线测试容器不使用部署凭据。依赖历史授权语料的测试只有在供应对应资产时执行，结果单列。使用新的隔离运行目录启动并设置 `--reviewer` 后，可执行 `python tests/integration/smoke_showcase.py <运行目录>`。该测试上传合成文档、审核图、发布、评测、回滚并重启本测试项目的 API，需要活动代次为零。其他集成用例有各自的资产和历史后端条件。
