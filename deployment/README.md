# RAG V3 隔离部署

应用项目使用 `ragv3app20261106`。API 与 Streamlit 分别监听本机 18631 和 18632。独立 MySQL 与 Redis 通过命名卷持久化。Milvus 通过外部 Docker 网络 `ragv3milvus20261105_default` 接入已有的 `standalone:19530`，并使用该项目的 MinIO 与 etcd。

1. 先准备 Milvus 网络和后端制品。新环境可使用 `compose.milvus-isolated.yml`，通过单独的安全环境文件注入 MinIO 凭据。确认集合中的历史记录与 MySQL 快照绑定和制品回执一致。
2. 按 `full-isolated.env.example` 填写资源与模型路径。资源目录需包含 `runtime/store`、`runtime/HF_HOME`、`evaluation-history` 和受保护的 `secrets`。将历史评测批次按原路径放入 `evaluation-history`，供历史接口进行 Hash 核验。
3. 先启动专用 MySQL 与 Redis。仅向明确身份的新数据库恢复独立备份，核对表结构、UTC 下的行内容、快照与发布代次。运行时沿用已有 schema。将实测 MySQL UUID、Redis 实例标记、模型文件 SHA-256 和对应历史模型身份写入只读挂载的运行清单。实例校验要求见 `container_identity.py`。MySQL 和 Redis 的密码分别通过单独挂载的文件传入。
4. 使用 `Dockerfile.full` 和锁定依赖构建 CPU 镜像。BGE 与 CrossEncoder 通过只读挂载加载，使用受控的模型运行缓存。真实 LLM 凭据只放在受保护的运行配置中，启用真实模型前准备共享费用账本并限制并行付费入口。
5. 运行 `deployment/stack.ps1 -Action Start -EnvFile <本机配置路径>`，使用 `Status` 检查服务。停止时使用 `Stop`，该脚本保留持久卷和外部 Milvus 服务。

容器启动日志和 `/system/health` 可用于定位模型、数据库或快照初始化问题。管理员可通过 `/__deployment/identity` 核对实际实例和活动知识包。迁移及恢复前保存目标的数据库备份与原始快照制品，核对完整性后再切换发布指针。
