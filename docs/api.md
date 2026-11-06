# API

运行后以 FastAPI `/docs` 和 `/openapi.json` 为参数与响应的准确来源。以下路径的 `/v2` 是现有接口版本，当前运行环境是 RAG V3。

| 用途 | 方法与路径 | 要点 |
| --- | --- | --- |
| 查看与上传文档 | `GET /v2/documents` · `POST /v2/documents` | 上传使用 multipart，字段 `file` |
| 构建与核对快照 | `POST /v2/index-snapshots` · `GET /v2/index-snapshots/{snapshot_id}/chunks` | 可核对 Child、Parent 与来源 |
| 发布与回滚 | `POST /v2/index-snapshots/{snapshot_id}/publish` · `/rollback` | JSON 包含 `expected_generation` 与 `idempotency_key` |
| 政策证据与普通问答 | `POST /rag/query` · `POST /ask` | 均接受 `question`，可选 `retrieval`。前者返回证据，后者生成答案 |
| 图谱 | `POST /v3/graph/{sid}/build` · `/review` · `/seal` · `GET /v3/graph/lookup` | 评审需图谱评审凭据，查询可传 `question` |
| 调试与可靠性 | `POST /v2/retrieval/debug` · `POST /v2/retrieval/reliable` | 前者输出检索诊断，后者可选择规则或语义 |
| 读取 Trace | `GET /v2/retrieval/reliable/{trace_id}` | 受服务端认证主体与过期限制 |
| Gold 和评测 | `/v2/evaluation/datasets` · `/v2/evaluation/runs` | 审核、冻结、绑定快照、启动、查看和比较 |

管理入口通过 `X-RAG-V2-Token` 授权。图谱审核使用独立 `X-KG-Reviewer`，Trace 支持服务端配置的 Bearer 主体。评测角色和 holdout 有独立授权。部署生成的配置存于私有运行目录。
