# V3 政策草案 RAG 参数对比实验报告

## 1. 结论

本次在独立 Docker 评测环境中，对 50 份 V3 政策草案形成的 100 个 Chunk 和同一套 66 道评测题执行了 4 组检索实验，共产生 264 条逐题记录，失败请求为 0。所有请求均使用 `cache_mode=bypass`，没有从回答缓存读取结果，也没有向 Redis 写入结果。

在本次数据和指标口径下，**B（`candidate_k=10`、`final_k=5`、关闭 Rerank）表现最好**：Evidence Hit@5 为 **90.91%**，宏平均来源 Recall@5 为 **75.51%**，MRR 为 **0.7321**，NDCG@5 为 **0.6735**。相对 A（10→3、关闭 Rerank），Evidence Hit 提高 **7.58 个百分点**，宏平均来源 Recall 提高 **10.10 个百分点**。

当前 Rerank 配置没有改善整体召回。C 相对 A 的 Evidence Hit 从 83.33% 降至 78.79%，D 相对 B 从 90.91% 降至 81.82%；Rerank 平均阶段耗时约 1.62–1.67 秒。该结果说明 `BAAI/bge-reranker-base` 与当前 `min_rerank_score=0.2` 的组合需要进一步校准，不能据此声称“开启精排必然提升质量”。

## 2. 实验对象与可追溯标识

- 实验 ID：`8b5516f5-327c-4329-b85f-af578f44c6ba`
- 评测集：`policy-docs-v3-draft-20261019`，66 题
- 评测集 SHA-256：`3475e076245c6aa628e6baab42f79497ba281817339caf9758711df9ec0695fe`
- 政策文档：50 份 V3 草案；政策 Manifest SHA-256：`d7bf7724a411ceff8f98032220b3a95220b315216dbee2e58178acb6ddac92fc`
- 文档快照 ID：`policy_v3_draft_20261019`
- 快照逻辑 Manifest SHA-256：`e3682a70b47aacc038c8b2d9339aa5074d301cf9ad7d70bb9ecb906cd994946a`
- 索引规模：100 Chunk
- Embedding：`BAAI/bge-small-zh-v1.5`
- Rerank：`BAAI/bge-reranker-base`
- 源码文件集合 SHA-256：`5985239099dce583486a329311f8b4bea0585677c4375b0849f9abbbc4c2f54b`
- 索引 Manifest SHA-256：`4f410f0527f80fb5278f7e54ae35621e598e649690b8ba187313109a86982155`
- FAISS 索引文件 SHA-256：`ad4949a69e87c9bbf300dad53097e3781b6a1147eb5a3600139279e1ed7f443e`
- 模型权重 revision：未固定/未验证；实际 Python 包版本和文件哈希见 `reproducibility_manifest.json`

## 3. 隔离环境证据

评测服务、MySQL 和 Redis 均属于 Compose 项目 `rag-evaluation`，只连接网络 `rag-evaluation_default`：

- RAG：`rag-evaluation-rag-1`，仅映射 `127.0.0.1:18080`；`/app/cache` 挂载独立卷 `rag-evaluation_eval_index`。
- MySQL：`rag-evaluation-mysql-1`，使用 `rag-evaluation_eval_mysql`，没有宿主机端口。
- Redis：`rag-evaluation-redis-1`，使用 `rag-evaluation_eval_redis`，没有宿主机端口。
- 模型缓存：`rag-evaluation_eval_model_cache`。
- 没有 bind mount 到项目共享 `cache`，没有使用正式 MySQL/Redis 卷。

实验后评测 Redis `DBSIZE` 为 0。评测 MySQL 中实际有 1 个 experiment、4 个 run、264 个 case run；每个 run 恰好 66 条。智能客服 `app` 项目的 Backend、Frontend、Admin Workbench、Policy RAG、Redis、MySQL 在实验结束后仍均为 `healthy`。

## 4. 参数矩阵与真实语义

| 配置 | Run ID | candidate_k | final_k | Rerank | min_rerank_score |
|---|---|---:|---:|---|---:|
| A | `6df27ce9-4c95-4d57-806c-eb7fbc76824d` | 10 | 3 | 关闭 | 0.2（关闭时不生效） |
| B | `949426ba-7169-44f1-b776-b0f92b59f459` | 10 | 5 | 关闭 | 0.2（关闭时不生效） |
| C | `5c91347b-5b26-457b-a983-b7e05068b670` | 10 | 3 | 开启 | 0.2 |
| D | `b63a0dea-1b5b-4076-bc8b-0fccb375e4fd` | 10 | 5 | 开启 | 0.2 |

源码中 `candidate_k` 决定 FAISS 初筛数量，`final_k` 决定最终证据 Chunk 数量；显式提供 `candidate_k` 后，`faiss_multiplier` 不再改变候选数。Rerank 关闭时保持 FAISS 距离排序且跳过分数阈值；Rerank 开启时使用 CrossEncoder 重新排序，并在最高分低于 0.2 时拒绝返回证据。因此 A/C、B/D 同时反映了“排序 + 0.2 阈值”的组合效果，尚不能把差异完全归因于排序模型本身。

66/66 道题在 A、B、C、D 中拥有完全一致的 10 条初始候选（按 FAISS 索引号、Chunk ID、距离、来源逐项比较），控制变量成立。统一的候选 Hit@10 为 95.45%，候选宏平均来源 Recall@10 为 79.80%。

## 5. 指标口径

- **Evidence Hit@K**：最终返回的 K 个 Chunk 中，至少有一个 Chunk 的来源文件属于该题 `reference_sources`。
- **宏平均来源 Recall@K**：逐题计算“最终结果覆盖的唯一预期来源数 / 该题预期来源总数”，再对 66 题取平均。
- **MRR**：最终结果中首个来自预期来源的 Chunk 的倒数排名；未命中记 0。
- **NDCG@K**：按预期来源进行二元相关性计算。同一来源的多个 Chunk 只让第一次出现获得增益，避免重复 Chunk 被重复计分。
- **耗时**：服务端阶段计时来自每次 API 结果的 `timing`；客户端墙钟包含 HTTP、序列化和 MySQL 逐题持久化时间。P95 使用线性插值分位数。
- **失败**：任何请求或执行错误都应保留并计入失败。本次 264 次均成功持久化；低相关拒绝属于有效业务状态，不算请求失败。

这些指标属于**来源文件级证据指标**。评测集尚无冻结索引下的精确 Chunk 标签，因此 NDCG 和 Recall 不能解释为精确段落级相关性。

## 6. 四组真实结果

| 配置 | Hit@K | 宏 Recall@K | MRR | NDCG@K | 服务端平均 | 中位数 | P95 | 请求失败 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 83.33% | 65.40% | 0.7146 | 0.6238 | 18.26 ms | 17.53 ms | 25.41 ms | 0 |
| B | **90.91%** | **75.51%** | **0.7321** | **0.6735** | 15.22 ms | 15.71 ms | 20.04 ms | 0 |
| C | 78.79% | 60.86% | 0.7096 | 0.6403 | 1628.28 ms | 1625.18 ms | 1739.90 ms | 0 |
| D | 81.82% | 64.39% | 0.7172 | 0.6599 | 1677.69 ms | 1675.91 ms | 1779.08 ms | 0 |

客户端平均墙钟耗时分别为 A 85.52 ms、B 78.60 ms、C 1696.40 ms、D 1748.01 ms。实验按 A→B→C→D 顺序单次运行，没有随机化或重复轮次，因此 A 与 B 之间约 3 ms 的服务端差异可能来自模型预热和系统噪声，不应写成性能优化成果。Rerank 带来的约 1.6 秒量级差异足够大，但仍只代表本机 CPU 容器条件。

### A → B：仅增加 final_k

- Hit：83.33% → 90.91%，**+7.58 个百分点**，相对提高 9.09%。
- 宏 Recall：65.40% → 75.51%，**+10.10 个百分点**，相对提高 15.44%。
- MRR：0.7146 → 0.7321，绝对增加 0.0174。
- NDCG：0.6238 → 0.6735，绝对增加 0.0496。

这说明在当前 100-Chunk 小型语料上，保留 5 条证据比保留 3 条更能覆盖人工标注的来源。是否会给生成阶段带来噪声，需要真实回答实验另行验证。

### Rerank 排名变化

在 63 道初筛已包含预期来源的题中，Rerank 使首个相关来源排名上升 17 题、不变 34 题、下降 12 题；平均上升 0.19 位，中位数为 0。尽管部分题排名改善，0.2 阈值使 C/D 各有 3 题被拒绝，整体 Hit 和 Recall 反而低于对应的无 Rerank 配置。

被拒绝的 3 题为：

- `PDV2-C014`：部分可回答的第 7 天精确截止时点问题；
- `PDV2-C055`：可回答的“Rerank 第一是否代表政策效力最高”错误前提题；
- `PDV2-C059`：知识库无统一一年维修期依据的问题。

严格无答案题共 8 道。C/D 只拒绝其中 1 道，严格无答案拒绝率为 12.50%；同时错误拒绝 1/52 道可回答题（1.92%）。A/B 在关闭 Rerank 时不执行低分阈值，因此拒绝率为 0。这套参数尚未形成可靠的无答案拒绝能力。

## 7. 分题型观察

B 的完整分题型结果见 `question_type_metrics.csv`。部分重叠题型如下：

| 题型 | 题数 | Hit@5 | 宏 Recall@5 |
|---|---:|---:|---:|
| 单文档事实题 | 20 | 90.00% | 90.00% |
| 跨文档问题 | 39 | 89.74% | 63.68% |
| 多条件判断 | 25 | 92.00% | 77.33% |
| 相似概念辨析 | 18 | 88.89% | 72.22% |
| 商品类别容易混淆 | 18 | 100.00% | 78.70% |
| 文档没有依据、应该拒答 | 8 | 100.00% | 77.08% |

题型标签可重叠，不能相加得到 66。无答案题 Hit 高只说明检索到了用于判断“缺少依据”的关联政策，并不等于系统已正确拒答。

## 8. 缓存与持久化验证

- 264/264 条记录的实际生效参数均为 `cache_mode=bypass`。
- 258 条正常证据结果明确记录 `cache_hit=false`；6 条低相关拒绝结果未输出 `cache_hit` 字段，但同样记录旁路模式，且没有 Redis 阶段耗时。
- 实验结束后独立 Redis `DBSIZE=0`，证明实验没有留下回答缓存键。
- 264 个 `case_run_id` 全部唯一；4 个 `run_id` 各关联 66 条记录。
- 每题都保存了 FAISS 初筛候选、Rerank 排序、最终证据、分数、参数、索引哈希和分阶段耗时。

## 9. 最终回答质量实验状态

本轮没有运行真实 LLM 最终回答实验。隔离容器只有明确的 retrieval-only 占位凭据，没有经过授权的真实付费 API Key。比较基线 A 与当前最佳 B 需要 **132 次 LLM 调用**（66×2），费用将由所选模型供应商账户承担；在获得模型、价格和费用授权前，不应调用外部服务。

因此本报告没有关键词命中率、必需事实覆盖率、Faithfulness 或生成耗时数据，也没有用模拟答案冒充真实结果。现有结论只能支持“检索证据覆盖”的简历表述，不能支持“最终回答准确率提升”。

## 10. 主要限制

1. 50 份政策和 66 道题仍是草案；49 题标记为 `needs_human_review`，17 题为证据已核对但政策仍待批准。
2. 相关性标签是来源文件级，缺少冻结 Chunk ID 级标注。
3. Rerank 开关同时改变了排序和 0.2 阈值行为，尚未进行阈值消融实验。
4. 每个配置只运行一次且顺序固定；小幅耗时差异不具备统计显著性。
5. 模型名称已记录，但 Hugging Face 权重 revision 未固定。
6. 语料仅 100 Chunk，不能外推到大规模知识库。
7. 没有真实 LLM 回答质量、Token 或成本指标。

## 11. 产物

- `experiment_results.csv`：264 条逐题结果。
- `experiment_summary.csv`：四组汇总指标。
- `raw_case_records.jsonl`：评测 API 返回的完整候选、排序、证据和 Manifest。
- `question_type_metrics.csv`：重叠题型指标。
- `reproducibility_manifest.json`：源码、数据、索引、模型、容器和卷的可追溯信息。
- `computed_metrics.json` 与 `analyze_retrieval.py`：指标计算结果和复算脚本。
