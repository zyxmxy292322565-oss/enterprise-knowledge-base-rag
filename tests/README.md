# 测试

- `unit/` 测算法、数据结构、分块、坐标、模型绑定和预算。
- `regression/` 测 API 契约、降级、历史快照、缓存与隔离边界，使用临时文件或替身。
- `integration/` 需要新的独立服务实例。`smoke_showcase.py` 通过真实 HTTP、MySQL、Redis 和 Milvus 完成合成知识包工作流。历史 FAISS 测试需要显式准备旧后端或授权评测资产，均保留原断言。
- `run_offline.py` 在无网络的临时容器中执行 unit 和 regression。命令见 [部署文档](../docs/deployment.md#回归验证)。
