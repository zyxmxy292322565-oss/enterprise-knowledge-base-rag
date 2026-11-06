CREATE TABLE IF NOT EXISTS kg_extraction_runs (
 snapshot_id CHAR(32) PRIMARY KEY, mode VARCHAR(16) NOT NULL, model VARCHAR(150) NOT NULL,
 prompt_version VARCHAR(32) NOT NULL, schema_version VARCHAR(32) NOT NULL,
 FOREIGN KEY(snapshot_id) REFERENCES kg_builds(snapshot_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS kg_extraction_chunks (
 snapshot_id CHAR(32) NOT NULL, child_id VARCHAR(80) NOT NULL, status VARCHAR(16) NOT NULL,
 attempts INT NOT NULL, result JSON, error_type VARCHAR(80), updated_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
 PRIMARY KEY(snapshot_id,child_id), FOREIGN KEY(snapshot_id) REFERENCES kg_builds(snapshot_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS kg_relation_semantics (
 snapshot_id CHAR(32) NOT NULL, relation_id CHAR(64) NOT NULL, polarity VARCHAR(16) NOT NULL,
 conditions_json JSON NOT NULL, exceptions_json JSON NOT NULL,
 PRIMARY KEY(snapshot_id,relation_id),
 FOREIGN KEY(snapshot_id,relation_id) REFERENCES kg_relations(snapshot_id,relation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
