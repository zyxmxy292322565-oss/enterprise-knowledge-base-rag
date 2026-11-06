CREATE TABLE IF NOT EXISTS kg_inheritance_runs (
 run_id CHAR(64) PRIMARY KEY, source_snapshot CHAR(32) NOT NULL, target_snapshot CHAR(32) NOT NULL,
 actor VARCHAR(100) NOT NULL, state VARCHAR(20) NOT NULL, metrics JSON NOT NULL,
 created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
 UNIQUE KEY inherited_pair(source_snapshot,target_snapshot)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS kg_inheritance_bindings (
 run_id CHAR(64) NOT NULL, source_evidence_id CHAR(64) NOT NULL, target_evidence_id CHAR(64),
 status VARCHAR(20) NOT NULL, details JSON NOT NULL,
 PRIMARY KEY(run_id,source_evidence_id), FOREIGN KEY(run_id) REFERENCES kg_inheritance_runs(run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS kg_inheritance_objects (
 run_id CHAR(64) NOT NULL, kind VARCHAR(12) NOT NULL, source_object CHAR(64) NOT NULL,
 target_object CHAR(64), status VARCHAR(20) NOT NULL, source_review_ids JSON NOT NULL,
 review_id BIGINT NOT NULL, details JSON NOT NULL,
 PRIMARY KEY(run_id,kind,source_object), FOREIGN KEY(run_id) REFERENCES kg_inheritance_runs(run_id),
 FOREIGN KEY(review_id) REFERENCES kg_reviews(review_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS kg_policy_revocations (
 event_id CHAR(32) PRIMARY KEY, snapshot_id CHAR(32) NOT NULL, relation_id CHAR(64) NOT NULL,
 effective_at DATETIME(6) NOT NULL, actor VARCHAR(100) NOT NULL, reason TEXT NOT NULL,
 created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6), INDEX policy_asof(snapshot_id,relation_id,effective_at),
 FOREIGN KEY(snapshot_id,relation_id) REFERENCES kg_policy_versions(snapshot_id,relation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS rag_repair_migrations (
 version VARCHAR(80) PRIMARY KEY, checksum CHAR(64) NOT NULL,
 applied_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
