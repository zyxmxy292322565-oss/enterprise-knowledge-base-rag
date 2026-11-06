CREATE TABLE IF NOT EXISTS kg_entity_merges (
 snapshot_id CHAR(32) NOT NULL, operation_id CHAR(32) NOT NULL,
 source_id CHAR(64) NOT NULL, target_id CHAR(64) NOT NULL, actor VARCHAR(100) NOT NULL,
 reason TEXT NOT NULL, created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
 undone_at DATETIME(6), undone_by VARCHAR(100), undo_reason TEXT,
 PRIMARY KEY(snapshot_id,operation_id),
 INDEX lookup_merge(snapshot_id,undone_at,source_id),
 FOREIGN KEY(snapshot_id,source_id) REFERENCES kg_entities(snapshot_id,entity_id),
 FOREIGN KEY(snapshot_id,target_id) REFERENCES kg_entities(snapshot_id,entity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS kg_policy_versions (
 snapshot_id CHAR(32) NOT NULL, relation_id CHAR(64) NOT NULL,
 policy_id VARCHAR(100) NOT NULL, policy_version VARCHAR(80) NOT NULL,
 policy_entity_id CHAR(64) NOT NULL, evidence_id CHAR(64) NOT NULL, metadata_json JSON NOT NULL,
 PRIMARY KEY(snapshot_id,relation_id),
 INDEX policy_history(snapshot_id,policy_id,policy_version),
 FOREIGN KEY(snapshot_id,relation_id,evidence_id) REFERENCES kg_relation_evidence(snapshot_id,relation_id,evidence_id),
 FOREIGN KEY(snapshot_id,policy_entity_id) REFERENCES kg_entities(snapshot_id,entity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
