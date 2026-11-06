CREATE TABLE IF NOT EXISTS kg_builds (
 snapshot_id CHAR(32) PRIMARY KEY, build_id CHAR(32) NOT NULL UNIQUE, corpus_id CHAR(64) NOT NULL,
 task_status VARCHAR(20) NOT NULL, total_chunks INT NOT NULL DEFAULT 0, processed_chunks INT NOT NULL DEFAULT 0,
 extracted_entities INT NOT NULL DEFAULT 0, extracted_relations INT NOT NULL DEFAULT 0,
 failed_chunks INT NOT NULL DEFAULT 0, error_message VARCHAR(255),
 started_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6), completed_at DATETIME(6),
 FOREIGN KEY(snapshot_id) REFERENCES rag_v2_snapshots(snapshot_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS kg_entities (
 snapshot_id CHAR(32) NOT NULL, entity_id CHAR(64) NOT NULL, entity_type VARCHAR(32) NOT NULL,
 entity_name VARCHAR(200) NOT NULL, normalized_name VARCHAR(200) NOT NULL, description TEXT,
 status VARCHAR(20) NOT NULL DEFAULT 'CANDIDATE', created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
 PRIMARY KEY(snapshot_id,entity_id), FOREIGN KEY(snapshot_id) REFERENCES kg_builds(snapshot_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS kg_entity_aliases (
 snapshot_id CHAR(32) NOT NULL, entity_id CHAR(64) NOT NULL, alias VARCHAR(200) NOT NULL,
 PRIMARY KEY(snapshot_id,alias,entity_id),
 FOREIGN KEY(snapshot_id,entity_id) REFERENCES kg_entities(snapshot_id,entity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS kg_relations (
 snapshot_id CHAR(32) NOT NULL, relation_id CHAR(64) NOT NULL,
 source_entity_id CHAR(64) NOT NULL, target_entity_id CHAR(64) NOT NULL, relation_type VARCHAR(32) NOT NULL,
 description TEXT, status VARCHAR(20) NOT NULL DEFAULT 'CANDIDATE', extraction_method VARCHAR(32) NOT NULL,
 valid_from DATE, valid_to DATE, created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
 PRIMARY KEY(snapshot_id,relation_id),
 INDEX outbound(snapshot_id,source_entity_id,status,relation_id),
 INDEX inbound(snapshot_id,target_entity_id,status,relation_id),
 FOREIGN KEY(snapshot_id,source_entity_id) REFERENCES kg_entities(snapshot_id,entity_id),
 FOREIGN KEY(snapshot_id,target_entity_id) REFERENCES kg_entities(snapshot_id,entity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS kg_relation_evidence (
 snapshot_id CHAR(32) NOT NULL, relation_id CHAR(64) NOT NULL, evidence_id CHAR(64) NOT NULL,
 document_id CHAR(32) NOT NULL, document_version CHAR(32) NOT NULL, file_hash CHAR(32) NOT NULL,
 chunk_id VARCHAR(80) NOT NULL, parent_id VARCHAR(80) NOT NULL, source_position JSON NOT NULL, evidence_text TEXT NOT NULL,
 PRIMARY KEY(snapshot_id,relation_id,evidence_id),
 FOREIGN KEY(snapshot_id,relation_id) REFERENCES kg_relations(snapshot_id,relation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS kg_reviews (
 review_id BIGINT AUTO_INCREMENT PRIMARY KEY, snapshot_id CHAR(32) NOT NULL, entity_kind VARCHAR(12) NOT NULL,
 object_id CHAR(64) NOT NULL, previous_status VARCHAR(20) NOT NULL, status VARCHAR(20) NOT NULL,
 actor VARCHAR(100) NOT NULL, note TEXT NOT NULL, created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
 FOREIGN KEY(snapshot_id) REFERENCES kg_builds(snapshot_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
