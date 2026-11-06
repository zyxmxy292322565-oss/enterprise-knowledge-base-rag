-- Apply explicitly to the verified, isolated rag_v2_* schema.
CREATE TABLE IF NOT EXISTS rag_v2_head (
    id TINYINT PRIMARY KEY,
    snapshot_id CHAR(32) NULL,
    generation BIGINT NOT NULL DEFAULT 0,
    draft_revision BIGINT NOT NULL DEFAULT 0
) ENGINE=InnoDB;

INSERT IGNORE INTO rag_v2_head (id) VALUES (1);

CREATE TABLE IF NOT EXISTS rag_v2_documents (
    document_id CHAR(32) PRIMARY KEY,
    source VARCHAR(255) COLLATE utf8mb4_bin NOT NULL UNIQUE,
    current_version_id CHAR(32) NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS rag_v2_document_versions (
    document_version_id CHAR(32) PRIMARY KEY,
    document_id CHAR(32) NOT NULL,
    revision BIGINT NOT NULL,
    content_hash CHAR(64) NOT NULL,
    legacy_file_hash CHAR(32) NOT NULL,
    policy_version VARCHAR(64) NULL,
    category VARCHAR(64) NULL,
    origin VARCHAR(255) NULL,
    document_status VARCHAR(16) NOT NULL,
    processed_sha CHAR(64) NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    UNIQUE KEY (document_id, revision),
    FOREIGN KEY (document_id) REFERENCES rag_v2_documents(document_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS rag_v2_snapshots (
    snapshot_id CHAR(32) PRIMARY KEY,
    corpus_snapshot_id CHAR(64) NOT NULL,
    draft_revision BIGINT NOT NULL,
    base_generation BIGINT NOT NULL,
    build_status VARCHAR(16) NOT NULL,
    validation_status VARCHAR(16) NOT NULL,
    manifest_sha CHAR(64) NULL,
    error_message VARCHAR(255) NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS rag_v2_publications (
    publish_generation BIGINT PRIMARY KEY,
    idempotency_key VARCHAR(120) COLLATE utf8mb4_bin NOT NULL UNIQUE,
    previous_snapshot_id CHAR(32) NULL,
    snapshot_id CHAR(32) NOT NULL,
    expected_generation BIGINT NOT NULL,
    operation VARCHAR(12) NOT NULL,
    task_status VARCHAR(16) NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    FOREIGN KEY (snapshot_id) REFERENCES rag_v2_snapshots(snapshot_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS rag_v2_publication_tasks (
    idempotency_key VARCHAR(120) COLLATE utf8mb4_bin PRIMARY KEY,
    snapshot_id CHAR(32) NOT NULL,
    expected_generation BIGINT NOT NULL,
    operation VARCHAR(12) NOT NULL,
    task_status VARCHAR(16) NOT NULL,
    publish_generation BIGINT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    FOREIGN KEY (publish_generation) REFERENCES rag_v2_publications(publish_generation)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
