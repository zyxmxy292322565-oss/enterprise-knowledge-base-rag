-- Phase 2. Apply only to the identity-verified development database.
CREATE TABLE IF NOT EXISTS rag_eval_center_datasets (
    dataset_id CHAR(32) NOT NULL,
    dataset_version VARCHAR(120) COLLATE utf8mb4_bin NOT NULL,
    split VARCHAR(16) NOT NULL,
    frozen BOOLEAN NOT NULL DEFAULT FALSE,
    body JSON NOT NULL,
    created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (dataset_id,dataset_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS rag_eval_center_runs (
    run_id CHAR(32) PRIMARY KEY,
    dataset_id CHAR(32) NOT NULL,
    dataset_version VARCHAR(120) COLLATE utf8mb4_bin NOT NULL,
    state VARCHAR(20) NOT NULL,
    worker_id CHAR(32) NULL,
    total INT NOT NULL,
    completed INT NOT NULL DEFAULT 0,
    config JSON NOT NULL,
    inputs JSON NOT NULL,
    error_message VARCHAR(255) NULL,
    started_at DATETIME(6) NULL,
    finished_at DATETIME(6) NULL,
    heartbeat_at DATETIME(6) NULL,
    created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
    FOREIGN KEY (dataset_id,dataset_version) REFERENCES rag_eval_center_datasets(dataset_id,dataset_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS rag_eval_center_cases (
    run_id CHAR(32) NOT NULL,
    question_id VARCHAR(120) COLLATE utf8mb4_bin NOT NULL,
    body JSON NOT NULL,
    PRIMARY KEY (run_id,question_id),
    FOREIGN KEY (run_id) REFERENCES rag_eval_center_runs(run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS rag_eval_center_audit (
    audit_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    scope_id CHAR(32) NOT NULL,
    version VARCHAR(120) COLLATE utf8mb4_bin NOT NULL,
    entity_id VARCHAR(120) COLLATE utf8mb4_bin NOT NULL,
    event JSON NOT NULL,
    created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
    INDEX audit_scope (scope_id,version,entity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
