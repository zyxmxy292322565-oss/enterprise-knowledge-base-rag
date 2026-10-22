-- RAG evaluation tracking v2 (REVIEW BEFORE EXECUTION)
-- Target: the independent RAG rag_demo database only.
-- This file is intentionally not executed by application startup.
-- MySQL DDL auto-commits; take and verify a restorable backup first.

ALTER TABLE rag_eval_results
    ADD COLUMN cache_status VARCHAR(16) NULL AFTER from_cache,
    ADD CONSTRAINT chk_rag_eval_results_cache_status
        CHECK (cache_status IS NULL OR cache_status IN ('bypass', 'hit', 'miss'));

ALTER TABLE rag_eval_experiments
    ADD COLUMN experiment_goal TEXT NULL AFTER experiment_name,
    ADD COLUMN comparison_variable VARCHAR(255) NULL AFTER experiment_goal,
    ADD COLUMN dataset_version VARCHAR(255) NULL AFTER comparison_variable,
    ADD COLUMN dataset_manifest_sha256 CHAR(64) NULL AFTER dataset_version,
    ADD COLUMN planned_parameters_json JSON NULL AFTER dataset_manifest_sha256,
    ADD COLUMN latest_run_id CHAR(36) NULL AFTER planned_parameters_json;

CREATE TABLE IF NOT EXISTS rag_eval_runs (
    id CHAR(36) PRIMARY KEY,
    experiment_id INT NOT NULL,
    status VARCHAR(32) NOT NULL,
    cache_mode VARCHAR(16) NOT NULL,
    requested_parameters_json JSON NOT NULL,
    effective_parameters_json JSON NULL,
    dataset_version VARCHAR(255) NOT NULL,
    dataset_manifest_sha256 CHAR(64) NOT NULL,
    source_files_sha256 CHAR(64) NOT NULL,
    git_commit VARCHAR(64) NULL,
    document_manifest_sha256 CHAR(64) NOT NULL,
    document_manifest_json JSON NOT NULL,
    index_manifest_sha256 CHAR(64) NOT NULL,
    index_manifest_json JSON NOT NULL,
    model_manifest_json JSON NOT NULL,
    prompt_code_sha256 CHAR(64) NOT NULL,
    environment_manifest_json JSON NOT NULL,
    total_cases INT NOT NULL DEFAULT 0,
    completed_cases INT NOT NULL DEFAULT 0,
    success_count INT NOT NULL DEFAULT 0,
    failed_count INT NOT NULL DEFAULT 0,
    error_type VARCHAR(255) NULL,
    started_at DATETIME(6) NOT NULL,
    ended_at DATETIME(6) NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_rag_eval_runs_cache_mode
        CHECK (cache_mode IN ('bypass', 'normal')),
    CONSTRAINT fk_rag_eval_runs_experiment
        FOREIGN KEY (experiment_id) REFERENCES rag_eval_experiments(id),
    INDEX idx_rag_eval_runs_experiment_started (experiment_id, started_at),
    INDEX idx_rag_eval_runs_status_started (status, started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS rag_eval_case_runs (
    id CHAR(36) PRIMARY KEY,
    run_id CHAR(36) NOT NULL,
    case_id INT NOT NULL,
    eval_result_id INT NULL,
    status VARCHAR(64) NOT NULL,
    question TEXT NOT NULL,
    cache_status VARCHAR(16) NOT NULL,
    effective_parameters_json JSON NULL,
    retrieval_trace_json JSON NULL,
    final_answer LONGTEXT NULL,
    scoring_json JSON NULL,
    timing_json JSON NULL,
    refusal_reason VARCHAR(255) NULL,
    error_stage VARCHAR(100) NULL,
    error_type VARCHAR(255) NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_rag_eval_case_runs_cache_status
        CHECK (cache_status IN ('bypass', 'hit', 'miss', 'unknown')),
    CONSTRAINT fk_rag_eval_case_runs_run
        FOREIGN KEY (run_id) REFERENCES rag_eval_runs(id),
    CONSTRAINT fk_rag_eval_case_runs_case
        FOREIGN KEY (case_id) REFERENCES rag_eval_cases(id),
    CONSTRAINT fk_rag_eval_case_runs_result
        FOREIGN KEY (eval_result_id) REFERENCES rag_eval_results(id),
    CONSTRAINT uq_rag_eval_case_run UNIQUE (run_id, case_id),
    CONSTRAINT uq_rag_eval_case_result UNIQUE (eval_result_id),
    INDEX idx_rag_eval_case_runs_case_created (case_id, created_at),
    INDEX idx_rag_eval_case_runs_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS rag_eval_case_run_artifacts (
    case_run_id CHAR(36) PRIMARY KEY,
    context_text MEDIUMTEXT NULL,
    prompt_text MEDIUMTEXT NULL,
    context_sha256 CHAR(64) NULL,
    prompt_sha256 CHAR(64) NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_rag_eval_artifacts_case_run
        FOREIGN KEY (case_run_id) REFERENCES rag_eval_case_runs(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Historical policy:
--   cache_status IS NULL AND from_cache = 1 => report as historical hit.
--   cache_status IS NULL AND from_cache = 0 => report as unknown.
-- Do not backfill historical from_cache=0 rows as miss or bypass.
