-- For verified isolated schemas. Fresh Milvus snapshots keep the receipt in their
-- immutable manifest; migrated snapshots use this registry to bind a receipt.
CREATE TABLE IF NOT EXISTS rag_v3_vector_bindings (
    snapshot_id CHAR(32) PRIMARY KEY,
    source_manifest_sha CHAR(64) NOT NULL,
    receipt_json LONGTEXT NOT NULL,
    receipt_sha CHAR(64) NOT NULL,
    state VARCHAR(16) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;
