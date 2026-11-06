ALTER TABLE kg_inheritance_bindings ADD COLUMN source_relation_id CHAR(64) NOT NULL AFTER run_id;
ALTER TABLE kg_inheritance_bindings DROP PRIMARY KEY, ADD PRIMARY KEY(run_id,source_relation_id,source_evidence_id);
