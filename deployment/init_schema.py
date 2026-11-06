"""One-time schema setup for a verified, empty Compose database."""
from pathlib import Path
import os

import isolated_config
from db_utils import get_connection, init_db

MIGRATIONS = (
    "20261019_rag_eval_tracking_v2.sql",
    "20261103_knowledge_v2.sql",
    "20261103_evaluation_center.sql",
    "20261103_kg.sql",
    "20261104_kg_extraction.sql",
    "20261105_kg_governance.sql",
    "20261105_directed_repair.sql",
    "20261105_directed_binding_identity.sql",
    "20261106_milvus_bindings.sql",
)


def main():
    if not isolated_config._CONTAINER or not os.environ["MYSQL_DATABASE"].startswith("rag_v2_docker_"):
        raise SystemExit("Schema initialization requires an isolated Compose database")
    with get_connection() as conn, conn.cursor() as cursor:
        cursor.execute("SHOW FULL TABLES")
        if cursor.fetchone() is not None:
            raise SystemExit("Database is not empty; schema initialization refused")
    init_db()
    migrations_dir = Path(__file__).resolve().parents[1] / "migrations"
    for name in MIGRATIONS:
        migration = migrations_dir / name
        # Checked-in migrations contain only simple DDL and seed SQL.
        sql = "\n".join(line for line in migration.read_text(encoding="utf-8").splitlines()
                        if not line.lstrip().startswith("--"))
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        if name == MIGRATIONS[0]:
            # init_db already creates the columns added by the historical ALTERs.
            statements = [s for s in statements if s.startswith("CREATE TABLE")]
            statements.insert(0, """ALTER TABLE rag_eval_results
                ADD CONSTRAINT chk_rag_eval_results_cache_status
                CHECK (cache_status IS NULL OR cache_status IN ('bypass', 'hit', 'miss'))""")
        with get_connection() as conn, conn.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)
            conn.commit()
        print("Applied", migration.name)


if __name__ == "__main__":
    main()
