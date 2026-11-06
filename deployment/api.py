"""Container entrypoint: identity checks precede app imports, no migration."""
import isolated_config


def create_app():
    isolated_config.require_target("mysql")
    isolated_config.require_target("redis")
    from api_app import app
    from fastapi import Depends
    from knowledge_v2_api import require_management_token

    @app.get('/__deployment/identity', dependencies=[Depends(require_management_token)])
    def identity():
        import os
        from db_utils import get_connection
        from redis_utils import get_redis_client
        with get_connection() as db:
            mysql_peer = db._sock.getpeername()
            with db.cursor() as cursor:
                cursor.execute('SELECT VERSION(),@@port,DATABASE(),@@server_uuid,@@datadir,CURRENT_USER()')
                row = cursor.fetchone()
        with get_redis_client() as client:
            client.ping()
            wire = client._raw.connection_pool.get_connection()
            try:
                wire.connect()
                redis_peer = wire._sock.getpeername()
            finally:
                client._raw.connection_pool.release(wire)
        view = app.state.knowledge.pin()
        return dict(mysql=dict(zip(['version','port','database','uuid','datadir','user'],row)),
            mysql_tcp_peer=mysql_peer, redis_tcp_peer=redis_peer, redis_db=int(os.environ['REDIS_DB']),
            redis_prefix=os.environ['RAG_V2_REDIS_PREFIX'], redis_identity_verified=True,
            snapshot_id=view.snapshot_id, generation=view.generation,
            dense_class=type(view.index).__name__ if view.index is not None else None,
            vector_count=view.index.ntotal if view.index is not None else 0,
            model_identity=app.state.knowledge.model_identity, embed_path=os.environ['RAG_V2_EMBED_PATH'],
            store=str(app.state.knowledge.root), hf_home=os.environ['HF_HOME'],
            model_fixture=os.environ.get('RAG_V2_ISOLATED_LLM')=='1',
            real_llm=os.environ.get('RAG_LLM_BUDGETED_REAL')=='1', init_db_executed=False)
    return app
