import os
import json
import redis
from dotenv import load_dotenv
from cache_utils import RAG_QUERY_CACHE_PREFIX


load_dotenv()


def get_redis_client():
    """
    创建 Redis 连接对象。
    Docker Compose 环境下：
    REDIS_HOST=redis
    REDIS_PORT=6379
    """
    connection_kwargs = dict(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True
    )
    password = os.getenv("REDIS_PASSWORD")
    if password:
        connection_kwargs["password"] = password

    client = redis.Redis(**connection_kwargs)

    return client


def check_redis_connection():
    """
    测试 Redis 是否可用。
    """
    client = get_redis_client()
    return client.ping()


def get_cache(key: str):
    """
    从 Redis 中读取缓存。

    Redis异常时返回None，
    不影响主业务继续执行。
    """

    try:
        client = get_redis_client()

        value = client.get(key)

        if value is None:
            return None

        try:
            return json.loads(value)

        except Exception:
            return value

    except Exception as e:

        print(
            f"Redis读取失败:{e}"
        )

        return None


def set_cache(
    key: str,
    value,
    expire_seconds: int = 3600
):
    """
    写入 Redis。

    Redis失败不影响业务流程。
    """

    try:

        client = get_redis_client()

        if isinstance(value, str):
            cache_value = value

        else:
            cache_value = json.dumps(
                value,
                ensure_ascii=False
            )


        client.set(
            name=key,
            value=cache_value,
            ex=expire_seconds
        )


    except Exception as e:

        print(
            f"Redis写入失败:{e}"
        )


def delete_cache(key: str):
    """
    删除指定缓存。
    """
    client = get_redis_client()
    client.delete(key)

def delete_cache_by_prefix(prefix: str) -> int:
    """
    按前缀批量删除 Redis 缓存。

    例如：
    prefix = "rag:answer:"
    会删除所有以 rag:answer: 开头的缓存 key。
    """
    client = get_redis_client()

    deleted_count = 0

    for key in client.scan_iter(match=f"{prefix}*"):
        client.delete(key)
        deleted_count += 1

    return deleted_count


def invalidate_rag_query_cache() -> int:
    """Invalidate current query results and keys written by older versions."""
    return (
        delete_cache_by_prefix(RAG_QUERY_CACHE_PREFIX)
        + delete_cache_by_prefix("rag:answer:")
    )

def incr_cache_stat(stat_name: str, amount: int = 1):
    """
    增加缓存统计计数。

    例如：
    incr_cache_stat("hit")  表示缓存命中次数 +1
    incr_cache_stat("miss") 表示缓存未命中次数 +1
    """
    client = get_redis_client()
    key = f"rag:cache:stats:{stat_name}"
    client.incrby(key, amount)


def get_cache_stat(stat_name: str) -> int:
    """
    获取某个缓存统计值。
    """
    client = get_redis_client()
    key = f"rag:cache:stats:{stat_name}"

    value = client.get(key)

    if value is None:
        return 0

    return int(value)


def count_cache_by_prefix(prefix: str) -> int:
    """
    统计某个前缀下有多少缓存 key。
    """
    client = get_redis_client()

    count = 0

    for _ in client.scan_iter(match=f"{prefix}*"):
        count += 1

    return count


def get_rag_cache_stats() -> dict:
    """
    获取 RAG 问答缓存统计信息。
    """
    hit = get_cache_stat("hit")
    miss = get_cache_stat("miss")
    total = hit + miss

    if total > 0:
        hit_rate = hit / total
    else:
        hit_rate = 0

    cache_count = (
        count_cache_by_prefix(RAG_QUERY_CACHE_PREFIX)
        + count_cache_by_prefix("rag:answer:")
    )

    return {
        "hit": hit,
        "miss": miss,
        "total": total,
        "hit_rate": hit_rate,
        "cache_count": cache_count
    }

def reset_rag_cache_stats() -> int:
    """
    清空 RAG 缓存统计计数。
    会删除：
    - rag:cache:stats:hit
    - rag:cache:stats:miss
    """
    client = get_redis_client()

    keys = [
        "rag:cache:stats:hit",
        "rag:cache:stats:miss"
    ]

    deleted_count = 0

    for key in keys:
        deleted_count += client.delete(key)

    return deleted_count

def clear_all_rag_cache_data() -> dict:
    """
    清空 RAG 问答缓存和缓存统计。
    """
    answer_deleted_count = invalidate_rag_query_cache()
    stats_deleted_count = reset_rag_cache_stats()

    return {
        "answer_deleted_count": answer_deleted_count,
        "stats_deleted_count": stats_deleted_count
    }
