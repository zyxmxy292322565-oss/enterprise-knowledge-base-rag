import os
import pymysql
from datetime import datetime
from dotenv import load_dotenv
from pymysql.connections import Connection
import json

load_dotenv()

# 1. 数据库连接
def get_connection() -> Connection:
    """
    作用：
    创建并返回一个 MySQL 数据库连接对象。

    输入：
    无。
    数据库连接参数从 .env 文件中读取。

    读取的环境变量：
    MYSQL_HOST：
        MySQL 主机地址，本地一般是 localhost。

    MYSQL_PORT：
        MySQL 端口，默认是 3306。

    MYSQL_USER：
        MySQL 用户名，比如 root。

    MYSQL_PASSWORD：
        MySQL 密码。

    MYSQL_DATABASE：
        当前项目使用的数据库名，比如 rag_demo。

    处理流程：
    1. 使用 load_dotenv() 加载 .env 配置。
    2. 使用 os.getenv() 读取 MySQL 配置。
    3. 使用 pymysql.connect() 创建数据库连接。
    4. 设置 charset="utf8mb4"，保证支持中文。
    5. 返回数据库连接对象 conn。

    输出：
    conn：
        MySQL 数据库连接对象。
        后续执行 SQL、提交事务、关闭连接都依赖它。
    """
    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE", "rag_demo"),
        charset="utf8mb4"
    )

    return conn

# 2. 初始化表结构
def init_db() -> None:
    """
    作用：
    初始化数据库表。
    如果 qa_logs 表和 documents 表不存在，就创建它们。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS qa_logs (
        id INT PRIMARY KEY AUTO_INCREMENT,
        question TEXT,
        answer LONGTEXT,
        search_scope TEXT,
        source_file VARCHAR(255),
        chunk_id VARCHAR(50),
        rerank_score DOUBLE,
        cosine_score DOUBLE,
        faiss_distance DOUBLE,
        created_at DATETIME
    ) DEFAULT CHARSET=utf8mb4
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INT PRIMARY KEY AUTO_INCREMENT,
        file_name VARCHAR(255),
        file_hash VARCHAR(64) UNIQUE,
        file_type VARCHAR(20),
        chunk_count INT,
        created_at DATETIME
    ) DEFAULT CHARSET=utf8mb4
    """)

    conn.commit()
    cursor.close()
    conn.close()

    ensure_documents_metadata_columns()
    ensure_qa_logs_note_column()
    init_agent_logs_table()
    ensure_agent_logs_note_column()
    ensure_agent_logs_router_columns()
    init_eval_logs_table()
    ensure_eval_logs_param_columns()
    init_conversation_messages_table()
    init_rag_request_logs_table()
    init_rag_eval_cases_table()
    init_rag_eval_results_table()
    init_rag_eval_experiments_table()


def ensure_documents_metadata_columns() -> None:
    """Add management metadata to existing Policy RAG document catalogs."""
    columns = {
        "category": "VARCHAR(50) NOT NULL DEFAULT 'general_policy'",
        "product_scope": "VARCHAR(100) NOT NULL DEFAULT 'general'",
        "version": "VARCHAR(64) NOT NULL DEFAULT 'v1'",
        "created_by": "VARCHAR(100) NOT NULL DEFAULT 'legacy'",
        "updated_at": "DATETIME NULL",
    }
    conn = get_connection()
    cursor = conn.cursor()
    try:
        for name, definition in columns.items():
            cursor.execute("SHOW COLUMNS FROM documents LIKE %s", (name,))
            if cursor.fetchone() is None:
                cursor.execute(f"ALTER TABLE documents ADD COLUMN `{name}` {definition}")
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def ensure_qa_logs_note_column() -> None:
    """
    作用：
    确保 qa_logs 表中存在 note 备注字段。
    如果 note 字段不存在，就新增。
    如果已经存在，就跳过。

    输出：
    无返回值。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SHOW COLUMNS FROM qa_logs LIKE 'note'
    """)

    result = cursor.fetchone()

    if result is None:
        cursor.execute("""
        ALTER TABLE qa_logs
        ADD COLUMN note TEXT
        """)
        conn.commit()

    cursor.close()
    conn.close()

# 3. 问答记录 qa_logs 操作
def save_qa_logs(
    query: str,
    answer: str,
    search_scope: str,
    reranked_results: list
) -> None:
    """
    作用：
    把一次问答结果和对应的检索来源保存到 MySQL 数据库。

    输入：
    query：
        当前用户问题。

    answer：
        大模型生成的回答。

    search_scope：
        当前检索范围。
        如果用户没有选择文件，一般是“全部文档”。
        如果用户选择了某些文件，则是这些文件名拼接后的字符串。

    reranked_results：
        Rerank 排序后的检索结果列表。
        每个 item 的结构是：
        item[0]：完整文本
        item[1]：预览文本
        item[2]：文本向量
        item[3]：Rerank 分数
        item[4]：余弦相似度
        item[5]：FAISS 距离
        item[6]：来源文件
        item[7]：文本块编号

    处理流程：
    1. 调用 get_connection() 连接数据库。
    2. 创建 cursor。
    3. 获取当前时间 created_at。
    4. 遍历 reranked_results。
    5. 每个检索结果插入 qa_logs 表一行记录。
    6. 使用 conn.commit() 提交保存。
    7. 关闭 cursor 和连接。

    输出：
    无返回值。
    """
    conn = get_connection()
    cursor = conn.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for item in reranked_results:
        cursor.execute("""
        INSERT INTO qa_logs (
            question,
            answer,
            search_scope,
            source_file,
            chunk_id,
            rerank_score,
            cosine_score,
            faiss_distance,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            query,
            answer,
            search_scope,
            item[6],
            str(item[7]),
            float(item[3]),
            float(item[4]),
            float(item[5]),
            created_at
        ))

    conn.commit()
    cursor.close()
    conn.close()

def load_qa_logs(
    limit: int = 20,
    source_file: str | None = None,
    keyword: str | None = None
) -> list:
    """
    作用：
    从 MySQL 的 qa_logs 表中读取问答记录。
    支持限制数量、按来源文件筛选、按问题关键词搜索。

    参数：
    limit：
        最多读取多少条记录，默认读取最近 20 条。

    source_file：
        来源文件名。
        如果传入文件名，就只查询这个文件相关的问答记录。
        如果为 None 或空字符串，就不按文件筛选。

    keyword：
        问题关键词。
        如果传入关键词，就只查询 question 中包含这个关键词的记录。
        如果为 None 或空字符串，就不按关键词筛选。

    返回：
    logs：
        问答记录字典列表，可以直接转成 DataFrame 展示。
    """
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    SELECT
        id,
        question,
        answer,
        search_scope,
        source_file,
        chunk_id,
        rerank_score,
        cosine_score,
        faiss_distance,
        created_at,
        note
    FROM qa_logs
    WHERE 1 = 1
    """

    params = []

    if source_file:
        sql += " AND source_file = %s"
        params.append(source_file)

    if keyword:
        sql += " AND question LIKE %s"
        params.append(f"%{keyword}%")

    sql += " ORDER BY id DESC LIMIT %s"
    params.append(limit)

    cursor.execute(sql, params)

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    logs = []

    for row in rows:
        logs.append({
            "ID": row[0],
            "问题": row[1],
            "回答": row[2],
            "检索范围": row[3],
            "来源文件": row[4],
            "文本块编号": row[5],
            "Rerank评分": row[6],
            "余弦相似度": row[7],
            "FAISS距离": row[8],
            "时间": row[9],
            "备注": row[10]
        })

    return logs

def load_recent_qa_logs(limit: int = 5) -> list:
    """
    作用：
    查询最近的问答记录，用于数据库统计面板展示。

    参数：
    limit：
        最多查询多少条记录，默认查询最近 5 条。

    返回：
    recent_logs：
        最近问答记录列表，可以直接转成 DataFrame 展示。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        question,
        source_file,
        created_at,
        note
    FROM qa_logs
    ORDER BY id DESC
    LIMIT %s
    """, (limit,))

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    recent_logs = []

    for row in rows:
        recent_logs.append({
            "ID": row[0],
            "问题": row[1],
            "来源文件": row[2],
            "时间": row[3],
            "备注": row[4]
        })

    return recent_logs

def delete_qa_log_by_id(log_id: int) -> None:
    """
    作用：
    根据 ID 删除 qa_logs 表中的一条问答记录。

    参数：
    log_id：
        要删除的问答记录 ID。

    处理流程：
    1. 连接 MySQL。
    2. 创建 cursor。
    3. 执行 DELETE FROM qa_logs WHERE id = %s。
    4. 提交事务。
    5. 关闭 cursor 和连接。

    返回：
    无返回值。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    DELETE FROM qa_logs
    WHERE id = %s
    """, (log_id,))

    conn.commit()
    cursor.close()
    conn.close()

def clear_qa_logs() -> None:
    """
    作用：
    清空 qa_logs 表中的所有问答记录。

    输入：
    无。

    处理流程：
    1. 连接 MySQL。
    2. 创建 cursor。
    3. 执行 DELETE FROM qa_logs。
    4. 提交事务。
    5. 关闭 cursor 和连接。

    返回：
    无返回值。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    DELETE FROM qa_logs
    """)

    conn.commit()
    cursor.close()
    conn.close()

def update_qa_log_note(log_id: int, note: str) -> None:
    """
    作用：
    根据 ID 更新 qa_logs 表中某条问答记录的备注。

    参数：
    log_id：
        要更新的问答记录 ID。

    note：
        要保存的备注内容。

    处理流程：
    1. 连接 MySQL。
    2. 创建 cursor。
    3. 执行 UPDATE qa_logs SET note = %s WHERE id = %s。
    4. 提交修改。
    5. 关闭 cursor 和连接。

    返回：
    无返回值。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE qa_logs
    SET note = %s
    WHERE id = %s
    """, (note, log_id))

    conn.commit()
    cursor.close()
    conn.close()

# 4. 文档记录 documents 操作
def save_document_records(new_docs: list) -> None:
    """
    作用：
    把本次新上传文档的记录保存到 MySQL 的 documents 表中。

    输入：
    new_docs：
        本次上传文件切分出来的新 Document 文本块列表。
        每个 doc.metadata 中应该包含：
        - source：文件名
        - file_hash：文件 MD5
        - chunk_id：文本块编号

    处理流程：
    1. 遍历 new_docs。
    2. 根据 file_hash 统计每个文件的文本块数量。
    3. 获取文件名、文件类型、文件 hash、文本块数量。
    4. 使用 INSERT IGNORE 插入 documents 表。
    5. 如果 file_hash 已经存在，则跳过，不重复插入。

    输出：
    无返回值。
    """
    if not new_docs:
        return

    document_map = {}

    for doc in new_docs:
        file_name = doc.metadata.get("source", "未知文件")
        file_hash = doc.metadata.get("file_hash", "未知hash")

        if file_hash not in document_map:
            file_type = file_name.split(".")[-1].lower() if "." in file_name else "unknown"

            document_map[file_hash] = {
                "file_name": file_name,
                "file_hash": file_hash,
                "file_type": file_type,
                "category": doc.metadata.get("category", "general_policy"),
                "product_scope": doc.metadata.get("product_scope", "general"),
                "version": doc.metadata.get("version", "v1"),
                "created_by": doc.metadata.get("created_by", "policy_rag_api"),
                "chunk_count": 0
            }

        document_map[file_hash]["chunk_count"] += 1

    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for doc_info in document_map.values():
        cursor.execute("""
        INSERT IGNORE INTO documents (
            file_name,
            file_hash,
            file_type,
            category,
            product_scope,
            version,
            created_by,
            chunk_count,
            created_at,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            doc_info["file_name"],
            doc_info["file_hash"],
            doc_info["file_type"],
            doc_info["category"],
            doc_info["product_scope"],
            doc_info["version"],
            doc_info["created_by"],
            doc_info["chunk_count"],
            now,
            now,
        ))

    conn.commit()
    cursor.close()
    conn.close()



def load_document_records() -> list:
    """
    作用：
    从 MySQL 的 documents 表中读取已上传文档记录。

    输入：
    无。

    处理流程：
    1. 连接 MySQL。
    2. 查询 documents 表。
    3. 按 id 倒序排列，让最新上传的文档排在前面。
    4. 把查询结果 rows 转成字典列表。
    5. 返回 document_records。

    输出：
    document_records：
        文档记录列表，可以直接转成 DataFrame 展示。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        file_name,
        file_hash,
        file_type,
        category,
        product_scope,
        version,
        created_by,
        chunk_count,
        created_at,
        updated_at
    FROM documents
    ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    document_records = []

    for row in rows:
        document_records.append({
            "ID": row[0],
            "文件名": row[1],
            "文件hash": row[2],
            "文件类型": row[3],
            "分类": row[4],
            "商品范围": row[5],
            "版本": row[6],
            "创建人": row[7],
            "文本块数量": row[8],
            "上传时间": row[9],
            "更新时间": row[10],
        })

    return document_records

def delete_document_record_by_hash(file_hash: str) -> int:
    """
    根据 file_hash 删除 documents 表中的文档记录。

    返回：
    deleted_count：删除的记录数量
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    DELETE FROM documents
    WHERE file_hash = %s
    """, (file_hash,))

    deleted_count = cursor.rowcount

    conn.commit()
    cursor.close()
    conn.close()

    return deleted_count

def load_recent_documents(limit: int = 5) -> list:
    """
    作用：
    查询最近上传的文档记录，用于数据库统计面板展示。

    参数：
    limit：
        最多查询多少条记录，默认查询最近 5 条。

    返回：
    recent_documents：
        最近上传文档记录列表，可以直接转成 DataFrame 展示。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        file_name,
        file_type,
        chunk_count,
        created_at
    FROM documents
    ORDER BY id DESC
    LIMIT %s
    """, (limit,))

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    recent_documents = []

    for row in rows:
        recent_documents.append({
            "ID": row[0],
            "文件名": row[1],
            "文件类型": row[2],
            "文本块数量": row[3],
            "上传时间": row[4]
        })

    return recent_documents

# 5. 统计面板
def load_dashboard_stats() -> dict:
    """
    作用：
    读取数据库统计信息，用于页面展示统计面板。

    统计内容：
    1. qa_logs 表中的问答记录总数
    2. documents 表中的文档记录总数
    3. 最近一次问答时间
    4. 每个来源文件被引用次数

    返回：
    stats：
        一个字典，包含总数和分组统计结果。
    """
    conn = get_connection()
    cursor = conn.cursor()

    # 1. 统计问答记录总数
    cursor.execute("""
    SELECT COUNT(*)
    FROM qa_logs
    """)
    qa_log_count = cursor.fetchone()[0]

    # 2. 统计文档记录总数
    cursor.execute("""
    SELECT COUNT(*)
    FROM documents
    """)
    document_count = cursor.fetchone()[0]

    # 3. 查询最近一次问答时间
    cursor.execute("""
    SELECT MAX(created_at)
    FROM qa_logs
    """)
    latest_qa_time = cursor.fetchone()[0]

    # 4. 按来源文件统计引用次数
    cursor.execute("""
    SELECT source_file, COUNT(*) AS reference_count
    FROM qa_logs
    GROUP BY source_file
    ORDER BY reference_count DESC
    """)
    source_stats_rows = cursor.fetchall()

    cursor.close()
    conn.close()

    source_stats = []

    for row in source_stats_rows:
        source_stats.append({
            "来源文件": row[0],
            "引用次数": row[1]
        })

    stats = {
        "问答记录总数": qa_log_count,
        "文档记录总数": document_count,
        "最近问答时间": latest_qa_time,
        "来源文件统计": source_stats
    }

    return stats

def init_agent_logs_table():
    """
    初始化 Agent 调用日志表。
    用来保存每一次 Agent 的工具调用过程。
    """
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    CREATE TABLE IF NOT EXISTS agent_logs (
        id INT PRIMARY KEY AUTO_INCREMENT,
        question TEXT,
        tool_name VARCHAR(100),
        args JSON,
        tool_result LONGTEXT,
        answer LONGTEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """

    cursor.execute(sql)
    conn.commit()

    cursor.close()
    conn.close()

def save_agent_log(
    question: str,
    tool_name: str,
    args: dict,
    tool_result: str,
    answer: str,
    router_raw: str = "",
    router_validated: dict | None = None
):
    conn = get_connection()
    cursor = conn.cursor()

    if router_validated is None:
        router_validated = {}

    sql = """
    INSERT INTO agent_logs
    (question, tool_name, args, tool_result, answer, router_raw, router_validated)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    cursor.execute(
        sql,
        (
            question,
            tool_name,
            json.dumps(args, ensure_ascii=False),
            tool_result,
            answer,
            router_raw,
            json.dumps(router_validated, ensure_ascii=False)
        )
    )

    conn.commit()
    cursor.close()
    conn.close()

def load_agent_logs(
    limit: int = 20,
    tool_name: str | None = None,
    keyword: str | None = None
):
    """
    查询 Agent 调用日志。
    支持：
    - limit：限制返回数量
    - tool_name：按工具名筛选
    - keyword：按用户问题/最终回答关键词筛选
    """

    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    SELECT
    id,
    question,
    tool_name,
    args,
    tool_result,
    answer,
    router_raw,
    router_validated,
    created_at,
    note
    FROM agent_logs
    WHERE 1 = 1
    """

    params = []

    if tool_name:
        sql += " AND tool_name = %s"
        params.append(tool_name)

    if keyword:
        sql += " AND (question LIKE %s OR answer LIKE %s)"
        keyword_like = f"%{keyword}%"
        params.extend([keyword_like, keyword_like])

    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    cursor.execute(sql, params)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    logs = []

    for row in rows:
        logs.append({
            "ID": row[0],
            "问题": row[1],
            "工具名称": row[2],
            "工具参数": row[3],
            "工具原始结果": row[4],
            "最终回答": row[5],
            "Router原始输出": row[6],
            "Router校验结果": row[7],
            "创建时间": row[8],
            "备注": row[9]
        })

    return logs

def load_agent_log_stats():
    """
    查询 Agent 调用统计信息。
    """

    conn = get_connection()
    cursor = conn.cursor()

    # 1. Agent 总调用次数
    cursor.execute("""
        SELECT COUNT(*)
        FROM agent_logs
    """)
    total_count = cursor.fetchone()[0]

    # 2. 最近一次调用时间
    cursor.execute("""
        SELECT MAX(created_at)
        FROM agent_logs
    """)
    latest_time = cursor.fetchone()[0]

    # 3. 各工具调用次数
    cursor.execute("""
        SELECT tool_name, COUNT(*) AS call_count
        FROM agent_logs
        GROUP BY tool_name
        ORDER BY call_count DESC
    """)
    tool_rows = cursor.fetchall()

    cursor.close()
    conn.close()

    tool_stats = []

    for row in tool_rows:
        tool_stats.append({
            "工具名称": row[0],
            "调用次数": row[1]
        })

    most_used_tool = tool_stats[0]["工具名称"] if tool_stats else "暂无"

    return {
        "Agent调用总数": total_count,
        "最近调用时间": latest_time,
        "最常用工具": most_used_tool,
        "工具调用统计": tool_stats
    }

def delete_agent_log_by_id(log_id: int):
    """
    根据 ID 删除一条 Agent 调用日志。
    """
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    DELETE FROM agent_logs
    WHERE id = %s
    """

    cursor.execute(sql, (log_id,))

    conn.commit()

    cursor.close()
    conn.close()

def clear_agent_logs():
    """
    清空所有 Agent 调用日志。
    """
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    DELETE FROM agent_logs
    """

    cursor.execute(sql)
    conn.commit()

    cursor.close()
    conn.close()

def ensure_agent_logs_note_column():
    """
    确保 agent_logs 表中存在 note 字段。
    如果没有，就自动添加。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SHOW COLUMNS FROM agent_logs LIKE 'note'")
    result = cursor.fetchone()

    if not result:
        cursor.execute("""
            ALTER TABLE agent_logs
            ADD COLUMN note TEXT
        """)
        conn.commit()

    cursor.close()
    conn.close()

def update_agent_log_note(log_id: int, note: str):
    """
    更新某条 Agent 日志的备注。
    """
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    UPDATE agent_logs
    SET note = %s
    WHERE id = %s
    """

    cursor.execute(sql, (note, log_id))
    conn.commit()

    cursor.close()
    conn.close()

def init_eval_logs_table():
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    CREATE TABLE IF NOT EXISTS eval_logs (
        id INT PRIMARY KEY AUTO_INCREMENT,
        question TEXT,
        expected_answer LONGTEXT,
        rag_answer LONGTEXT,
        sources LONGTEXT,
        score INT,
        note TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    ) DEFAULT CHARSET=utf8mb4
    """

    cursor.execute(sql)
    conn.commit()

    cursor.close()
    conn.close()

def save_eval_log(
    question: str,
    expected_answer: str,
    rag_answer: str,
    sources: str,
    score: int,
    note: str = "",
    top_k: int = 0,
    faiss_k: int = 0,
    top_n: int = 0,
    rerank_threshold: float = 0,
    use_rerank: int = 1
):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    INSERT INTO eval_logs (
        question,
        expected_answer,
        rag_answer,
        sources,
        score,
        note,
        top_k,
        faiss_k,
        top_n,
        rerank_threshold,
        use_rerank
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    cursor.execute(
        sql,
        (
            question,
            expected_answer,
            rag_answer,
            sources,
            score,
            note,
            top_k,
            faiss_k,
            top_n,
            rerank_threshold,
            use_rerank
        )
    )

    conn.commit()

    cursor.close()
    conn.close()

def load_eval_logs(limit: int = 20):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    SELECT
        id,
        question,
        expected_answer,
        rag_answer,
        sources,
        score,
        note,
        top_k,
        faiss_k,
        top_n,
        rerank_threshold,
        use_rerank,
        created_at
    FROM eval_logs
    ORDER BY created_at DESC
    LIMIT %s
    """

    cursor.execute(sql, (limit,))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    logs = []

    for row in rows:
        logs.append({
            "ID": row[0],
            "问题": row[1],
            "标准答案": row[2],
            "RAG回答": row[3],
            "来源": row[4],
            "评分": row[5],
            "备注": row[6],
            "top_k": row[7],
            "faiss_k": row[8],
            "top_n": row[9],
            "rerank_threshold": row[10],
            "use_rerank": row[11],
            "创建时间": row[12]
        })

    return logs

def load_eval_stats():
    conn = get_connection()
    cursor = conn.cursor()

    # 评估总数
    cursor.execute("SELECT COUNT(*) FROM eval_logs")
    total_count = cursor.fetchone()[0]

    # 平均分、最高分、最低分
    cursor.execute("""
        SELECT
            AVG(score),
            MAX(score),
            MIN(score)
        FROM eval_logs
    """)
    score_row = cursor.fetchone()

    avg_score = score_row[0] if score_row and score_row[0] is not None else 0
    max_score = score_row[1] if score_row and score_row[1] is not None else 0
    min_score = score_row[2] if score_row and score_row[2] is not None else 0

    # 每个分数对应的数量
    cursor.execute("""
        SELECT score, COUNT(*) AS count
        FROM eval_logs
        GROUP BY score
        ORDER BY score DESC
    """)
    score_rows = cursor.fetchall()

    cursor.close()
    conn.close()

    score_distribution = []
    for row in score_rows:
        score_distribution.append({
            "评分": row[0],
            "数量": row[1]
        })

    return {
        "评估总数": total_count,
        "平均分": round(float(avg_score), 2),
        "最高分": max_score,
        "最低分": min_score,
        "评分分布": score_distribution
    }

def delete_eval_log_by_id(log_id: int):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    DELETE FROM eval_logs
    WHERE id = %s
    """

    cursor.execute(sql, (log_id,))
    conn.commit()

    cursor.close()
    conn.close()

def update_eval_log_note(log_id: int, note: str):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    UPDATE eval_logs
    SET note = %s
    WHERE id = %s
    """

    cursor.execute(sql, (note, log_id))
    conn.commit()

    cursor.close()
    conn.close()

def ensure_eval_logs_param_columns():
    conn = get_connection()
    cursor = conn.cursor()

    columns = [
        ("top_k", "INT DEFAULT 0"),
        ("faiss_k", "INT DEFAULT 0"),
        ("top_n", "INT DEFAULT 0"),
        ("rerank_threshold", "FLOAT DEFAULT 0"),
        ("use_rerank", "INT DEFAULT 1")
    ]

    for column_name, column_type in columns:
        cursor.execute(f"SHOW COLUMNS FROM eval_logs LIKE '{column_name}'")
        result = cursor.fetchone()

        if not result:
            cursor.execute(f"""
                ALTER TABLE eval_logs
                ADD COLUMN {column_name} {column_type}
            """)
            conn.commit()

    cursor.close()
    conn.close()

def load_eval_top_k_stats():
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    SELECT
        top_k,
        COUNT(*) AS eval_count,
        AVG(score) AS avg_score,
        MAX(score) AS max_score,
        MIN(score) AS min_score
    FROM eval_logs
    WHERE top_k > 0
    GROUP BY top_k
    ORDER BY top_k ASC
    """

    cursor.execute(sql)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    stats = []

    for row in rows:
        stats.append({
            "top_k": row[0],
            "评估数量": row[1],
            "平均分": round(float(row[2]), 2) if row[2] is not None else 0,
            "最高分": row[3],
            "最低分": row[4]
        })

    return stats

def ensure_agent_logs_router_columns():
    conn = get_connection()
    cursor = conn.cursor()

    columns = [
        ("router_raw", "LONGTEXT"),
        ("router_validated", "JSON")
    ]

    for column_name, column_type in columns:
        cursor.execute(f"SHOW COLUMNS FROM agent_logs LIKE '{column_name}'")
        result = cursor.fetchone()

        if not result:
            sql = f"ALTER TABLE agent_logs ADD COLUMN {column_name} {column_type}"
            cursor.execute(sql)

    conn.commit()
    cursor.close()
    conn.close()

def init_conversation_messages_table():
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    CREATE TABLE IF NOT EXISTS conversation_messages (
        id INT PRIMARY KEY AUTO_INCREMENT,
        thread_id VARCHAR(100),
        role VARCHAR(20),
        content LONGTEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    ) DEFAULT CHARSET=utf8mb4
    """

    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()

def save_conversation_message(thread_id: str, role: str, content: str):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    INSERT INTO conversation_messages (thread_id, role, content)
    VALUES (%s, %s, %s)
    """

    cursor.execute(sql, (thread_id, role, content))

    conn.commit()
    cursor.close()
    conn.close()

def load_conversation_history(thread_id: str, limit: int = 20):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    SELECT role, content, created_at
    FROM conversation_messages
    WHERE thread_id = %s
    ORDER BY created_at DESC, id DESC
    LIMIT %s
    """

    cursor.execute(sql, (thread_id, limit))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    history = []

    for row in reversed(rows):
        history.append({
            "role": row[0],
            "content": row[1],
            "created_at": str(row[2]) if row[2] else ""
        })

    return history

def list_conversation_threads(limit: int = 50):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    SELECT
        thread_id,
        COUNT(*) AS message_count,
        MAX(created_at) AS last_time
    FROM conversation_messages
    GROUP BY thread_id
    ORDER BY last_time DESC
    LIMIT %s
    """

    cursor.execute(sql, (limit,))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    threads = []

    for row in rows:
        threads.append({
            "thread_id": row[0],
            "message_count": row[1],
            "last_time": str(row[2]) if row[2] else ""
        })

    return threads

def clear_conversation_history(thread_id: str):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    DELETE FROM conversation_messages
    WHERE thread_id = %s
    """

    cursor.execute(sql, (thread_id,))
    deleted_count = cursor.rowcount

    conn.commit()
    cursor.close()
    conn.close()

    return deleted_count

def init_rag_request_logs_table():
    """
    初始化 RAG 请求耗时日志表。
    用于保存每次 /ask 接口的耗时数据。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rag_request_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        question TEXT,
        answer_preview TEXT,
        from_cache BOOLEAN,
        top_k INT,
        used_doc_count INT,
        source_count INT,

        total_ms FLOAT,
        cache_read_ms FLOAT,
        embedding_ms FLOAT,
        faiss_search_ms FLOAT,
        rerank_ms FLOAT,
        llm_ms FLOAT,
        redis_write_ms FLOAT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    cursor.close()
    conn.close()

def save_rag_request_log(
    question: str,
    answer: str,
    from_cache: bool,
    top_k: int,
    used_doc_count: int,
    source_count: int,
    timing: dict
):
    """
    保存一次 RAG 请求耗时日志。
    """

    conn = get_connection()
    cursor = conn.cursor()

    answer_preview = answer[:300] if answer else ""

    cursor.execute("""
    INSERT INTO rag_request_logs (
        question,
        answer_preview,
        from_cache,
        top_k,
        used_doc_count,
        source_count,
        total_ms,
        cache_read_ms,
        embedding_ms,
        faiss_search_ms,
        rerank_ms,
        llm_ms,
        redis_write_ms
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        question,
        answer_preview,
        from_cache,
        top_k,
        used_doc_count,
        source_count,
        float(timing.get("total_ms", 0)),
        float(timing.get("cache_read_ms", 0)),
        float(timing.get("embedding_ms", 0)),
        float(timing.get("faiss_search_ms", 0)),
        float(timing.get("rerank_ms", 0)),
        float(timing.get("llm_ms", 0)),
        float(timing.get("redis_write_ms", 0))
    ))

    conn.commit()
    cursor.close()
    conn.close()

def load_recent_rag_request_logs(limit: int = 20):
    """
    查询最近的 RAG 请求耗时日志。
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        question,
        answer_preview,
        from_cache,
        top_k,
        used_doc_count,
        source_count,
        total_ms,
        cache_read_ms,
        embedding_ms,
        faiss_search_ms,
        rerank_ms,
        llm_ms,
        redis_write_ms,
        created_at
    FROM rag_request_logs
    ORDER BY id DESC
    LIMIT %s
    """, (limit,))

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    result = []

    for row in rows:
        result.append({
            "ID": row[0],
            "问题": row[1],
            "回答预览": row[2],
            "是否命中缓存": bool(row[3]),
            "top_k": row[4],
            "使用文档数": row[5],
            "来源数量": row[6],
            "总耗时ms": row[7],
            "Redis读ms": row[8],
            "Embedding毫秒": row[9],
            "FAISS检索ms": row[10],
            "Rerank毫秒": row[11],
            "LLM毫秒": row[12],
            "Redis写ms": row[13],
            "创建时间": str(row[14])
        })

    return result

def load_slow_rag_request_logs(limit: int = 10):
    """
    查询最慢的 RAG 请求。
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        question,
        from_cache,
        top_k,
        used_doc_count,
        total_ms,
        embedding_ms,
        faiss_search_ms,
        rerank_ms,
        llm_ms,
        created_at
    FROM rag_request_logs
    ORDER BY total_ms DESC
    LIMIT %s
    """, (limit,))

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    result = []

    for row in rows:
        result.append({
            "ID": row[0],
            "问题": row[1],
            "是否命中缓存": bool(row[2]),
            "top_k": row[3],
            "使用文档数": row[4],
            "总耗时ms": row[5],
            "Embedding毫秒": row[6],
            "FAISS检索ms": row[7],
            "Rerank毫秒": row[8],
            "LLM毫秒": row[9],
            "创建时间": str(row[10])
        })

    return result

def load_rag_request_perf_stats():
    """
    统计 RAG 请求整体性能。
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        COUNT(*) AS total_count,
        AVG(total_ms) AS avg_total_ms,
        AVG(llm_ms) AS avg_llm_ms,
        AVG(embedding_ms) AS avg_embedding_ms,
        AVG(faiss_search_ms) AS avg_faiss_ms,
        AVG(rerank_ms) AS avg_rerank_ms,
        SUM(CASE WHEN from_cache = 1 THEN 1 ELSE 0 END) AS cache_hit_count
    FROM rag_request_logs
    """)

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    total_count = int(row[0] or 0)
    cache_hit_count = int(row[6] or 0)

    if total_count > 0:
        cache_hit_rate = cache_hit_count / total_count
    else:
        cache_hit_rate = 0

    return {
        "total_count": total_count,
        "avg_total_ms": float(row[1] or 0),
        "avg_llm_ms": float(row[2] or 0),
        "avg_embedding_ms": float(row[3] or 0),
        "avg_faiss_ms": float(row[4] or 0),
        "avg_rerank_ms": float(row[5] or 0),
        "cache_hit_count": cache_hit_count,
        "cache_hit_rate": cache_hit_rate
    }

def init_rag_eval_cases_table():
    """
    初始化 RAG 评测用例表。
    用来保存标准问题、标准答案、期望关键词。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rag_eval_cases (
        id INT AUTO_INCREMENT PRIMARY KEY,
        question TEXT NOT NULL,
        expected_answer TEXT,
        expected_keywords TEXT,
        enabled BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    cursor.close()
    conn.close()

def init_rag_eval_results_table():
    """
    初始化 RAG 评测结果表。
    用来保存每次评测运行结果。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rag_eval_results (
        id INT AUTO_INCREMENT PRIMARY KEY,
        case_id INT,
        question TEXT,
        expected_answer TEXT,
        expected_keywords TEXT,
        actual_answer TEXT,

        hit_keywords TEXT,
        keyword_score FLOAT,
        hit_count INT,
        keyword_count INT,

        from_cache BOOLEAN,
        cache_status VARCHAR(16) NULL,
        total_ms FLOAT,
        llm_ms FLOAT,
        top_k INT,
        source_count INT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    cursor.close()
    conn.close()

def add_rag_eval_case(question: str, expected_answer: str, expected_keywords: str):
    """
    新增一条 RAG 评测用例。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO rag_eval_cases (
        question,
        expected_answer,
        expected_keywords
    )
    VALUES (%s, %s, %s)
    """, (
        question,
        expected_answer,
        expected_keywords
    ))

    conn.commit()
    case_id = cursor.lastrowid

    cursor.close()
    conn.close()

    return case_id


def load_rag_eval_cases():
    """
    查询所有启用的 RAG 评测用例。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        question,
        expected_answer,
        expected_keywords,
        enabled,
        created_at
    FROM rag_eval_cases
    WHERE enabled = TRUE
    ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    result = []

    for row in rows:
        result.append({
            "ID": row[0],
            "问题": row[1],
            "标准答案": row[2],
            "期望关键词": row[3],
            "是否启用": bool(row[4]),
            "创建时间": str(row[5])
        })

    return result


def get_rag_eval_case_by_id(case_id: int):
    """
    根据 ID 查询一条评测用例。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        question,
        expected_answer,
        expected_keywords,
        enabled,
        created_at
    FROM rag_eval_cases
    WHERE id = %s
    """, (case_id,))

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "question": row[1],
        "expected_answer": row[2],
        "expected_keywords": row[3],
        "enabled": bool(row[4]),
        "created_at": str(row[5])
    }


def delete_rag_eval_case(case_id: int):
    """
    删除评测用例。
    这里采用软删除，把 enabled 改成 False。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE rag_eval_cases
    SET enabled = FALSE
    WHERE id = %s
    """, (case_id,))

    affected = cursor.rowcount

    conn.commit()
    cursor.close()
    conn.close()

    return affected

def _insert_rag_eval_result(
    cursor,
    case_id: int,
    question: str,
    expected_answer: str,
    expected_keywords: str,
    actual_answer: str,
    hit_keywords: list,
    keyword_score: float,
    hit_count: int,
    keyword_count: int,
    from_cache: bool,
    total_ms: float,
    llm_ms: float,
    top_k: int,
    source_count: int,
    cache_status: str | None = None
):
    cursor.execute("""
    INSERT INTO rag_eval_results (
        case_id,
        question,
        expected_answer,
        expected_keywords,
        actual_answer,
        hit_keywords,
        keyword_score,
        hit_count,
        keyword_count,
        from_cache,
        cache_status,
        total_ms,
        llm_ms,
        top_k,
        source_count
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        case_id,
        question,
        expected_answer,
        expected_keywords,
        actual_answer,
        ",".join(hit_keywords),
        float(keyword_score),
        int(hit_count),
        int(keyword_count),
        bool(from_cache),
        cache_status,
        float(total_ms),
        float(llm_ms),
        int(top_k),
        int(source_count)
    ))

    return cursor.lastrowid


def save_rag_eval_result(
    case_id: int,
    question: str,
    expected_answer: str,
    expected_keywords: str,
    actual_answer: str,
    hit_keywords: list,
    keyword_score: float,
    hit_count: int,
    keyword_count: int,
    from_cache: bool,
    total_ms: float,
    llm_ms: float,
    top_k: int,
    source_count: int,
    cache_status: str | None = None
):
    """保存一次兼容旧页面的 RAG 评测结果。"""
    conn = get_connection()
    cursor = conn.cursor()
    result_id = _insert_rag_eval_result(
        cursor=cursor,
        case_id=case_id,
        question=question,
        expected_answer=expected_answer,
        expected_keywords=expected_keywords,
        actual_answer=actual_answer,
        hit_keywords=hit_keywords,
        keyword_score=keyword_score,
        hit_count=hit_count,
        keyword_count=keyword_count,
        from_cache=from_cache,
        total_ms=total_ms,
        llm_ms=llm_ms,
        top_k=top_k,
        source_count=source_count,
        cache_status=cache_status,
    )

    conn.commit()

    cursor.close()
    conn.close()

    return result_id


def load_rag_eval_results(limit: int = 50):
    """
    查询最近的 RAG 评测结果。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        r.id,
        r.case_id,
        r.question,
        r.actual_answer,
        r.hit_keywords,
        r.keyword_score,
        r.hit_count,
        r.keyword_count,
        r.from_cache,
        r.cache_status,
        r.total_ms,
        r.llm_ms,
        r.top_k,
        r.source_count,
        r.created_at,
        cr.id,
        cr.run_id
    FROM rag_eval_results r
    LEFT JOIN rag_eval_case_runs cr ON cr.eval_result_id = r.id
    ORDER BY r.id DESC
    LIMIT %s
    """, (limit,))

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    result = []

    for row in rows:
        resolved_cache_status = row[9] or ("hit" if bool(row[8]) else "unknown")
        result.append({
            "ID": row[0],
            "用例ID": row[1],
            "问题": row[2],
            "系统回答预览": (row[3] or "")[:200],
            "命中关键词": row[4],
            "关键词得分": row[5],
            "命中数量": row[6],
            "关键词总数": row[7],
            "是否命中缓存": bool(row[8]),
            "缓存状态": resolved_cache_status,
            "总耗时ms": row[10],
            "LLM耗时ms": row[11],
            "top_k": row[12],
            "来源数量": row[13],
            "创建时间": str(row[14]),
            "Case Run ID": row[15],
            "Run ID": row[16]
        })

    return result


def load_rag_eval_stats():
    """
    查询 RAG 评测整体统计。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        COUNT(*) AS total_count,
        AVG(keyword_score) AS avg_keyword_score,
        AVG(total_ms) AS avg_total_ms,
        AVG(llm_ms) AS avg_llm_ms,
        SUM(CASE
            WHEN cache_status = 'hit' THEN 1
            WHEN cache_status IS NULL AND from_cache = 1 THEN 1
            ELSE 0 END) AS cache_hit_count,
        SUM(CASE WHEN cache_status = 'miss' THEN 1 ELSE 0 END) AS cache_miss_count,
        SUM(CASE WHEN cache_status = 'bypass' THEN 1 ELSE 0 END) AS cache_bypass_count,
        SUM(CASE WHEN cache_status IS NULL AND from_cache = 0 THEN 1 ELSE 0 END) AS cache_unknown_count
    FROM rag_eval_results
    """)

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    total_count = int(row[0] or 0)
    cache_hit_count = int(row[4] or 0)
    cache_miss_count = int(row[5] or 0)
    cache_bypass_count = int(row[6] or 0)
    cache_unknown_count = int(row[7] or 0)

    normal_cache_count = cache_hit_count + cache_miss_count
    if normal_cache_count > 0:
        cache_hit_rate = cache_hit_count / normal_cache_count
    else:
        cache_hit_rate = 0

    return {
        "total_count": total_count,
        "avg_keyword_score": float(row[1] or 0),
        "avg_total_ms": float(row[2] or 0),
        "avg_llm_ms": float(row[3] or 0),
        "cache_hit_count": cache_hit_count,
        "cache_miss_count": cache_miss_count,
        "cache_bypass_count": cache_bypass_count,
        "cache_unknown_count": cache_unknown_count,
        "normal_cache_count": normal_cache_count,
        "cache_hit_rate": cache_hit_rate
    }

def load_low_score_rag_eval_results(threshold: float = 0.6, limit: int = 20):
    """
    查询低分 RAG 评测结果。
    threshold 默认 0.6，表示关键词得分低于 60% 的结果。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        case_id,
        question,
        actual_answer,
        hit_keywords,
        keyword_score,
        hit_count,
        keyword_count,
        from_cache,
        cache_status,
        total_ms,
        llm_ms,
        top_k,
        source_count,
        created_at
    FROM rag_eval_results
    WHERE keyword_score < %s
    ORDER BY keyword_score ASC, id DESC
    LIMIT %s
    """, (threshold, limit))

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    result = []

    for row in rows:
        resolved_cache_status = row[9] or ("hit" if bool(row[8]) else "unknown")
        result.append({
            "ID": row[0],
            "用例ID": row[1],
            "问题": row[2],
            "系统回答预览": (row[3] or "")[:200],
            "命中关键词": row[4],
            "关键词得分": row[5],
            "命中数量": row[6],
            "关键词总数": row[7],
            "是否命中缓存": bool(row[8]),
            "缓存状态": resolved_cache_status,
            "总耗时ms": row[10],
            "LLM耗时ms": row[11],
            "top_k": row[12],
            "来源数量": row[13],
            "创建时间": str(row[14])
        })

    return result

def load_rag_eval_case_score_stats():
    """
    按评测用例统计平均关键词得分。
    用于观察哪些问题长期表现不好。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        case_id,
        question,
        COUNT(*) AS run_count,
        AVG(keyword_score) AS avg_keyword_score,
        AVG(total_ms) AS avg_total_ms,
        AVG(llm_ms) AS avg_llm_ms,
        MAX(created_at) AS last_run_time
    FROM rag_eval_results
    GROUP BY case_id, question
    ORDER BY avg_keyword_score ASC, run_count DESC
    """)

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    result = []

    for row in rows:
        result.append({
            "用例ID": row[0],
            "问题": row[1],
            "运行次数": int(row[2] or 0),
            "平均关键词得分": float(row[3] or 0),
            "平均总耗时ms": float(row[4] or 0),
            "平均LLM耗时ms": float(row[5] or 0),
            "最近运行时间": str(row[6])
        })

    return result

def init_rag_eval_experiments_table():
    """
    初始化 RAG 参数实验记录表。
    每次批量评测都可以保存为一次实验。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rag_eval_experiments (
        id INT AUTO_INCREMENT PRIMARY KEY,
        experiment_name VARCHAR(255),
        experiment_goal TEXT NULL,
        comparison_variable VARCHAR(255) NULL,
        dataset_version VARCHAR(255) NULL,
        dataset_manifest_sha256 CHAR(64) NULL,
        planned_parameters_json JSON NULL,
        latest_run_id CHAR(36) NULL,

        top_k INT,
        faiss_multiplier INT,
        min_rerank_score FLOAT,
        max_context_chars INT,
        max_chunk_chars INT,
        low_score_threshold FLOAT,

        total_cases INT,
        success_count INT,
        low_score_count INT,

        avg_keyword_score FLOAT,
        avg_total_ms FLOAT,
        avg_llm_ms FLOAT,
        cache_hit_count INT,
        cache_hit_rate FLOAT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    cursor.close()
    conn.close()

def save_rag_eval_experiment(
    experiment_name: str,
    top_k: int,
    faiss_multiplier: int,
    min_rerank_score: float,
    max_context_chars: int,
    max_chunk_chars: int,
    low_score_threshold: float,
    total_cases: int,
    success_count: int,
    low_score_count: int,
    avg_keyword_score: float,
    avg_total_ms: float,
    avg_llm_ms: float,
    cache_hit_count: int,
    cache_hit_rate: float
):
    """
    保存一次 RAG 参数实验汇总结果。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO rag_eval_experiments (
        experiment_name,
        top_k,
        faiss_multiplier,
        min_rerank_score,
        max_context_chars,
        max_chunk_chars,
        low_score_threshold,
        total_cases,
        success_count,
        low_score_count,
        avg_keyword_score,
        avg_total_ms,
        avg_llm_ms,
        cache_hit_count,
        cache_hit_rate
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        experiment_name,
        int(top_k),
        int(faiss_multiplier),
        float(min_rerank_score),
        int(max_context_chars),
        int(max_chunk_chars),
        float(low_score_threshold),
        int(total_cases),
        int(success_count),
        int(low_score_count),
        float(avg_keyword_score),
        float(avg_total_ms),
        float(avg_llm_ms),
        int(cache_hit_count),
        float(cache_hit_rate)
    ))

    conn.commit()
    experiment_id = cursor.lastrowid

    cursor.close()
    conn.close()

    return experiment_id


def load_rag_eval_experiments(limit: int = 50):
    """
    查询最近的 RAG 参数实验记录。
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        experiment_name,
        top_k,
        faiss_multiplier,
        min_rerank_score,
        max_context_chars,
        max_chunk_chars,
        low_score_threshold,
        total_cases,
        success_count,
        low_score_count,
        avg_keyword_score,
        avg_total_ms,
        avg_llm_ms,
        cache_hit_count,
        cache_hit_rate,
        created_at,
        experiment_goal,
        comparison_variable,
        dataset_version,
        dataset_manifest_sha256,
        latest_run_id
    FROM rag_eval_experiments
    ORDER BY id DESC
    LIMIT %s
    """, (limit,))

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    result = []

    for row in rows:
        result.append({
            "实验ID": row[0],
            "实验名称": row[1],
            "top_k": row[2],
            "FAISS初筛倍数": row[3],
            "Rerank阈值": row[4],
            "最大上下文": row[5],
            "单块最大字符": row[6],
            "低分阈值": row[7],
            "用例总数": row[8],
            "成功数量": row[9],
            "低分数量": row[10],
            "平均关键词得分": row[11],
            "平均总耗时ms": row[12],
            "平均LLM耗时ms": row[13],
            "缓存命中数": row[14],
            "缓存命中率": row[15],
            "创建时间": str(row[16]),
            "实验目标": row[17],
            "对比变量": row[18],
            "数据集版本": row[19],
            "数据集哈希": row[20],
            "最新Run ID": row[21]
        })

    return result


def create_rag_eval_experiment_definition(
    experiment_name: str,
    experiment_goal: str,
    comparison_variable: str,
    dataset_version: str,
    dataset_manifest_sha256: str,
    planned_parameters: dict
):
    """Create a v2 experiment definition without executing a run."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO rag_eval_experiments (
            experiment_name,
            experiment_goal,
            comparison_variable,
            dataset_version,
            dataset_manifest_sha256,
            planned_parameters_json
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            experiment_name,
            experiment_goal or None,
            comparison_variable or None,
            dataset_version,
            dataset_manifest_sha256,
            json.dumps(planned_parameters, ensure_ascii=False, allow_nan=False),
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        cursor.close()
        conn.close()


def get_rag_eval_experiment_definition(experiment_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT id, experiment_name, dataset_version, dataset_manifest_sha256
        FROM rag_eval_experiments
        WHERE id = %s
        """, (experiment_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "experiment_id": row[0],
            "experiment_name": row[1],
            "dataset_version": row[2],
            "dataset_manifest_sha256": row[3],
        }
    finally:
        cursor.close()
        conn.close()


def create_rag_eval_run(
    run_id: str,
    experiment_id: int,
    cache_mode: str,
    requested_parameters: dict,
    dataset: dict,
    manifest: dict,
    total_cases: int,
    started_at: datetime
):
    """Persist one immutable run snapshot before executing cases."""
    model_manifest = {
        "embedding_model": manifest.get("embedding_model"),
        "rerank_model": manifest.get("rerank_model"),
        "llm_model": manifest.get("llm_model"),
        "model_revision": manifest.get("model_revision"),
        "model_revision_note": manifest.get("model_revision_note"),
        "llm_temperature": manifest.get("llm_temperature"),
    }
    environment_manifest = {
        "python_version": manifest.get("python_version"),
        "package_versions": manifest.get("package_versions"),
        "platform": manifest.get("platform"),
        "git_note": manifest.get("git_note"),
        "source_files": manifest.get("source_files"),
    }
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO rag_eval_runs (
            id, experiment_id, status, cache_mode,
            requested_parameters_json, effective_parameters_json,
            dataset_version, dataset_manifest_sha256,
            source_files_sha256, git_commit,
            document_manifest_sha256, document_manifest_json,
            index_manifest_sha256, index_manifest_json,
            model_manifest_json, prompt_code_sha256,
            environment_manifest_json, total_cases, started_at
        ) VALUES (
            %s, %s, 'running', %s,
            %s, NULL,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s, %s
        )
        """, (
            run_id,
            experiment_id,
            cache_mode,
            json.dumps(requested_parameters, ensure_ascii=False, allow_nan=False),
            dataset["dataset_version"],
            dataset["dataset_manifest_sha256"],
            manifest["source_files_sha256"],
            manifest.get("git_commit"),
            manifest["document_manifest_sha256"],
            json.dumps(manifest.get("document_manifest", []), ensure_ascii=False, allow_nan=False),
            manifest["index_manifest_sha256"],
            json.dumps(manifest.get("index_files", {}), ensure_ascii=False, allow_nan=False),
            json.dumps(model_manifest, ensure_ascii=False, allow_nan=False),
            manifest["prompt_code_sha256"],
            json.dumps(environment_manifest, ensure_ascii=False, allow_nan=False),
            int(total_cases),
            started_at,
        ))
        cursor.execute(
            "UPDATE rag_eval_experiments SET latest_run_id=%s WHERE id=%s",
            (run_id, experiment_id),
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def finish_rag_eval_run(
    run_id: str,
    status: str,
    completed_cases: int,
    success_count: int,
    failed_count: int,
    effective_parameters: dict | None,
    ended_at: datetime,
    error_type: str | None = None
):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        UPDATE rag_eval_runs
        SET status=%s,
            completed_cases=%s,
            success_count=%s,
            failed_count=%s,
            effective_parameters_json=%s,
            ended_at=%s,
            error_type=%s
        WHERE id=%s
        """, (
            status,
            int(completed_cases),
            int(success_count),
            int(failed_count),
            json.dumps(effective_parameters, ensure_ascii=False, allow_nan=False)
            if effective_parameters is not None else None,
            ended_at,
            error_type,
            run_id,
        ))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def update_rag_eval_experiment_summary(
    experiment_id: int,
    *,
    parameters: dict,
    total_cases: int,
    success_count: int,
    low_score_count: int,
    avg_keyword_score: float,
    avg_total_ms: float,
    avg_llm_ms: float,
    cache_hit_count: int,
    cache_hit_rate: float
):
    """Keep the legacy experiment page populated with the latest run summary."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        UPDATE rag_eval_experiments
        SET top_k=%s,
            faiss_multiplier=%s,
            min_rerank_score=%s,
            max_context_chars=%s,
            max_chunk_chars=%s,
            low_score_threshold=%s,
            total_cases=%s,
            success_count=%s,
            low_score_count=%s,
            avg_keyword_score=%s,
            avg_total_ms=%s,
            avg_llm_ms=%s,
            cache_hit_count=%s,
            cache_hit_rate=%s
        WHERE id=%s
        """, (
            int(parameters["top_k"]),
            int(parameters["faiss_multiplier"]),
            float(parameters["min_rerank_score"]),
            int(parameters["max_context_chars"]),
            int(parameters["max_chunk_chars"]),
            float(parameters["low_score_threshold"]),
            int(total_cases),
            int(success_count),
            int(low_score_count),
            float(avg_keyword_score),
            float(avg_total_ms),
            float(avg_llm_ms),
            int(cache_hit_count),
            float(cache_hit_rate),
            int(experiment_id),
        ))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def save_tracked_rag_eval_case_run(
    *,
    case_run_id: str,
    run_id: str,
    case_id: int,
    question: str,
    expected_answer: str,
    expected_keywords: str,
    actual_answer: str,
    hit_keywords: list,
    keyword_score: float,
    hit_count: int,
    keyword_count: int,
    from_cache: bool,
    cache_status: str,
    total_ms: float,
    llm_ms: float,
    top_k: int,
    source_count: int,
    status: str,
    effective_parameters: dict | None,
    retrieval_trace: dict | None,
    scoring: dict | None,
    timing: dict | None,
    refusal_reason: str | None,
    error_stage: str | None,
    error_type: str | None,
    artifacts: dict | None,
):
    """Atomically save the legacy result, Case Run and its large artifacts."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        result_id = _insert_rag_eval_result(
            cursor=cursor,
            case_id=case_id,
            question=question,
            expected_answer=expected_answer,
            expected_keywords=expected_keywords,
            actual_answer=actual_answer,
            hit_keywords=hit_keywords,
            keyword_score=keyword_score,
            hit_count=hit_count,
            keyword_count=keyword_count,
            from_cache=from_cache,
            cache_status=cache_status,
            total_ms=total_ms,
            llm_ms=llm_ms,
            top_k=top_k,
            source_count=source_count,
        )
        cursor.execute("""
        INSERT INTO rag_eval_case_runs (
            id, run_id, case_id, eval_result_id, status, question,
            cache_status, effective_parameters_json, retrieval_trace_json,
            final_answer, scoring_json, timing_json, refusal_reason,
            error_stage, error_type
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            case_run_id,
            run_id,
            case_id,
            result_id,
            status,
            question,
            cache_status,
            json.dumps(effective_parameters, ensure_ascii=False, allow_nan=False)
            if effective_parameters is not None else None,
            json.dumps(retrieval_trace, ensure_ascii=False, allow_nan=False)
            if retrieval_trace is not None else None,
            actual_answer,
            json.dumps(scoring, ensure_ascii=False, allow_nan=False) if scoring is not None else None,
            json.dumps(timing, ensure_ascii=False, allow_nan=False) if timing is not None else None,
            refusal_reason,
            error_stage,
            error_type,
        ))
        if artifacts and (artifacts.get("context_text") is not None or artifacts.get("prompt_text") is not None):
            cursor.execute("""
            INSERT INTO rag_eval_case_run_artifacts (
                case_run_id, context_text, prompt_text, context_sha256, prompt_sha256
            ) VALUES (%s, %s, %s, %s, %s)
            """, (
                case_run_id,
                artifacts.get("context_text"),
                artifacts.get("prompt_text"),
                artifacts.get("context_sha256"),
                artifacts.get("prompt_sha256"),
            ))
        conn.commit()
        return result_id
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def save_failed_rag_eval_case_run(
    *,
    case_run_id: str,
    run_id: str,
    case_id: int,
    question: str,
    cache_status: str,
    status: str,
    effective_parameters: dict | None,
    retrieval_trace: dict | None,
    timing: dict | None,
    error_stage: str | None,
    error_type: str | None,
    artifacts: dict | None,
):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO rag_eval_case_runs (
            id, run_id, case_id, eval_result_id, status, question,
            cache_status, effective_parameters_json, retrieval_trace_json,
            final_answer, scoring_json, timing_json, refusal_reason,
            error_stage, error_type
        ) VALUES (%s, %s, %s, NULL, %s, %s, %s, %s, %s, NULL, NULL, %s, NULL, %s, %s)
        """, (
            case_run_id,
            run_id,
            case_id,
            status,
            question,
            cache_status,
            json.dumps(effective_parameters, ensure_ascii=False, allow_nan=False)
            if effective_parameters is not None else None,
            json.dumps(retrieval_trace, ensure_ascii=False, allow_nan=False)
            if retrieval_trace is not None else None,
            json.dumps(timing, ensure_ascii=False, allow_nan=False) if timing is not None else None,
            error_stage,
            error_type,
        ))
        if artifacts and (artifacts.get("context_text") is not None or artifacts.get("prompt_text") is not None):
            cursor.execute("""
            INSERT INTO rag_eval_case_run_artifacts (
                case_run_id, context_text, prompt_text, context_sha256, prompt_sha256
            ) VALUES (%s, %s, %s, %s, %s)
            """, (
                case_run_id,
                artifacts.get("context_text"),
                artifacts.get("prompt_text"),
                artifacts.get("context_sha256"),
                artifacts.get("prompt_sha256"),
            ))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def load_rag_eval_runs(limit: int = 50):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT r.id, r.experiment_id, e.experiment_name, r.status, r.cache_mode,
               r.dataset_version, r.dataset_manifest_sha256,
               r.source_files_sha256, r.document_manifest_sha256,
               r.index_manifest_sha256, r.total_cases, r.completed_cases,
               r.success_count, r.failed_count, r.started_at, r.ended_at
        FROM rag_eval_runs r
        JOIN rag_eval_experiments e ON e.id = r.experiment_id
        ORDER BY r.started_at DESC
        LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
        return [{
            "Run ID": row[0],
            "实验ID": row[1],
            "实验名称": row[2],
            "状态": row[3],
            "缓存模式": row[4],
            "数据集版本": row[5],
            "数据集哈希": row[6],
            "源码哈希": row[7],
            "文档哈希": row[8],
            "索引哈希": row[9],
            "用例总数": row[10],
            "已完成": row[11],
            "成功": row[12],
            "失败": row[13],
            "开始时间": str(row[14]),
            "结束时间": str(row[15]) if row[15] is not None else None,
        } for row in rows]
    finally:
        cursor.close()
        conn.close()


def load_rag_eval_case_run(case_run_id: str, include_artifacts: bool = False):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT cr.id, cr.run_id, cr.case_id, cr.eval_result_id, cr.status,
               cr.question, cr.cache_status, cr.effective_parameters_json,
               cr.retrieval_trace_json, cr.final_answer, cr.scoring_json,
               cr.timing_json, cr.refusal_reason, cr.error_stage, cr.error_type,
               a.context_text, a.prompt_text, a.context_sha256, a.prompt_sha256,
               cr.created_at
        FROM rag_eval_case_runs cr
        LEFT JOIN rag_eval_case_run_artifacts a ON a.case_run_id = cr.id
        WHERE cr.id = %s
        """, (case_run_id,))
        row = cursor.fetchone()
        if not row:
            return None

        def decode(value):
            if value is None or isinstance(value, (dict, list)):
                return value
            return json.loads(value)

        return {
            "case_run_id": row[0],
            "run_id": row[1],
            "case_id": row[2],
            "eval_result_id": row[3],
            "status": row[4],
            "question": row[5],
            "cache_status": row[6],
            "effective_parameters": decode(row[7]),
            "retrieval_trace": decode(row[8]),
            "final_answer": row[9],
            "scoring": decode(row[10]),
            "timing": decode(row[11]),
            "refusal_reason": row[12],
            "error_stage": row[13],
            "error_type": row[14],
            "artifacts": {
                "content_loaded": bool(include_artifacts),
                "context_text": row[15] if include_artifacts else None,
                "prompt_text": row[16] if include_artifacts else None,
                "context_sha256": row[17],
                "prompt_sha256": row[18],
            },
            "created_at": str(row[19]),
        }
    finally:
        cursor.close()
        conn.close()
