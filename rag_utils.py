import hashlib
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import io
from pypdf import PdfReader

import re
from langchain_core.documents import Document

def is_heading_line(line: str) -> bool:
    """
    判断一行是否像标题。
    适配：
    1. 一、项目背景
    2. 1. 项目背景
    3. 1.1 系统架构
    4. 第一章 项目介绍
    """
    line = line.strip()

    if not line:
        return False

    patterns = [
        r"^第[一二三四五六七八九十\d]+[章节部分].*",
        r"^[一二三四五六七八九十]+、.*",
        r"^\d+(\.\d+)*[、.．\s]+.*",
        r"^#+\s+.*",
    ]

    return any(re.match(pattern, line) for pattern in patterns)


def is_noise_line(line: str) -> bool:
    """
    判断是否是无意义噪声行。
    """
    line = line.strip()

    if not line:
        return True

    noise_patterns = [
        r"^第\s*\d+\s*页$",
        r"^Page\s+\d+$",
        r"^\d+$",
        r"^[\-—_=·.。]+$",
        r"^公司内部资料$",
        r"^内部资料$",
        r"^Confidential$",
    ]

    return any(re.match(pattern, line, re.IGNORECASE) for pattern in noise_patterns)


def clean_document_text(text: str) -> dict:
    """
    文档清洗：
    1. 统一换行符
    2. 去除页码、页眉页脚、纯数字等噪声行
    3. 合并 PDF 断行
    4. 压缩多余空行
    5. 返回清洗前后统计信息
    """
    raw_text = text or ""
    raw_length = len(raw_text)

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if is_noise_line(line):
            continue

        # 压缩行内多个空格
        line = re.sub(r"[ \t]+", " ", line)

        cleaned_lines.append(line)

    # 合并 PDF 断行
    merged_lines = []
    buffer = ""

    for line in cleaned_lines:
        line = line.strip()

        if not line:
            if buffer:
                merged_lines.append(buffer.strip())
                buffer = ""
            merged_lines.append("")
            continue

        # 标题单独成行
        if is_heading_line(line):
            if buffer:
                merged_lines.append(buffer.strip())
                buffer = ""
            merged_lines.append(line)
            continue

        if not buffer:
            buffer = line
        else:
            # 如果上一句已经完整结束，就另起一行
            if buffer.endswith(("。", "！", "？", "；", ".", "!", "?", ";")):
                merged_lines.append(buffer.strip())
                buffer = line
            else:
                # 中文 PDF 断行通常可以直接拼接
                buffer += line

    if buffer:
        merged_lines.append(buffer.strip())

    cleaned_text = "\n".join(merged_lines)

    # 压缩 3 个以上空行
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = cleaned_text.strip()

    cleaned_length = len(cleaned_text)

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "raw_length": raw_length,
        "cleaned_length": cleaned_length,
        "removed_chars": raw_length - cleaned_length,
        "clean_rate": round((raw_length - cleaned_length) / raw_length, 4) if raw_length > 0 else 0
    }


def validate_cleaned_text(cleaned_text: str, min_length: int = 30) -> dict:
    """
    清洗后文本质量校验。
    """
    if cleaned_text is None:
        return {
            "valid": False,
            "message": "清洗后文本为空"
        }

    text = cleaned_text.strip()

    if not text:
        return {
            "valid": False,
            "message": "清洗后文本为空，不能加入知识库"
        }

    if len(text) < min_length:
        return {
            "valid": False,
            "message": f"清洗后文本过短，至少需要 {min_length} 个字符"
        }

    return {
        "valid": True,
        "message": "清洗后文本质量校验通过",
        "text": text
    }


def split_by_paragraph_then_recursive(
    text: str,
    source: str,
    file_hash: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    extra_metadata: dict | None = None
) -> list:
    """
    段落优先切分：
    1. 先按空行切段落
    2. 短段落合并
    3. 超长段落用 RecursiveCharacterTextSplitter 兜底
    """
    extra_metadata = extra_metadata or {}

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    docs = []
    chunk_id = 1
    buffer = ""

    def flush_buffer():
        nonlocal buffer, chunk_id, docs

        if not buffer.strip():
            return

        if len(buffer) <= chunk_size:
            doc = Document(
                page_content=buffer.strip(),
                metadata={
                    "source": source,
                    "file_hash": file_hash,
                    "chunk_id": chunk_id,
                    "split_type": "paragraph",
                    "chunk_length": len(buffer.strip()),
                    **extra_metadata
                }
            )
            docs.append(doc)
            chunk_id += 1
        else:
            sub_docs = splitter.create_documents(
                [buffer],
                metadatas=[{
                    "source": source,
                    "file_hash": file_hash,
                    "split_type": "paragraph_recursive",
                    **extra_metadata
                }]
            )

            for sub_doc in sub_docs:
                sub_doc.metadata["chunk_id"] = chunk_id
                sub_doc.metadata["chunk_length"] = len(sub_doc.page_content)
                docs.append(sub_doc)
                chunk_id += 1

        buffer = ""

    for paragraph in paragraphs:
        # 单个段落太长，先把已有 buffer 刷出去，再递归切当前段落
        if len(paragraph) > chunk_size:
            flush_buffer()

            sub_docs = splitter.create_documents(
                [paragraph],
                metadatas=[{
                    "source": source,
                    "file_hash": file_hash,
                    "split_type": "paragraph_recursive",
                    **extra_metadata
                }]
            )

            for sub_doc in sub_docs:
                sub_doc.metadata["chunk_id"] = chunk_id
                sub_doc.metadata["chunk_length"] = len(sub_doc.page_content)
                docs.append(sub_doc)
                chunk_id += 1

            continue

        # 尽量合并短段落，避免 chunk 太碎
        candidate = paragraph if not buffer else buffer + "\n\n" + paragraph

        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            flush_buffer()
            buffer = paragraph

    flush_buffer()

    return docs


def split_by_heading_then_recursive(
    text: str,
    source: str,
    file_hash: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    extra_metadata: dict | None = None
) -> list:
    """
    标题优先切分：
    1. 根据标题行切章节
    2. 每个章节太长再递归切分
    """
    extra_metadata = extra_metadata or {}

    lines = text.splitlines()
    sections = []
    current_section = []

    for line in lines:
        if is_heading_line(line) and current_section:
            sections.append("\n".join(current_section).strip())
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section).strip())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    docs = []
    chunk_id = 1

    for section in sections:
        if not section.strip():
            continue

        if len(section) <= chunk_size:
            doc = Document(
                page_content=section.strip(),
                metadata={
                    "source": source,
                    "file_hash": file_hash,
                    "chunk_id": chunk_id,
                    "split_type": "heading",
                    "chunk_length": len(section.strip()),
                    **extra_metadata
                }
            )
            docs.append(doc)
            chunk_id += 1
        else:
            sub_docs = splitter.create_documents(
                [section],
                metadatas=[{
                    "source": source,
                    "file_hash": file_hash,
                    "split_type": "heading_recursive",
                    **extra_metadata
                }]
            )

            for sub_doc in sub_docs:
                sub_doc.metadata["chunk_id"] = chunk_id
                sub_doc.metadata["chunk_length"] = len(sub_doc.page_content)
                docs.append(sub_doc)
                chunk_id += 1

    return docs


def split_document_text(
    text: str,
    source: str,
    file_hash: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    strategy: str = "paragraph",
    extra_metadata: dict | None = None
) -> list:
    """
    统一文档切分入口。

    strategy:
    - recursive：普通递归切分
    - paragraph：段落优先切分
    - heading：标题优先切分
    """
    extra_metadata = extra_metadata or {}

    if strategy == "heading":
        return split_by_heading_then_recursive(
            text=text,
            source=source,
            file_hash=file_hash,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extra_metadata=extra_metadata
        )

    if strategy == "paragraph":
        return split_by_paragraph_then_recursive(
            text=text,
            source=source,
            file_hash=file_hash,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extra_metadata=extra_metadata
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    docs = splitter.create_documents(
        [text],
        metadatas=[{
            "source": source,
            "file_hash": file_hash,
            "split_type": "recursive",
            **extra_metadata
        }]
    )

    for chunk_id, doc in enumerate(docs, start=1):
        doc.metadata["chunk_id"] = chunk_id
        doc.metadata["chunk_length"] = len(doc.page_content)

    return docs

def build_docs_from_uploaded_files(
    uploaded_files:list,
    chunk_size:int,
    chunk_overlap:int,
    processed_file_hashes:set
)-> tuple[list,set]:
    """
    作用：
    处理用户上传的 txt 文件，把文件内容切分成 Document 文本块。
    输入：
    uploaded_files：
        Streamlit 上传的文件列表，每个 uploaded_file 都可以通过 getvalue() 读取 bytes。
    chunk_size：
        每个文本块的大小。
    chunk_overlap：
        相邻文本块之间的重叠大小，用来减少上下文断裂。
    processed_file_hashes：
        已经处理过的文件 MD5 集合，用来防止重复上传同一个文件时重复切分、重复入库。
    处理流程：
    1. 遍历 uploaded_files 中的每个文件。
    2. 使用 uploaded_file.getvalue() 获取文件的原始 bytes。
    3. 根据 file_bytes 生成 file_hash，也就是文件内容的 MD5 指纹。
    4. 如果 file_hash 已经存在于 processed_file_hashes，说明该文件处理过，直接跳过。
    5. 如果是新文件，把 file_bytes decode 成字符串 text。
    6. 使用 RecursiveCharacterTextSplitter 根据 chunk_size 和 chunk_overlap 切分文本。
    7. 使用 create_documents() 生成 Document 文本块。
    8. 给每个 Document 添加 metadata：
       - source：文件名
       - file_hash：文件 MD5
    9. 把本次文件切分出的 docs 添加到 new_docs。
    10. 把 file_hash 添加到 processed_file_hashes，表示这个文件已经处理过。
    输出：
    new_docs：
        本次新上传文件切分出来的 Document 文本块列表。
    processed_file_hashes：
        更新后的已处理文件 MD5 集合。
    """
    new_docs = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if file_hash in processed_file_hashes:
            continue

        text = load_uploaded_file_text(uploaded_file.name, file_bytes)

        if not text.strip():
            continue
        # 如果文本去掉空格和换行后什么都没有，就跳过这个文件

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        docs = splitter.create_documents(
            [text],
            metadatas=[{"source": uploaded_file.name, "file_hash": file_hash}]
        )
        for chunk_id, doc in enumerate(docs, start=1):
            doc.metadata["chunk_id"] = chunk_id
            # 遍历当前文件切出来的每个doc
            # 给每个doc.metadata添加chunk_id从1开始编号
        if not docs:
            continue
        # 有时候文本太短或者内容异常，docs可能为空

        new_docs.extend(docs)
        processed_file_hashes.add(file_hash)

    return new_docs, processed_file_hashes
# 输入uploaded_files, chunk_size, chunk_overlap, processed_file_hashes，
# 对上传的每个文件进行getvalue得到原始bytes，然后得到MD5，如果他的MD5在processed_file_hashes，就跳过
# 接着转为字符串，进行切分文本块，docs是切分好后的文本，添加来源和file_hash
# 把本次上传文件切分出的 docs 添加到 new_docs
# 把当前文件的 file_hash 添加到 processed_file_hashes，防止重复处理
#返回新文本和新的MD5

def encode_docs(
    new_docs:list,
    embed_model
)->np.ndarray:
    """
    作用：
    把 Document 文本块转换成 embedding 向量。
    输入：
    new_docs：
        本次新上传文件切分后的 Document 文本块列表。
    embed_model：
        embedding 模型，例如 SentenceTransformer("BAAI/bge-small-zh-v1.5")。
    处理流程：
    1. 遍历 new_docs。
    2. 取出每个 doc.page_content，也就是文本块正文。
    3. 使用 embed_model.encode() 把文本块转换成向量。
    4. 使用 np.array() 把所有向量整理成 NumPy 数组。
    5. 指定 dtype=np.float32，保证向量格式适合 FAISS 使用。
    输出：
    new_vectors：
        本次新文本块对应的向量数组。
        形状大概是：
        [文本块数量, 向量维度]
    """
    if not new_docs:
        return np.array([], dtype=np.float32)
    # 如果没有新文本块，就不要继续向量化，直接返回空数组

    new_vectors = np.array(
        [embed_model.encode(doc.page_content) for doc in new_docs],
        dtype=np.float32
    )

    return new_vectors
# 把 new_docs 中每个 doc.page_content 转成 embedding 向量
# 返回 float32 类型的 NumPy 向量数组 new_vectors

def create_or_update_index(
    index,
    vectors:np.ndarray,
    new_vectors:np.ndarray,
    index_type:str
):
    """
    作用：
    创建新的 FAISS 索引，或者更新已有的 FAISS 索引。
    输入：
    index：
        当前 FAISS 索引。
        如果还没有索引，index 为 None。
        如果已经加载过缓存，index 就是已有的 FAISS 索引对象。
    vectors：
        当前全部文本块的向量。
        如果是第一次创建索引，需要把全部 vectors 加入 FAISS。
    new_vectors：
        本次新上传文本块对应的向量。
        如果 index 已经存在，只需要把 new_vectors 加入索引，不能重复加入全部 vectors。
    index_type：
        用户选择的 FAISS 索引类型。
        当前支持：
        - IndexFlatL2
        - IndexIVFFlat
        - IndexHNSWFlat
    处理流程：
    1. 通过 vectors.shape[1] 获取向量维度 dim。
    2. 如果 index is None，说明当前没有索引，需要新建索引。
    3. 根据 index_type 创建不同类型的 FAISS 索引。
    4. 新建索引后，把全部 vectors 添加进去。
    5. 如果 index 已经存在，说明旧向量已经在索引中，只添加 new_vectors。
    6. 返回创建或更新后的 index。
    输出：
    index：
        创建或更新后的 FAISS 索引对象。
    """
    dim = vectors.shape[1]

    if index is None:
        if index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(dim)
            index.add(vectors)

        elif index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(dim)
            nlist = min(10, len(vectors))
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
            index.train(vectors)
            index.add(vectors)

        elif index_type == "IndexHNSWFlat":
            index = faiss.IndexHNSWFlat(dim, 32)
            index.add(vectors)

    else:
        index.add(new_vectors)

    return index
# 如果 index 为空，说明还没有 FAISS 索引：
#   根据 index_type 创建一个新的 FAISS 索引，并把全部 vectors 添加进去
# 如果 index 已经存在：
#   说明旧向量已经在索引里了，只需要把 new_vectors 添加进去，避免重复添加旧向量

def retrieve_and_rerank(
    query: str,
    query_vector: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    all_docs: list,
    vectors: np.ndarray,
    rerank_model
) -> list:
    """
    作用：
    根据 FAISS 检索返回的索引取回文本块，并使用 Rerank 模型重新排序。
    输入：
    query：
        当前用户问题，字符串形式。
    query_vector：
        当前用户问题对应的向量。
        一般形状是 [1, 向量维度]。
    D：
        FAISS 返回的距离列表。
        距离越小，说明向量距离越近。
    I：
        FAISS 返回的文本块索引列表。
        I 里面的每个 idx 都可以用来从 all_docs 中取回对应文本块。
    all_docs：
        所有 Document 文本块列表。
    vectors：
        所有文本块对应的向量数组。
        vectors[idx] 和 all_docs[idx] 是对应的。
    rerank_model：
        CrossEncoder 重排序模型。
        它会根据 [query, text] 判断文本和问题的相关性。
    处理流程：
    1. 创建几个空列表，用来保存检索结果：
       - retrieved_full_texts：完整文本
       - retrieved_preview_texts：预览文本
       - retrieved_vectors：文本向量
       - retrieved_sources：来源文件
       - retrieved_distances：FAISS 距离
    2. 遍历 FAISS 返回的 I 和 D。
    3. 如果 idx 小于 0，说明是无效索引，跳过。
    4. 使用 all_docs[idx] 取回对应 Document。
    5. 从 doc.page_content 中获取完整文本和预览文本。
    6. 从 vectors[idx] 中获取该文本块的向量。
    7. 从 doc.metadata 中获取来源文件 source。
    8. 如果没有检索到任何有效向量，直接返回空列表。
    9. 计算 query_vector 和 retrieved_vectors 的余弦相似度。
    10. 构造 Rerank 模型需要的输入格式：
        [[query, text1], [query, text2], ...]
    11. 使用 rerank_model.predict() 计算每个文本块的 Rerank 分数。
    12. 使用 zip() 把完整文本、预览文本、向量、Rerank分数、余弦相似度、FAISS距离、来源文件打包。
    13. 根据 item[3]，也就是 Rerank 分数，从大到小排序。
    14. 返回排序后的 reranked_results。
    输出：
    reranked_results：
        按 Rerank 分数从高到低排序后的检索结果列表。
    reranked_results 中每个 item 的结构：
    item[0]：完整文本 full_text，给大模型使用
    item[1]：预览文本 preview_text，给页面表格使用
    item[2]：文本向量
    item[3]：Rerank 分数
    item[4]：余弦相似度评分
    item[5]：FAISS 原始距离
    item[6]：来源文件 source
    """
    retrieved_full_texts = []
    retrieved_preview_texts = []
    retrieved_vectors = []
    retrieved_sources = []
    retrieved_distances = []
    retrieved_chunk_ids = []

    for idx, distance in zip(I, D):
        idx = int(idx)

        if idx < 0:
            continue

        doc = all_docs[idx]

        full_text = doc.page_content
        preview_text = doc.page_content[:500] + "..."

        retrieved_full_texts.append(full_text)
        retrieved_preview_texts.append(preview_text)
        retrieved_vectors.append(vectors[idx])
        retrieved_distances.append(float(distance))

        source = doc.metadata.get("source", "未知来源")
        retrieved_sources.append(source)
        chunk_id = doc.metadata.get("chunk_id", "未知")
        retrieved_chunk_ids.append(chunk_id)
# 获得检索文本的距离d和索引i，把完全的文本内容添加到retrieved_full_texts，预览内容为retrieved_preview_texts
# 根据 FAISS 返回的索引 I，从 all_docs 中取出对应文本块
# 保存完整文本、预览文本、文本向量、FAISS距离、来源文件
    if not retrieved_vectors:
        return []

    retrieved_vectors = np.array(retrieved_vectors, dtype=np.float32)
# 返回的向量转为np数组
    sim_scores = cosine_similarity(
        query_vector,
        retrieved_vectors
    ).flatten()
# 计算当前问题向量 query_vector 与检索出的文本向量 retrieved_vectors 之间的余弦相似度
# flatten() 把二维结果转成一维数组，方便后面和文本一起打包
    rerank_pairs = []

    for text in retrieved_full_texts:
        rerank_pairs.append([query, text])
# 构造 Rerank 模型需要的输入格式：[问题, 完整文本]
# Rerank 需要同时看到 query 和 text，才能判断文本是否能回答这个问题
    rerank_scores = rerank_model.predict(rerank_pairs)
# 获取rerank值
    reranked_results = sorted(
        zip(
            retrieved_full_texts,
            retrieved_preview_texts,
            retrieved_vectors,
            rerank_scores,
            sim_scores,
            retrieved_distances,
            retrieved_sources,
            retrieved_chunk_ids
        ),
        key=lambda x: x[3],
        reverse=True
    )
# 把完整文本、预览文本、文本向量、Rerank分数、余弦相似度、FAISS距离、来源文件打包在一起
# 按 item[3]，也就是 Rerank 分数，从大到小排序
    return reranked_results
# 返回排序后的结果

def build_history_text(
    history: list,
    max_history: int
) -> str:
    """
    作用：
    把最近 max_history 轮历史对话拼接成字符串，供 prompt 使用。
    输入：
    history：
        历史对话列表。
        每一轮通常是一个字典：
        {
            "question": 用户问题,
            "answer": 模型回答
        }
    max_history：
        最多保留最近几轮历史对话。
    处理流程：
    1. 初始化空字符串 history_text。
    2. 使用 history[-max_history:] 取最近 max_history 轮历史。
    3. 遍历 recent_history。
    4. 把每一轮的 question 和 answer 拼接成字符串。
    5. 返回拼接后的 history_text。
    输出：
    history_text：
        拼接好的历史对话字符串，用于放进 prompt。
    """
    history_text = ""

    recent_history = history[-max_history:]

    for turn in recent_history:
        history_text += f"用户问题: {turn['question']}\n模型回答: {turn['answer']}\n\n"

    return history_text
# 从 history 中取最近 max_history 轮对话
# 把每轮 question 和 answer 拼接成 history_text，供 prompt 使用

def build_prompt(
    history_text: str,
    reranked_results: list,
    query: str
) -> tuple[str, str]:
    """
    作用：
    根据历史对话、检索上下文和当前问题，构造最终发送给大模型的 prompt。
    输入：
    history_text：
        已经拼接好的历史对话字符串。
    reranked_results：
        Rerank 排序后的检索结果。
        其中 item[0] 是完整文本，用来作为大模型回答问题的上下文。
    query：
        当前用户问题。
    处理流程：
    1. 从 reranked_results 中取出 item[0]，也就是完整文本。
    2. 把多个完整文本用换行符拼接成 context。
    3. 定义 prompt_template。
    4. 把 history_text、context、query 填入 prompt_template。
    5. 得到最终 prompt。
    6. 返回 prompt 和 context。
    输出：
    prompt：
        最终发送给大模型的提示词。
    context：
        由 Rerank 后的完整文本拼接成的检索上下文。
    """
    reranked_texts = [item[0] for item in reranked_results]

    context = "\n".join(reranked_texts)

    prompt_template = """
你是招聘专家，严格只使用提供内容回答问题。
历史对话:
{history}

检索内容:
{context}

当前问题:
{question}

请给出简明、准确、完整的答案。
"""

    prompt = prompt_template.format(
        history=history_text,
        context=context,
        question=query
    )

    return prompt, context
# 从 reranked_results 中取 item[0]，也就是完整文本，拼成 context
# 把 history_text、context、query 填入 prompt_template
# 返回最终 prompt 和 context


def build_source_reference_table(reranked_results: list, top_n: int = 3) -> list:
    """
    作用：
    根据 Rerank 后的检索结果，生成来源引用表格数据。

    输入：
    reranked_results：
        Rerank 排序后的检索结果列表。

    top_n：
        最多展示前几个来源，默认展示前 3 个。

    处理流程：
    1. 取 reranked_results 前 top_n 条结果。
    2. 从每条结果中取出来源文件、预览文本、Rerank分数、余弦相似度、FAISS距离。
    3. 整理成字典列表，方便在 Streamlit 中用 st.dataframe() 展示。

    输出：
    source_rows：
        来源引用表格数据列表。
    """
    source_rows = []

    for rank, item in enumerate(reranked_results[:top_n], start=1):
        source_rows.append({
            "排名": rank,
            "来源文件": item[6],
            "文本块编号": item[7],
            "Rerank评分": round(float(item[3]), 4),
            "余弦相似度": round(float(item[4]), 4),
            "FAISS距离": round(float(item[5]), 4),
            "内容预览": item[1]
        })

    return source_rows

def load_uploaded_file_text(file_name: str, file_bytes: bytes) -> str:
    """
    作用：
    根据上传文件的类型，提取文件中的文本内容。

    输入：
    uploaded_file：
        Streamlit 上传的文件对象，用来获取文件名和后缀。

    file_bytes：
        上传文件的原始 bytes。

    处理流程：
    1. 如果文件名以 .txt 结尾，就按文本文件解码。
    2. 如果文件名以 .pdf 结尾，就使用 PdfReader 读取每一页文本。
    3. 如果是不支持的文件类型，返回空字符串。

    输出：
    text：
        从文件中提取出来的文本字符串。
    """
    file_name = file_name.lower()

    if file_name.endswith(".txt"):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("gbk", errors="ignore")

    if file_name.endswith(".pdf"):
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)

        page_texts = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                page_texts.append(page_text)

        return "\n".join(page_texts)

    return ""

def build_document_summary(all_docs: list) -> list:
    """
    作用：
    根据 all_docs 统计当前知识库中每个文件的文本块数量。

    输入：
    all_docs：
        当前所有 Document 文本块列表。
        每个 doc 的 metadata 中应该包含 source 和 file_hash。

    处理流程：
    1. 遍历 all_docs。
    2. 从每个 doc.metadata 中取出 source 和 file_hash。
    3. 按 file_hash 统计每个文件的文本块数量。
    4. 整理成表格数据列表，方便在 Streamlit 中展示。

    输出：
    document_rows：
        文档统计表格数据列表。
        每一项包含：
        - 文件名
        - 文件hash
        - 文本块数量
    """
    document_map = {}

    for doc in all_docs:
        source = doc.metadata.get("source", "未知文件")
        file_hash = doc.metadata.get("file_hash", "未知hash")

        if file_hash not in document_map:
            document_map[file_hash] = {
                "文件名": source,
                "文件hash": file_hash,
                "文本块数量": 0
            }

        document_map[file_hash]["文本块数量"] += 1
    # 每遇到一个属于这个文件的doc，就让这个文件的文本块数量 + 1。
    document_rows = list(document_map.values())
    # 把字典转成列表。
    return document_rows

def filter_docs_and_vectors_by_sources(
    all_docs: list,
    vectors: np.ndarray,
    selected_sources: list
) -> tuple[list, np.ndarray, list]:
    """
    作用：
    根据用户选择的来源文件，过滤 all_docs 和 vectors。

    输入：
    all_docs：
        当前全部 Document 文本块列表。

    vectors：
        当前全部文本块向量数组。
        vectors[i] 和 all_docs[i] 一一对应。

    selected_sources：
        用户选择的文件名列表。
        如果为空列表，表示不筛选，使用全部文档。

    处理流程：
    1. 如果 selected_sources 为空，直接返回全部 all_docs、vectors 和原始索引映射。
    2. 遍历 all_docs。
    3. 判断每个 doc.metadata["source"] 是否在 selected_sources 中。
    4. 如果在，就保留该 doc、对应向量和它在原始 all_docs 中的索引。
    5. 返回过滤后的 docs、vectors、original_indices。

    输出：
    filtered_docs：
        筛选后的文本块列表。

    filtered_vectors：
        筛选后的向量数组。

    original_indices：
        筛选后每个文本块在原始 all_docs 中的索引。
    """
    if not selected_sources:
        original_indices = list(range(len(all_docs)))
        return all_docs, vectors, original_indices

    filtered_docs = []
    filtered_vectors = []
    original_indices = []

    for idx, doc in enumerate(all_docs):
        source = doc.metadata.get("source", "未知文件")

        if source in selected_sources:
            filtered_docs.append(doc)
            filtered_vectors.append(vectors[idx])
            original_indices.append(idx)

    if not filtered_vectors:
        return [], np.array([], dtype=np.float32), []

    filtered_vectors = np.array(filtered_vectors, dtype=np.float32)

    return filtered_docs, filtered_vectors, original_indices