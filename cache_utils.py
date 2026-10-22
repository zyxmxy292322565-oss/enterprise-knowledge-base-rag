import os
import pickle
import hashlib
import faiss
import numpy as np

RAG_QUERY_CACHE_PREFIX = "policy_rag:"


def get_cache_paths(
    embed_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    index_type: str
) -> tuple[str, str, str]:
    """
        作用：
        根据当前实验参数生成对应的缓存文件路径。

        输入：
        embed_model_name：
            当前使用的 embedding 模型名称。
            不同模型生成的向量维度或效果可能不同，所以缓存文件需要区分。

        chunk_size：
            当前文本切块大小。
            不同 chunk_size 会导致切出来的文本块不同，所以缓存文件需要区分。

        chunk_overlap：
            当前文本块重叠大小。
            不同 chunk_overlap 也会导致文本块不同，所以缓存文件需要区分。

        index_type：
            当前选择的 FAISS 索引类型。
            不同索引类型保存出来的 index 文件不同，所以缓存文件需要区分。

        处理流程：
        1. 把 embed_model_name、chunk_size、chunk_overlap、index_type 拼接成 cache_config 字符串。
        2. 对 cache_config 进行 MD5 计算，生成 cache_key。
        3. 创建 cache 文件夹，如果已经存在则不会报错。
        4. 根据 cache_key 生成三个缓存文件路径：
           - vectors_cache_file：保存向量
           - all_docs_cache_file：保存文本块
           - index_cache_file：保存 FAISS 索引
        5. 返回这三个缓存文件路径。

        输出：
        vectors_cache_file：
            向量缓存文件路径，例如 cache/vectors_xxxxxxxx.pkl。

        all_docs_cache_file：
            文本块缓存文件路径，例如 cache/all_docs_xxxxxxxx.pkl。

        index_cache_file：
            FAISS 索引缓存文件路径，例如 cache/faiss_index_xxxxxxxx.index。
        """
    cache_config = f"{embed_model_name}_{chunk_size}_{chunk_overlap}_{index_type}"
    cache_key = hashlib.md5(cache_config.encode("utf-8")).hexdigest()[:8]

    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    vectors_cache_file = os.path.join(cache_dir, f"vectors_{cache_key}.pkl")
    all_docs_cache_file = os.path.join(cache_dir, f"all_docs_{cache_key}.pkl")
    index_cache_file = os.path.join(cache_dir, f"faiss_index_{cache_key}.index")

    return vectors_cache_file, all_docs_cache_file, index_cache_file


def clear_cache_files(cache_files: list[str]) -> None:
    """
        作用：
        删除指定的缓存文件。

        输入：
        cache_files：
            需要删除的缓存文件路径列表。
            一般包括：
            - vectors_cache_file
            - all_docs_cache_file
            - index_cache_file

        处理流程：
        1. 遍历 cache_files。
        2. 判断每个 cache_file 是否存在。
        3. 如果存在，就使用 os.remove() 删除。
        4. 如果不存在，就跳过，不报错。

        输出：
        无返回值。
        这个函数只负责删除文件。

        注意：
        它不会删除整个 cache 文件夹。
        它只删除传进来的具体缓存文件。
        """
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)


def load_cache(
    vectors_cache_file: str,
    all_docs_cache_file: str,
    index_cache_file: str
) -> tuple:
    """
        作用：
        从本地缓存文件中加载之前保存好的向量、文本块和 FAISS 索引。

        输入：
        vectors_cache_file：
            向量缓存文件路径。

        all_docs_cache_file：
            文本块缓存文件路径。

        index_cache_file：
            FAISS 索引缓存文件路径。

        处理流程：
        1. 使用 pickle 读取 vectors_cache_file，恢复 vectors。
        2. 使用 pickle 读取 all_docs_cache_file，恢复 all_docs。
        3. 使用 faiss.read_index() 读取 index_cache_file，恢复 FAISS 索引。
        4. 返回 all_docs、vectors、index。

        输出：
        all_docs：
            已缓存的所有 Document 文本块。

        vectors：
            已缓存的所有文本块向量。

        index：
            已缓存的 FAISS 索引对象。

        注意：
        如果缓存文件不存在，会抛出 FileNotFoundError。
        这个异常一般在 main.py 里处理。
        """
    vectors = pickle.load(open(vectors_cache_file, "rb"))
    vectors = np.asarray(vectors, dtype=np.float32)

    all_docs = pickle.load(open(all_docs_cache_file, "rb"))

    index = faiss.read_index(index_cache_file)

    return all_docs, vectors, index


def save_cache(
    vectors: np.ndarray,
    all_docs: list,
    index,
    vectors_cache_file: str,
    all_docs_cache_file: str,
    index_cache_file: str
) -> None:
    """
        作用：
        把当前完整的向量、文本块和 FAISS 索引保存到本地缓存文件中。

        输入：
        vectors：
            当前全部文本块对应的向量。
            注意：这里保存的是完整 vectors，不只是 new_vectors。

        all_docs：
            当前全部 Document 文本块。
            注意：这里保存的是完整 all_docs，不只是 new_docs。

        index：
            当前完整的 FAISS 索引对象。

        vectors_cache_file：
            向量缓存文件路径。

        all_docs_cache_file：
            文本块缓存文件路径。

        index_cache_file：
            FAISS 索引缓存文件路径。

        处理流程：
        1. 使用 pickle 把 vectors 保存到 vectors_cache_file。
        2. 使用 pickle 把 all_docs 保存到 all_docs_cache_file。
        3. 使用 faiss.write_index() 把 FAISS 索引保存到 index_cache_file。

        输出：
        无返回值。
        这个函数的作用是把数据写入本地文件。
        """
    with open(vectors_cache_file, "wb") as f:
        pickle.dump(vectors, f)

    with open(all_docs_cache_file, "wb") as f:
        pickle.dump(all_docs, f)

    faiss.write_index(index, index_cache_file)


def recover_processed_file_hashes(all_docs: list) -> set:
    """
        作用：
        从已加载的 all_docs 中恢复已经处理过的文件 MD5 集合。

        输入：
        all_docs：
            从缓存中加载出来的所有 Document 文本块。
            每个 doc 的 metadata 里通常保存了 file_hash。

        处理流程：
        1. 创建一个空集合 processed_file_hashes。
        2. 遍历 all_docs 中的每个 doc。
        3. 从 doc.metadata 中取出 file_hash。
        4. 如果 file_hash 存在，就添加到 processed_file_hashes。
        5. 返回恢复后的 processed_file_hashes。

        输出：
        processed_file_hashes：
            已经处理过的文件 MD5 集合。

        注意：
        这个函数不是重新生成 MD5。
        它只是从 doc.metadata["file_hash"] 里恢复之前保存过的 MD5。
        """
    processed_file_hashes = set()

    for doc in all_docs:
        file_hash = doc.metadata.get("file_hash")
        if file_hash:
            processed_file_hashes.add(file_hash)

    return processed_file_hashes

def build_policy_cache_key(
    question: str,
    top_k: int,
    faiss_multiplier: int,
    min_rerank_score: float,
    policy_version: str
):
    import hashlib

    cache_config = (
        f"{question}_"
        f"{top_k}_"
        f"{faiss_multiplier}_"
        f"{min_rerank_score}_"
        f"{policy_version}"
    )

    cache_hash = hashlib.sha256(
        cache_config.encode("utf-8")
    ).hexdigest()

    return (
        f"{RAG_QUERY_CACHE_PREFIX}{cache_hash}"
    )
