import os
from functools import lru_cache
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_openai import ChatOpenAI


EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def load_embedding_and_rerank_models():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    rerank_model = CrossEncoder(RERANK_MODEL_NAME)
    return embed_model, rerank_model


@lru_cache(maxsize=1)
def load_llm():
    load_dotenv()

    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_API_BASE")
    )

    return llm