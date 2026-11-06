import os
from functools import lru_cache
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_openai import ChatOpenAI


EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def load_embedding_model():
    if os.getenv("RAG_V2_ENABLED") == "1":
        return SentenceTransformer(os.environ['RAG_V2_EMBED_PATH'],local_files_only=True,device=os.getenv('RAG_V2_MODEL_DEVICE','cpu'))
    return SentenceTransformer(EMBED_MODEL_NAME)

@lru_cache(maxsize=1)
def load_embedding_and_rerank_models():
    # All application and integration callers share the same native model objects.
    os.environ.setdefault('HF_DEACTIVATE_ASYNC_LOAD','1')
    embed_model=load_embedding_model()
    rerank_model=(CrossEncoder(os.environ['RAG_V2_RERANK_PATH'],local_files_only=True,device=os.getenv('RAG_V2_MODEL_DEVICE','cpu'))
                  if os.getenv('RAG_V2_ENABLED')=='1' else CrossEncoder(RERANK_MODEL_NAME))
    return embed_model,rerank_model


@lru_cache(maxsize=1)
def load_llm():
    if os.getenv("RAG_LLM_BUDGETED_REAL")=="1":
        from llm_budget import BudgetedChatModel
        return BudgetedChatModel(os.environ["DEEPSEEK_MODEL"])
    if os.getenv("RAG_V2_ISOLATED_LLM")=="1":
        from langchain_core.language_models.fake import FakeListLLM
        return FakeListLLM(responses=["LOCAL_TEST_SUBSTITUTE_NO_EXTERNAL_API"])
    load_dotenv()

    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_API_BASE")
    )

    return llm


@lru_cache(maxsize=1)
def load_hyde_llm():
    if os.getenv("RAG_LLM_BUDGETED_REAL")=="1":
        from llm_budget import BudgetedChatModel
        return BudgetedChatModel(os.getenv("RAG_V2_HYDE_MODEL",os.environ["DEEPSEEK_MODEL"]),256)
    if os.getenv("RAG_V2_ISOLATED_LLM")=="1":return load_llm()
    return ChatOpenAI(model=os.getenv("RAG_V2_HYDE_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
                      api_key=os.environ["DEEPSEEK_API_KEY"], base_url=os.environ["DEEPSEEK_API_BASE"],
                      timeout=8, max_retries=0, max_tokens=256, temperature=0)
