from functools import lru_cache
from pathlib import Path
from chains.chains import ChatChain, TopicChain, LLM, Translator
from chains.rag import RagChain
from chains.rag_chat import RagChatChain
from retrievers import build_vector_retriever, build_keyword_retriever, build_hybrid_retriever
from tools import build_rag_tools

_VECTORSTORE_PATH = Path(__file__).parent.parent.parent / "data" / "vectorstore"
_DOCS_PATH = _VECTORSTORE_PATH / "docs.pkl"
_EMBEDDING_MODEL = "bge-m3"
_VECTOR_WEIGHT = 0.6
_KEYWORD_WEIGHT = 0.4
_TOP_K = 4


@lru_cache()
def get_rag_retriever():
    faiss_retriever = build_vector_retriever(
        vectorstore_path=_VECTORSTORE_PATH,
        embedding_model=_EMBEDDING_MODEL,
        top_k=_TOP_K,
    )
    bm25_retriever = build_keyword_retriever(
        docs_path=_DOCS_PATH,
        top_k=_TOP_K,
    )
    return build_hybrid_retriever(
        vector_retriever=faiss_retriever,
        keyword_retriever=bm25_retriever,
        vector_weight=_VECTOR_WEIGHT,
        keyword_weight=_KEYWORD_WEIGHT,
    )


@lru_cache()
def get_translate_chain():
    return Translator().create()


@lru_cache()
def get_llm_chain():
    return LLM().create()


@lru_cache()
def get_topic_chain():
    return TopicChain().create()


@lru_cache()
def get_rag_chain():
    return RagChain(retriever=get_rag_retriever()).create()


@lru_cache()
def get_rag_tools():
    return build_rag_tools(get_rag_retriever())


@lru_cache()
def get_rag_chat_chain():
    return RagChatChain(tools=get_rag_tools()).create()


@lru_cache()
def get_chat_chain():
    return ChatChain().create()
