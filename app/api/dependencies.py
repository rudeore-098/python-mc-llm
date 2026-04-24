from functools import lru_cache
from chains.chains import ChatChain, TopicChain, LLM, Translator
from chains.rag import RagChain
from chains.rag_chat import RagChatChain
from core.settings import settings
from retrievers import build_vector_retriever, build_keyword_retriever, build_hybrid_retriever
from tools import build_rag_tools


@lru_cache()
def get_rag_retriever():
    r = settings.retriever
    vectorstore_path = r.vectorstore_path
    faiss_retriever = build_vector_retriever(
        vectorstore_path=vectorstore_path,
        embedding_model=r.embedding_model,
        top_k=r.top_k,
    )
    bm25_retriever = build_keyword_retriever(
        docs_path=vectorstore_path / "docs.pkl",
        top_k=r.top_k,
    )
    return build_hybrid_retriever(
        vector_retriever=faiss_retriever,
        keyword_retriever=bm25_retriever,
        vector_weight=r.vector_weight,
        keyword_weight=r.keyword_weight,
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
    return build_rag_tools(get_rag_retriever(), settings.web.allowed_domains)


@lru_cache()
def get_rag_chat_chain():
    return RagChatChain(tools=get_rag_tools()).create()


@lru_cache()
def get_chat_chain():
    return ChatChain().create()
