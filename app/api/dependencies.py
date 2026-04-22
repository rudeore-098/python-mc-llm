from functools import lru_cache
from chains.chains import ChatChain, TopicChain, LLM, Translator
from chains.rag import RagChain


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
    return RagChain().create()


@lru_cache()
def get_chat_chain():
    return ChatChain().create()
