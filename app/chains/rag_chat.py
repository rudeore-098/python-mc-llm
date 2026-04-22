from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama
from core.prompt_loader import load_chat_prompt

from chains.base import BaseChain
from core.exceptions import RetrievalError
from core.logger import get_logger
from retrievers import (
    build_vector_retriever,
    build_keyword_retriever,
    build_hybrid_retriever,
)

logger = get_logger(__name__)

VECTORSTORE_PATH = Path(__file__).parent.parent.parent / "data" / "vectorstore"
DOCS_PATH = VECTORSTORE_PATH / "docs.pkl"
EMBEDDING_MODEL = "bge-m3"

VECTOR_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4
TOP_K = 4


def format_docs(docs):
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content>"
        f"<page>{doc.metadata['page']}</page>"
        f"<source>{doc.metadata['source']}</source></document>"
        for doc in docs
    )


class RagChatChain(BaseChain):
    """
    대화 히스토리를 지원하는 RAG 체인.

    입력: {"question": str, "chat_history": List[BaseMessage]}
    - chat_history가 비어 있으면 question을 그대로 검색에 사용
    - chat_history가 있으면 LLM으로 독립적인 질문으로 재구성 후 검색
    """

    def setup(self):
        try:
            faiss_retriever = build_vector_retriever(
                vectorstore_path=VECTORSTORE_PATH,
                embedding_model=EMBEDDING_MODEL,
                top_k=TOP_K,
            )
            bm25_retriever = build_keyword_retriever(
                docs_path=DOCS_PATH,
                top_k=TOP_K,
            )
            retriever = build_hybrid_retriever(
                vector_retriever=faiss_retriever,
                keyword_retriever=bm25_retriever,
                vector_weight=VECTOR_WEIGHT,
                keyword_weight=KEYWORD_WEIGHT,
            )
        except Exception as e:
            raise RetrievalError(f"리트리버 초기화 실패: {e}") from e

        logger.info(f"RAG Chat 체인 구성 중 (모델: {self.model}, temperature: {self.temperature})")

        base_dir = Path(__file__).parent.parent
        reformulation_prompt = load_chat_prompt(base_dir / "prompts/rag-reformulation.yaml")
        rag_prompt = load_chat_prompt(base_dir / "prompts/rag-chat.yaml")
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        reformulation_chain = reformulation_prompt | llm | StrOutputParser()

        async def get_standalone_question(inputs: dict) -> str:
            chat_history = inputs.get("chat_history", [])
            if not chat_history:
                return inputs["question"]
            return await reformulation_chain.ainvoke({
                "question": inputs["question"],
                "chat_history": chat_history,
            })

        chain = (
            RunnablePassthrough.assign(
                standalone_question=RunnableLambda(get_standalone_question)
            )
            | RunnablePassthrough.assign(
                context=lambda x: format_docs(retriever.invoke(x["standalone_question"]))
            )
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        logger.info(f"RAG_prompt  : {rag_prompt}")
        return chain
