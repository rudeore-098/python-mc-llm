import pickle
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import load_prompt
from langchain_ollama import ChatOllama

from chains.base import BaseChain
from core.exceptions import RagError, RetrievalError
from core.logger import get_logger
from retrievers import (
    build_vector_retriever,
    build_keyword_retriever,
    build_hybrid_retriever,
)

logger = get_logger(__name__)

# ingest.py 실행 시 저장된 FAISS 인덱스 및 문서 청크 경로
VECTORSTORE_PATH = Path(__file__).parent.parent.parent / "data" / "vectorstore"
DOCS_PATH = VECTORSTORE_PATH / "docs.pkl"
EMBEDDING_MODEL = "bge-m3"

# 하이브리드 검색 가중치 (벡터 : 키워드)
VECTOR_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4
TOP_K = 4


def format_docs(docs):
    # 검색된 문서를 XML 태그로 포맷팅 (출처 파일명·페이지 포함)
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content>"
        f"<page>{doc.metadata['page']}</page>"
        f"<source>{doc.metadata['source']}</source></document>"
        for doc in docs
    )


class RagChain(BaseChain):

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

            # 앙상블 조립
            retriever = build_hybrid_retriever(
                vector_retriever=faiss_retriever,
                keyword_retriever=bm25_retriever,
                vector_weight=VECTOR_WEIGHT,
                keyword_weight=KEYWORD_WEIGHT,
            )
        except Exception as e:
            raise RetrievalError(f"리트리버 초기화 실패: {e}") from e

        # RAG 프롬프트 및 LLM 체인 구성
        logger.info(f"RAG 체인 구성 중 (모델: {self.model}, temperature: {self.temperature})")
        prompt = load_prompt("prompts/rag-exaone.yaml", encoding="utf-8")
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain