import pickle
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import load_prompt
from langchain_ollama import OllamaEmbeddings, ChatOllama

from base import BaseChain
from core.exceptions import RagError, RetrievalError
from core.logger import get_logger

logger = get_logger(__name__)

# ingest.py 실행 시 저장된 FAISS 인덱스 및 문서 청크 경로
VECTORSTORE_PATH = Path(__file__).parent.parent / "data" / "vectorstore"
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
        # 사전 빌드된 인덱스 존재 여부 확인
        if not VECTORSTORE_PATH.exists() or not DOCS_PATH.exists():
            raise RagError(
                f"인덱스 파일이 없습니다: {VECTORSTORE_PATH}\n"
                "`python app/ingest.py`를 먼저 실행하세요."
            )

        try:
            # FAISS 인덱스 로드 (벡터 유사도 검색용)
            # 임베딩 모델은 ingest.py와 반드시 동일해야 함
            logger.info(f"FAISS 인덱스 로드 중: {VECTORSTORE_PATH}")
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = FAISS.load_local(
                str(VECTORSTORE_PATH),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS 인덱스 로드 완료")
        except Exception as e:
            raise RagError(f"FAISS 인덱스 로드 실패: {e}") from e

        try:
            # pickle로 저장된 문서 청크 로드 (BM25 키워드 검색용)
            logger.info(f"문서 청크 로드 중: {DOCS_PATH}")
            with open(DOCS_PATH, "rb") as f:
                docs = pickle.load(f)
            logger.info(f"문서 청크 로드 완료: {len(docs)}개")
        except Exception as e:
            raise RagError(f"문서 청크 로드 실패: {e}") from e

        try:
            # 벡터 유사도 기반 리트리버 (FAISS)
            faiss_retriever = vectorstore.as_retriever(
                search_kwargs={"k": TOP_K}
            )

            # 키워드 기반 리트리버 (BM25)
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = TOP_K

            # 하이브리드 리트리버: 벡터 60% + 키워드 40% 가중치로 결합
            retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever],
                weights=[VECTOR_WEIGHT, KEYWORD_WEIGHT],
            )
            logger.info(
                f"하이브리드 리트리버 초기화 완료 "
                f"(FAISS {VECTOR_WEIGHT*100:.0f}% + BM25 {KEYWORD_WEIGHT*100:.0f}%, TOP_K={TOP_K})"
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
