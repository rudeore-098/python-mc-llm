# retrievers/hybrid_retriever.py
# from langchain_classic.retrievers import EnsembleRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from core.exceptions import RetrievalError
from core.logger import get_logger

logger = get_logger(__name__)


def build_hybrid_retriever(
    vector_retriever: BaseRetriever,
    keyword_retriever: BaseRetriever,
    vector_weight: float,
    keyword_weight: float,
) -> EnsembleRetriever:
    """벡터 리트리버와 키워드 리트리버를 가중치로 결합한다."""
    try:
        retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[vector_weight, keyword_weight],
        )
        logger.info(
            f"하이브리드 리트리버 초기화 완료 "
            f"(FAISS {vector_weight*100:.0f}% + BM25 {keyword_weight*100:.0f}%)"
        )
        return retriever
    except Exception as e:
        raise RetrievalError(f"하이브리드 리트리버 조립 실패: {e}") from e