# retrievers/keyword_retriever.py
import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever

from core.exceptions import RagError
from core.logger import get_logger

logger = get_logger(__name__)


def build_keyword_retriever(docs_path: Path, top_k: int):
    """BM25 기반 키워드 리트리버를 생성한다."""
    if not docs_path.exists():
        raise RagError(f"청크 파일이 없습니다: {docs_path}")

    try:
        logger.info(f"문서 청크 로드 중: {docs_path}")
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
        logger.info(f"문서 청크 로드 완료: {len(docs)}개")
    except Exception as e:
        raise RagError(f"문서 청크 로드 실패: {e}") from e

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k
    return bm25_retriever