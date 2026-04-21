# retrievers/vector_retriever.py
import pickle
from pathlib import Path

from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaEmbeddings

from core.exceptions import RagError
from core.logger import get_logger

logger = get_logger(__name__)


def build_vector_retriever(vectorstore_path: Path, embedding_model: str, top_k: int):
    """FAISS 기반 벡터 유사도 리트리버를 생성한다."""
    if not vectorstore_path.exists():
        raise RagError(
            f"인덱스 파일이 없습니다: {vectorstore_path}\n"
            "`python app/ingest.py`를 먼저 실행하세요."
        )

    try:
        logger.info(f"FAISS 인덱스 로드 중: {vectorstore_path}")
        embeddings = OllamaEmbeddings(model=embedding_model)
        vectorstore = FAISS.load_local(
            str(vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS 인덱스 로드 완료")
    except Exception as e:
        raise RagError(f"FAISS 인덱스 로드 실패: {e}") from e

    return vectorstore.as_retriever(search_kwargs={"k": top_k})