class IngestionError(Exception):
    """
    ingestion 파이프라인 전용 예외.
    """
    pass


class RetrievalError(Exception):
    """retrieval 파이프라인 전용 예외. retrieval.py에서 사용."""
    pass

class RagError(Exception):
    """RAG 파이프라인 전용 예외 (인덱스 로드 실패 등). rag.py에서 사용."""
    pass

class ModelLoadError(Exception):
    """모델 로드 실패 전용 예외. model_loader.py에서 사용."""
    pass