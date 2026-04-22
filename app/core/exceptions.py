class AppError(Exception):
    """프로젝트의 모든 커스텀 예외의 베이스 클래스."""


class IngestionError(AppError):
    """ingestion 파이프라인 전용 예외. ingest.py에서 사용."""


class RetrievalError(AppError):
    """retrieval 파이프라인 전용 예외. retrievers/에서 사용."""


class RagError(AppError):
    """RAG 파이프라인 전용 예외 (인덱스 로드 실패 등). rag.py에서 사용."""


class ModelLoadError(AppError):
    """모델 로드 실패 전용 예외. model_loader.py에서 사용."""