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


# --- HTTP 예외 ---

class AppHTTPException(Exception):
    """API 레이어에서 사용하는 HTTP 예외 베이스 클래스."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    detail: str = "Internal Server Error"

    def __init__(self, detail: str | None = None, error_code: str | None = None):
        self.detail = detail if detail is not None else self.__class__.detail
        self.error_code = error_code if error_code is not None else self.__class__.error_code
        super().__init__(self.detail)


class BadRequestError(AppHTTPException):
    status_code = 400
    error_code = "BAD_REQUEST"
    detail = "Bad Request"


class NotFoundError(AppHTTPException):
    status_code = 404
    error_code = "NOT_FOUND"
    detail = "Not Found"


class InternalServerError(AppHTTPException):
    status_code = 500
    error_code = "INTERNAL_ERROR"
    detail = "Internal Server Error"


class ModelUnavailableError(AppHTTPException):
    """모델 로드/호출 실패 시 사용."""
    status_code = 503
    error_code = "MODEL_UNAVAILABLE"
    detail = "Model is unavailable"