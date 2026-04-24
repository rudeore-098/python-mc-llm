import yaml
from pathlib import Path
from pydantic import BaseModel

from core.logger import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


class LLMSettings(BaseModel):
    model: str = "gemma4:e4b"
    temperature: float = 0.0


class RetrieverSettings(BaseModel):
    vectorstore_path: Path = Path(__file__).parent.parent.parent / "data" / "vectorstore"
    embedding_model: str = "bge-m3"
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    top_k: int = 4


class WebSettings(BaseModel):
    allowed_domains: list[str] | None = None


class DatabaseSettings(BaseModel):
    url: str = "sqlite:///./app.db"
    echo: bool = False


class Settings(BaseModel):
    llm: LLMSettings = LLMSettings()
    retriever: RetrieverSettings = RetrieverSettings()
    web: WebSettings = WebSettings()
    database: DatabaseSettings = DatabaseSettings()


def _load() -> Settings:
    if not _CONFIG_PATH.exists():
        logger.warning(f"config.yaml 없음 — 기본값 사용 ({_CONFIG_PATH})")
        return Settings()
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    s = Settings(**data)
    logger.info(f"설정 로드 완료: model={s.llm.model}, embedding={s.retriever.embedding_model}, web_domains={s.web.allowed_domains}")
    return s


settings = _load()
