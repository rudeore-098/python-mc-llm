from .retriever_tools import build_retriever_tools
from .web_search_tools import build_web_search_tools
from chains.utils import format_docs
from core.logger import get_logger

logger = get_logger(__name__)


def build_rag_tools(retriever, allowed_domains: list[str] | None = None) -> list:
    """
    RAG 에이전트가 사용할 tool 목록을 조합해 반환합니다.

    allowed_domains=None  → 제한 없음 (개발/인터넷 환경)
    allowed_domains=[...] → 지정 도메인만 허용 (내부망 환경)
    """
    tools = [
        *build_retriever_tools(retriever, format_docs),
        *build_web_search_tools(allowed_domains),
    ]
    logger.info(f"RAG tools 구성 완료: {[t.name for t in tools]}")
    return tools


__all__ = ["build_rag_tools", "build_retriever_tools", "build_web_search_tools"]
