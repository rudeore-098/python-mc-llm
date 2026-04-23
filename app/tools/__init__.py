from .retriever_tools import build_retriever_tools
from chains.utils import format_docs


def build_rag_tools(retriever) -> list:
    """RAG 에이전트가 사용할 tool 목록을 조합해 반환합니다."""
    return [
        *build_retriever_tools(retriever, format_docs),
        # 웹검색 등 추가 tool은 여기에 append
    ]


__all__ = ["build_rag_tools", "build_retriever_tools"]
