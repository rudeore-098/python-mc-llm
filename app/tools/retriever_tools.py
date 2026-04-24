from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(description="검색할 질문 또는 키워드")


def build_retriever_tools(retriever, format_docs_fn) -> list:
    """retriever를 주입받아 tool 목록을 반환합니다."""

    def search_documents(query: str) -> str:
        """벡터+키워드 하이브리드 검색으로 관련 문서를 찾습니다.
        사용자 질문에 답하기 위해 문서에서 정보를 검색할 때 사용하세요."""
        docs = retriever.invoke(query)
        if not docs:
            return "관련 문서를 찾을 수 없습니다."
        return format_docs_fn(docs)

    return [
        StructuredTool.from_function(
            func=search_documents,
            name="search_documents",
            description=(
                "벡터+키워드 하이브리드 검색으로 관련 문서를 찾습니다. "
                "사용자 질문에 답하기 위해 문서에서 정보를 검색할 때 사용하세요."
            ),
            args_schema=SearchInput,
        )
    ]
