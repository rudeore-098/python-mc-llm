from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from api.schemas import RagChatInput, TextOutput
from api.dependencies import get_rag_chat_chain
from core.exceptions import InternalServerError

router = APIRouter(prefix="/rag/chat", tags=["RAG 대화"])

_ROLE_MAP = {
    "human": HumanMessage,
    "ai": AIMessage,
    "system": SystemMessage,
}


def _to_lc_messages(items):
    return [_ROLE_MAP[m.role](content=m.content) for m in items]


@router.post(
    "",
    response_model=TextOutput,
    summary="RAG 대화형 질의응답",
    description="이전 대화 기록을 포함한 하이브리드 RAG 질의응답입니다. 대화 기록이 있으면 질문을 자동으로 재구성해 검색합니다.",
)
async def rag_chat(body: RagChatInput, chain=Depends(get_rag_chat_chain)):
    try:
        result = await chain.ainvoke({
            "question": body.question,
            "chat_history": _to_lc_messages(body.messages),
        })
        return TextOutput(output=result)
    except Exception as e:
        raise InternalServerError(detail=str(e))


@router.post(
    "/stream",
    summary="RAG 대화형 질의응답 (스트리밍)",
    description="RAG 대화 응답을 SSE 스트림으로 반환합니다.",
    response_class=StreamingResponse,
)
async def rag_chat_stream(body: RagChatInput, chain=Depends(get_rag_chat_chain)):
    async def generator():
        try:
            async for chunk in chain.astream({
                "question": body.question,
                "chat_history": _to_lc_messages(body.messages),
            }):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
