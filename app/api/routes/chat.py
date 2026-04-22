from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from api.schemas import ChatInput, TextOutput
from api.dependencies import get_chat_chain
from core.exceptions import InternalServerError

router = APIRouter(prefix="/chat", tags=["대화"])

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
    summary="대화형 채팅",
    description="다중 턴 대화를 지원하는 채팅 엔드포인트입니다. messages 배열에 이전 대화 내역을 포함하세요.",
)
async def chat(body: ChatInput, chain=Depends(get_chat_chain)):
    try:
        messages = _to_lc_messages(body.messages)
        result = await chain.ainvoke({"messages": messages})
        return TextOutput(output=result)
    except Exception as e:
        raise InternalServerError(detail=str(e))


@router.post(
    "/stream",
    summary="대화형 채팅 (스트리밍)",
    description="채팅 응답을 SSE 스트림으로 반환합니다.",
    response_class=StreamingResponse,
)
async def chat_stream(body: ChatInput, chain=Depends(get_chat_chain)):
    async def generator():
        try:
            messages = _to_lc_messages(body.messages)
            async for chunk in chain.astream({"messages": messages}):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
