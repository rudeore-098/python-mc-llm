from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.schemas import TopicInput, TextOutput
from api.dependencies import get_topic_chain
from core.exceptions import InternalServerError

router = APIRouter(prefix="/topic", tags=["주제 설명"])


@router.post(
    "",
    response_model=TextOutput,
    summary="주제 설명",
    description="주어진 주제를 간결하게 설명합니다.",
)
async def explain_topic(body: TopicInput, chain=Depends(get_topic_chain)):
    try:
        result = await chain.ainvoke({"topic": body.topic})
        return TextOutput(output=result)
    except Exception as e:
        raise InternalServerError(detail=str(e))


@router.post(
    "/stream",
    summary="주제 설명 (스트리밍)",
    description="주제 설명을 SSE 스트림으로 반환합니다.",
    response_class=StreamingResponse,
)
async def explain_topic_stream(body: TopicInput, chain=Depends(get_topic_chain)):
    async def generator():
        try:
            async for chunk in chain.astream({"topic": body.topic}):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
