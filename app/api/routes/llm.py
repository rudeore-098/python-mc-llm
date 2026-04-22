from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import TextInput, TextOutput
from api.dependencies import get_llm_chain

router = APIRouter(prefix="/llm", tags=["LLM"])


@router.post(
    "",
    response_model=TextOutput,
    summary="LLM 직접 호출",
    description="프롬프트 없이 LLM에 직접 메시지를 전달합니다.",
)
async def llm_invoke(body: TextInput, chain=Depends(get_llm_chain)):
    try:
        result = await chain.ainvoke(body.input)
        return TextOutput(output=result.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/stream",
    summary="LLM 직접 호출 (스트리밍)",
    description="LLM 응답을 SSE 스트림으로 반환합니다.",
    response_class=StreamingResponse,
)
async def llm_stream(body: TextInput, chain=Depends(get_llm_chain)):
    async def generator():
        try:
            async for chunk in chain.astream(body.input):
                yield f"data: {chunk.content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
