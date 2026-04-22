from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import TextInput, TextOutput
from api.dependencies import get_translate_chain

router = APIRouter(prefix="/translate", tags=["번역"])


@router.post(
    "",
    response_model=TextOutput,
    summary="텍스트 번역",
    description="입력 텍스트를 한국어로 번역합니다.",
)
async def translate(body: TextInput, chain=Depends(get_translate_chain)):
    try:
        result = await chain.ainvoke({"input": body.input})
        return TextOutput(output=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/stream",
    summary="텍스트 번역 (스트리밍)",
    description="번역 결과를 SSE 스트림으로 반환합니다.",
    response_class=StreamingResponse,
)
async def translate_stream(body: TextInput, chain=Depends(get_translate_chain)):
    async def generator():
        try:
            async for chunk in chain.astream({"input": body.input}):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
