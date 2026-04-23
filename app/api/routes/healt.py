from fastapi import APIRouter


router = APIRouter(prefix="/health", tags=["서버 상태"])

@router.get("", summary="서버 상태 확인", description="서버가 정상적으로 작동하는지 확인합니다.")
async def healthy():
    return {"status": "health"}