from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.router import api_router
from api.dependencies import (
    get_translate_chain,
    get_llm_chain,
    get_topic_chain,
    get_rag_chain,
    get_chat_chain,
)
from core.exceptions import AppHTTPException
from dotenv import load_dotenv

load_dotenv()


# lifespan: 서버 시작 시 모든 체인을 미리 초기화합니다.
# lru_cache 덕분에 여기서 한 번 호출하면 이후 요청에서 재사용됩니다.
# 첫 요청에서 느린 초기화(벡터 인덱스 로드 등)가 발생하지 않습니다.
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_translate_chain()
    get_llm_chain()
    get_topic_chain()
    get_rag_chain()
    get_chat_chain()
    yield  # yield 이후는 서버 종료 시 실행 (현재는 정리 로직 없음)


# FastAPI 앱 생성
# title/description/version은 /docs Swagger UI 상단에 표시됩니다.
app = FastAPI(
    title="LangChain Ollama API",
    description="""
로컬 Ollama LLM을 활용한 LangChain 기반 REST API 서버입니다.

## 엔드포인트 목록

| 경로 | 설명 |
|---|---|
| `POST /api/translate` | 텍스트를 한국어로 번역 |
| `POST /api/llm` | LLM 직접 호출 |
| `POST /api/topic` | 주제 설명 생성 |
| `POST /api/rag` | 하이브리드 RAG 질의응답 |
| `POST /api/chat` | 다중 턴 대화 |
| `POST /api/rag/chat` | 하이브리드 RAG 질의응답 + 다중 턴 대화 |


각 엔드포인트에 `/stream`을 붙이면 SSE 스트리밍 응답을 받을 수 있습니다.
예: `POST /api/chat/stream`
""",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
# allow_origins=["*"]는 개발용입니다. 운영 환경에서는 실제 도메인을 명시하세요.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.exception_handler(AppHTTPException)
async def app_http_exception_handler(request: Request, exc: AppHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.error_code, "detail": exc.detail},
    )


# api/ 하위의 모든 라우터를 등록합니다.
# router.py에서 prefix="/api"를 붙이므로 /api/translate 형태가 됩니다.
app.include_router(api_router)


# 루트("/") 접근 시 Swagger 문서로 리다이렉트
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    # python fastapi_server.py 로 실행
    uvicorn.run(app, host="0.0.0.0", port=8001)  # LangServe(8000)와 포트 분리
