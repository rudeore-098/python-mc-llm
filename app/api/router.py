from fastapi import APIRouter
from api.routes import translate, llm, topic, rag, chat

api_router = APIRouter(prefix="/api")
api_router.include_router(translate.router)
api_router.include_router(llm.router)
api_router.include_router(topic.router)
api_router.include_router(rag.router)
api_router.include_router(chat.router)
