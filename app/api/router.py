from fastapi import APIRouter
from api.routes import healt, translate, llm, topic, rag, rag_chat, chat

api_router = APIRouter(prefix="/api")
api_router.include_router(healt.router)
api_router.include_router(translate.router)
api_router.include_router(llm.router)
api_router.include_router(topic.router)
api_router.include_router(rag.router)
api_router.include_router(rag_chat.router)
api_router.include_router(chat.router)
