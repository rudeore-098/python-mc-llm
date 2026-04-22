from typing import List, Literal
from pydantic import BaseModel, Field


class TextInput(BaseModel):
    input: str = Field(..., description="입력 텍스트", examples=["Hello, how are you?"])


class TopicInput(BaseModel):
    topic: str = Field(..., description="설명할 주제", examples=["양자 컴퓨터"])


class RagInput(BaseModel):
    question: str = Field(..., description="질문", examples=["이 문서에서 중요한 내용은 무엇인가요?"])


class MessageItem(BaseModel):
    role: Literal["human", "ai", "system"] = Field(..., description="메시지 역할")
    content: str = Field(..., description="메시지 내용")


class ChatInput(BaseModel):
    messages: List[MessageItem] = Field(
        ...,
        description="대화 메시지 목록",
        examples=[[{"role": "human", "content": "안녕하세요!"}]],
    )


class RagChatInput(BaseModel):
    question: str = Field(..., description="현재 질문", examples=["이 문서에서 중요한 내용은 무엇인가요?"])
    messages: List[MessageItem] = Field(
        default=[],
        description="이전 대화 기록",
        examples=[[{"role": "human", "content": "안녕하세요!"}, {"role": "ai", "content": "안녕하세요! 무엇을 도와드릴까요?"}]],
    )


class TextOutput(BaseModel):
    output: str = Field(..., description="LLM 응답")
