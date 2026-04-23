import re
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from chains.base import BaseChain
from chains.utils import format_docs
from core.logger import get_logger
from core.prompt_loader import load_system_message
from tools import build_retriever_tools

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rag-agent.yaml"


def _strip_eos(text: str) -> str:
    return re.sub(r"</?eos>", "", text).strip()


class _MessageBuilder:
    """메시지 리스트 구성 책임."""

    def __init__(self, system_content: str):
        self._system_content = system_content

    def build_initial(self, inputs: dict) -> list:
        messages = [SystemMessage(content=self._system_content)]
        messages.extend(inputs.get("chat_history", []))
        messages.append(HumanMessage(content=inputs["question"]))
        return messages

    def build_with_context(self, inputs: dict, tool_results: list[str]) -> list:
        context = "\n\n---\n\n".join(tool_results)
        messages = [SystemMessage(content=self._system_content)]
        messages.extend(inputs.get("chat_history", []))
        messages.append(HumanMessage(
            content=f"{inputs['question']}\n\n[검색된 문서]\n{context}"
        ))
        return messages


class _ToolCaller:
    """
    LLM tool calling + 결과 수집 책임.

    Gemma 4는 ToolMessage를 받은 후 최종 답변 생성에 실패하므로,
    tool 결과만 수집하고 ToolMessage는 최종 답변 단계에 전달하지 않습니다.
    """

    def __init__(self, llm_with_tools, tools_map: dict):
        self._llm = llm_with_tools
        self._tools_map = tools_map

    async def run(self, messages: list) -> tuple[str | None, list[str]]:
        """
        Returns:
            (direct_answer, [])  — 도구 미호출, 모델이 직접 답변
            (None, tool_results) — 도구 호출, 결과 수집 완료
        """
        response = await self._llm.ainvoke(messages)
        logger.info(f"[tool-phase] tool_calls={len(response.tool_calls)}")

        if not response.tool_calls:
            direct = _strip_eos(response.content)
            return (direct or None), []

        seen: set[str] = set()
        tool_results: list[str] = []

        for tc in response.tool_calls:
            result = self._tools_map[tc["name"]].invoke(tc["args"])
            if result in seen:
                logger.info(f"[tool-phase] '{tc['name']}' 중복 결과 무시")
                continue
            seen.add(result)
            tool_results.append(result)
            logger.info(f"[tool-phase] '{tc['name']}' 결과 길이: {len(result)}")

        return None, tool_results


class _AgentLoop(Runnable):
    """
    _ToolCaller와 _MessageBuilder를 조합하는 Runnable 오케스트레이터.
    - 도구 호출 여부에 따라 직접 답변 또는 context 임베딩 후 plain LLM 호출
    """

    def __init__(self, tool_caller: _ToolCaller, llm, message_builder: _MessageBuilder):
        self._tool_caller = tool_caller
        self._llm = llm
        self._message_builder = message_builder

    def invoke(self, input, config=None, **kwargs):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(input, config))

    async def ainvoke(self, input, config=None, **kwargs) -> str:
        initial = self._message_builder.build_initial(input)
        direct, tool_results = await self._tool_caller.run(initial)

        if not tool_results:
            return direct or ""

        messages = self._message_builder.build_with_context(input, tool_results)
        logger.info(f"[answer-phase] context 길이: {sum(len(r) for r in tool_results)}")
        response = await self._llm.ainvoke(messages)
        return _strip_eos(response.content)

    async def astream(self, input, config=None, **kwargs):
        initial = self._message_builder.build_initial(input)
        direct, tool_results = await self._tool_caller.run(initial)

        if not tool_results:
            yield direct or ""
            return

        messages = self._message_builder.build_with_context(input, tool_results)
        logger.info(f"[answer-phase] context 길이: {sum(len(r) for r in tool_results)}")
        async for chunk in self._llm.astream(messages):
            if chunk.content:
                cleaned = _strip_eos(chunk.content)
                if cleaned:
                    yield cleaned


class RagChatChain(BaseChain):
    """
    Tool loop 기반 RAG 대화 체인.

    입력: {"question": str, "chat_history": List[BaseMessage]}
    LLM이 search_documents tool 호출 여부를 판단하고, 결과를 받아 최종 답변을 생성합니다.
    """

    def __init__(self, retriever, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever

    def setup(self) -> _AgentLoop:
        logger.info(f"RAG Chat Agent 구성 중 (모델: {self.model}, temperature: {self.temperature})")

        tools = build_retriever_tools(self.retriever, format_docs)
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        return _AgentLoop(
            tool_caller=_ToolCaller(
                llm_with_tools=llm.bind_tools(tools),
                tools_map={t.name: t for t in tools},
            ),
            llm=llm,
            message_builder=_MessageBuilder(
                system_content=load_system_message(_PROMPT_PATH),
            ),
        )
