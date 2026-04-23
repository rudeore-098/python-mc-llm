from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from chains.base import BaseChain
from chains.utils import format_docs
from core.logger import get_logger
from core.prompt_loader import load_system_message
from tools import build_retriever_tools

logger = get_logger(__name__)

_MAX_ITERATIONS = 5
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rag-agent.yaml"


def _strip_eos(text: str) -> str:
    return text.replace("<eos>", "").strip()


class _AgentLoop(Runnable):
    """
    AgentExecutor 없이 tool loop을 직접 구현합니다.
    - ainvoke: 최종 답변 문자열 반환
    - astream:  최종 LLM 응답을 토큰 단위로 스트리밍
    """

    def __init__(self, llm_with_tools, llm, tools_map, system_content):
        self._llm_with_tools = llm_with_tools
        self._llm = llm
        self._tools_map = tools_map
        self._system_content = system_content

    def _build_messages(self, inputs: dict) -> list:
        messages = [SystemMessage(content=self._system_content)]
        messages.extend(inputs.get("chat_history", []))
        messages.append(HumanMessage(content=inputs["question"]))
        return messages

    async def _run_tool_loop(self, inputs: dict) -> tuple[list, str | None]:
        """tool 호출이 없어질 때까지 반복. 마지막 응답이 직접 답변이면 함께 반환."""
        messages = self._build_messages(inputs)

        for _ in range(_MAX_ITERATIONS):
            response = await self._llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return messages, _strip_eos(response.content)

            for tc in response.tool_calls:
                result = self._tools_map[tc["name"]].invoke(tc["args"])
                logger.debug(f"Tool '{tc['name']}' 호출, 결과 길이: {len(result)}")
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        return messages, None

    def invoke(self, input, config=None, **kwargs):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(input, config))

    async def ainvoke(self, input, config=None, **kwargs) -> str:
        messages, answer = await self._run_tool_loop(input)
        if answer is not None:
            return answer
        response = await self._llm_with_tools.ainvoke(messages)
        return _strip_eos(response.content)

    async def astream(self, input, config=None, **kwargs):
        messages, answer = await self._run_tool_loop(input)
        if answer is not None:
            yield answer
            return
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
        tools_map = {t.name: t for t in tools}

        llm = ChatOllama(model=self.model, temperature=self.temperature)
        llm_with_tools = llm.bind_tools(tools)
        system_content = load_system_message(_PROMPT_PATH)

        return _AgentLoop(
            llm_with_tools=llm_with_tools,
            llm=llm,
            tools_map=tools_map,
            system_content=system_content,
        )
