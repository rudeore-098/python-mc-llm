import asyncio
import re
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from chains.base import BaseChain
from core.logger import get_logger
from core.prompt_loader import load_system_message

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rag-agent.yaml"


def _strip_eos(text: str) -> str:
    return re.sub(r"</?eos>", "", text).strip()


class _MessageBuilder:
    def __init__(self, system_content: str):
        self._system_content = system_content

    def build_initial(self, inputs: dict) -> list:
        messages = [SystemMessage(content=self._system_content)]
        messages.extend(inputs.get("chat_history", []))
        messages.append(HumanMessage(content=inputs["question"]))
        return messages


class _ToolCaller:
    def __init__(self, llm_with_tools, tools_map: dict):
        self._llm = llm_with_tools
        self._tools_map = tools_map

    def _execute_tools(self, tool_calls: list) -> list[tuple[str, str]]:
        seen: set[str] = set()
        results = []
        for tc in tool_calls:
            result = self._tools_map[tc["name"]].invoke(tc["args"])
            if result in seen:
                logger.info(f"[tool-phase] '{tc['name']}' 중복 결과 무시")
                continue
            seen.add(result)
            results.append((tc["id"], result))
            logger.info(f"[tool-phase] '{tc['name']}' 결과 길이: {len(result)}")
        return results

    async def run(self, messages: list) -> str:
        response = await self._llm.ainvoke(messages)
        logger.info(f"[tool-phase] tool_calls={len(response.tool_calls)}")

        if not response.tool_calls:
            return _strip_eos(response.content)

        messages.append(response)
        for tc_id, result in self._execute_tools(response.tool_calls):
            messages.append(ToolMessage(content=result, tool_call_id=tc_id))

        final = await self._llm.ainvoke(messages)
        logger.info(f"[tool-phase] 최종 응답 raw={repr(final.content[:120])}")
        return _strip_eos(final.content)

    async def stream(self, messages: list):
        response = await self._llm.ainvoke(messages)

        if not response.tool_calls:
            yield _strip_eos(response.content)
            return

        messages.append(response)
        for tc_id, result in self._execute_tools(response.tool_calls):
            messages.append(ToolMessage(content=result, tool_call_id=tc_id))

        async for chunk in self._llm.astream(messages):
            if chunk.content:
                cleaned = _strip_eos(chunk.content)
                if cleaned:
                    yield cleaned


class _AgentLoop(Runnable):
    def __init__(self, tool_caller: _ToolCaller, message_builder: _MessageBuilder):
        self._tool_caller = tool_caller
        self._message_builder = message_builder

    def invoke(self, input, config=None, **kwargs):
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(input, config))

    async def ainvoke(self, input, config=None, **kwargs) -> str:
        messages = self._message_builder.build_initial(input)
        return await self._tool_caller.run(messages)

    async def astream(self, input, config=None, **kwargs):
        messages = self._message_builder.build_initial(input)
        async for chunk in self._tool_caller.stream(messages):
            yield chunk


class RagChatChain(BaseChain):
    """
    입력: {"question": str, "chat_history": List[BaseMessage]}
    LLM이 search_documents tool 호출 여부를 판단하고, 결과를 받아 최종 답변을 생성합니다.
    """

    def __init__(self, tools: list, **kwargs):
        super().__init__(**kwargs)
        self.tools = tools

    def setup(self) -> _AgentLoop:
        logger.info(f"RAG Chat Agent 구성 중 (모델: {self.model}, temperature: {self.temperature})")

        llm = ChatOllama(model=self.model, temperature=self.temperature)

        return _AgentLoop(
            tool_caller=_ToolCaller(
                llm_with_tools=llm.bind_tools(self.tools),
                tools_map={t.name: t for t in self.tools},
            ),
            message_builder=_MessageBuilder(
                system_content=load_system_message(_PROMPT_PATH),
            ),
        )
