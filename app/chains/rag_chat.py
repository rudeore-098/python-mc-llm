from pathlib import Path
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from core.prompt_loader import load_chat_prompt
from langchain.agents import create_tool_calling_agent, AgentExecutor

from chains.base import BaseChain
from core.exceptions import RetrievalError
from core.logger import get_logger
from retrievers import (
    build_vector_retriever,
    build_keyword_retriever,
    build_hybrid_retriever,
)
from tools import build_retriever_tools

logger = get_logger(__name__)

VECTORSTORE_PATH = Path(__file__).parent.parent.parent / "data" / "vectorstore"
DOCS_PATH = VECTORSTORE_PATH / "docs.pkl"
EMBEDDING_MODEL = "bge-m3"

VECTOR_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4
TOP_K = 4


def format_docs(docs):
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content>"
        f"<page>{doc.metadata['page']}</page>"
        f"<source>{doc.metadata['source']}</source></document>"
        for doc in docs
    )


class RagChatChain(BaseChain):
    """
    Tool Calling Agent 기반 RAG 대화 체인.

    입력: {"question": str, "chat_history": List[BaseMessage]}
    LLM이 검색 필요 여부를 판단해 search_documents tool을 호출합니다.
    """

    def setup(self):
        try:
            faiss_retriever = build_vector_retriever(
                vectorstore_path=VECTORSTORE_PATH,
                embedding_model=EMBEDDING_MODEL,
                top_k=TOP_K,
            )
            bm25_retriever = build_keyword_retriever(
                docs_path=DOCS_PATH,
                top_k=TOP_K,
            )
            retriever = build_hybrid_retriever(
                vector_retriever=faiss_retriever,
                keyword_retriever=bm25_retriever,
                vector_weight=VECTOR_WEIGHT,
                keyword_weight=KEYWORD_WEIGHT,
            )
        except Exception as e:
            raise RetrievalError(f"리트리버 초기화 실패: {e}") from e

        logger.info(f"RAG Chat Agent 구성 중 (모델: {self.model}, temperature: {self.temperature})")

        tools = build_retriever_tools(retriever, format_docs)
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        base_dir = Path(__file__).parent.parent
        prompt = load_chat_prompt(base_dir / "prompts/rag-agent.yaml")

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 외부 API의 "question"/"chat_history" 키를 AgentExecutor 입력으로 매핑하고
        # AgentExecutor 출력의 "output" 키만 반환합니다.
        chain = (
            RunnableLambda(lambda x: {
                "input": x["question"],
                "chat_history": x.get("chat_history", []),
            })
            | agent_executor
            | RunnableLambda(lambda x: x["output"])
        )
        return chain
