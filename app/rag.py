from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import load_prompt
from langchain_ollama import OllamaEmbeddings, ChatOllama

from base import BaseChain
from core.exceptions import RagError, RetrievalError

# ingest.py 실행 시 저장된 FAISS 인덱스 경로
VECTORSTORE_PATH = Path(__file__).parent.parent / "data" / "vectorstore"
EMBEDDING_MODEL = "bge-m3"


def format_docs(docs):
    # 검색된 문서를 XML 태그로 포맷팅 (출처 파일명·페이지 포함)
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content>"
        f"<page>{doc.metadata['page']}</page>"
        f"<source>{doc.metadata['source']}</source></document>"
        for doc in docs
    )


class RagChain(BaseChain):

    def setup(self):
        # 사전 빌드된 벡터스토어 존재 여부 확인
        if not VECTORSTORE_PATH.exists():
            raise RagError(
                f"벡터스토어가 없습니다: {VECTORSTORE_PATH}\n"
                "`python app/ingest.py`를 먼저 실행하세요."
            )

        try:
            # 디스크에서 FAISS 인덱스 로드 (임베딩 모델은 인제스트와 동일해야 함)
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = FAISS.load_local(
                str(VECTORSTORE_PATH),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            raise RagError(f"벡터스토어 로드 실패: {e}") from e

        try:
            retriever = vectorstore.as_retriever()
        except Exception as e:
            raise RetrievalError(f"리트리버 초기화 실패: {e}") from e

        # RAG 프롬프트 및 LLM 체인 구성
        prompt = load_prompt("prompts/rag-exaone.yaml", encoding="utf-8")
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain
