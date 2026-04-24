from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from core.logger import get_logger

logger = get_logger(__name__)

_PAGE_TEXT_LIMIT = 8000
_MAX_LINKS = 30


class FetchPageInput(BaseModel):
    url: str = Field(description="내용을 가져올 웹 페이지의 URL")


class FetchLinksInput(BaseModel):
    url: str = Field(description="링크 목록을 추출할 웹 페이지의 URL")


def _get_html(url: str) -> str:
    resp = httpx.get(url, timeout=10, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)[:_PAGE_TEXT_LIMIT]


def _is_allowed(url: str, allowed_domains: list[str] | None) -> bool:
    if allowed_domains is None:
        return True
    host = urlparse(url).hostname or ""
    return any(host == d or host.endswith(f".{d}") for d in allowed_domains)


def _build_fetch_page_tool(allowed_domains: list[str] | None) -> StructuredTool:
    def fetch_page(url: str) -> str:
        if not _is_allowed(url, allowed_domains):
            logger.warning(f"[fetch_page] 허용되지 않은 도메인: {url}")
            return f"허용되지 않은 도메인입니다. 허용 도메인: {', '.join(allowed_domains)}"
        logger.info(f"[fetch_page] 요청: {url}")
        try:
            text = _extract_text(_get_html(url))
            logger.info(f"[fetch_page] 완료: {url} (길이: {len(text)})")
            return text
        except httpx.HTTPError as e:
            logger.error(f"[fetch_page] 실패: {url} — {e}")
            return f"페이지 로드 실패: {e}"

    domain_hint = f" 허용 도메인: {', '.join(allowed_domains)}." if allowed_domains else ""
    return StructuredTool.from_function(
        func=fetch_page,
        name="fetch_page",
        description=f"URL의 웹 페이지 본문 내용을 가져옵니다.{domain_hint}",
        args_schema=FetchPageInput,
    )


def _build_fetch_links_tool(allowed_domains: list[str] | None) -> StructuredTool:
    def fetch_page_links(url: str) -> str:
        if not _is_allowed(url, allowed_domains):
            logger.warning(f"[fetch_page_links] 허용되지 않은 도메인: {url}")
            return f"허용되지 않은 도메인입니다. 허용 도메인: {', '.join(allowed_domains)}"
        logger.info(f"[fetch_page_links] 요청: {url}")
        try:
            soup = BeautifulSoup(_get_html(url), "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                text = a.get_text(strip=True)
                if href.startswith("http") and _is_allowed(href, allowed_domains) and text:
                    links.append(f"[{text}]({href})")
            logger.info(f"[fetch_page_links] 완료: {url} (링크 수: {len(links)})")
            if not links:
                return "링크가 없습니다."
            return "\n".join(links[:_MAX_LINKS])
        except httpx.HTTPError as e:
            logger.error(f"[fetch_page_links] 실패: {url} — {e}")
            return f"페이지 로드 실패: {e}"

    domain_hint = f" 허용 도메인: {', '.join(allowed_domains)}." if allowed_domains else ""
    return StructuredTool.from_function(
        func=fetch_page_links,
        name="fetch_page_links",
        description=(
            f"웹 페이지에서 링크 목록을 추출합니다. "
            f"페이지 내 어떤 링크가 있는지 파악한 뒤 fetch_page로 원하는 링크를 열람하세요.{domain_hint}"
        ),
        args_schema=FetchLinksInput,
    )


def build_web_search_tools(allowed_domains: list[str] | None = None) -> list:
    """
    웹 탐색 tool 목록을 반환합니다.

    allowed_domains=None  → 제한 없음 (개발/인터넷 환경)
    allowed_domains=[...] → 지정 도메인만 허용 (내부망 환경)
    """
    return [
        _build_fetch_page_tool(allowed_domains),
        _build_fetch_links_tool(allowed_domains),
    ]
