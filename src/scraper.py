from __future__ import annotations

import re
import time
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": "RAGDatabase-Bot/1.0 (educational project; respectful scraper)",
}
_TIMEOUT = 15  # seconds


@dataclass
class ScrapeResult:
    url: str
    title: str
    text: str           # clean extracted text
    word_count: int
    robots_blocked: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.robots_blocked and not self.error and len(self.text.strip()) > 50


def _check_robots(url: str) -> bool:
    """Return True if scraping *url* is allowed by robots.txt."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch(_HEADERS["User-Agent"], url)
    except Exception:
        return True  # if robots.txt is unreachable, assume allowed


def _extract_text(html: str) -> tuple[str, str]:
    """Return (title, clean_text) from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Remove noise tags
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe", "svg"]):
        tag.decompose()

    # Prefer <main> or <article> if present, else use <body>
    container = soup.find("main") or soup.find("article") or soup.find("body")
    if not container:
        return title, ""

    # Extract text, collapse whitespace
    raw = container.get_text(separator="\n")
    lines = [line.strip() for line in raw.splitlines()]
    # Drop very short lines (menus, button labels etc.)
    lines = [l for l in lines if len(l) > 30]
    text = "\n".join(lines)
    # Collapse 3+ consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return title, text.strip()


def scrape(url: str) -> ScrapeResult:
    """
    Scrape *url*, respect robots.txt, and return clean extracted text.

    Returns a ScrapeResult — always check .ok before using .text.
    """
    # 1. Normalise URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # 2. robots.txt check
    if not _check_robots(url):
        return ScrapeResult(
            url=url, title="", text="", word_count=0, robots_blocked=True,
            error="Blocked by robots.txt — this site does not allow scraping.",
        )

    # 3. Fetch
    try:
        time.sleep(0.5)  # polite delay
        response = httpx.get(url, headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        return ScrapeResult(url=url, title="", text="", word_count=0,
                            error=f"HTTP {e.response.status_code} — {e.response.reason_phrase}")
    except Exception as e:
        return ScrapeResult(url=url, title="", text="", word_count=0, error=str(e))

    # 4. Check content type — if it's a PDF, signal caller to use PDF parser
    content_type = response.headers.get("content-type", "")
    if "application/pdf" in content_type:
        return ScrapeResult(url=url, title="", text="__PDF__", word_count=0)

    # 5. Extract clean text
    title, text = _extract_text(response.text)
    if not text:
        return ScrapeResult(url=url, title=title, text="", word_count=0,
                            error="Could not extract readable text from this page. "
                                  "It may be JavaScript-heavy or require login.")

    word_count = len(text.split())
    return ScrapeResult(url=url, title=title or url, text=text, word_count=word_count)
