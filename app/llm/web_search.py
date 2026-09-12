"""
web_search.py

DuckDuckGo web search for the agent tool loop. No API key required, uses
the HTML endpoint (same approach as the Discord bot's MCP web scraper).

Design:
  - Returns plain strings, never raises — tool errors are surfaced as
    text so the agent can react to them (matches tools.py convention).
  - Gated by LLM_WEB_SEARCH_ENABLED (default true). Disabled returns a
    clear error string, mirroring run_bash's LLM_ALLOW_SHELL posture.
  - In-process results cache with a TTL so retries don't hammer the
    provider while still serving fresh results.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

SEARCH_URL = "https://html.duckduckgo.com/html/"

# query (lowercased) + num_results -> (timestamp, result_text)
_CACHE: Dict[Tuple[str, int], Tuple[float, str]] = {}
_CACHE_TTL_SECONDS = 300.0
_MAX_CACHE_ENTRIES = 64

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _cache_get(key: Tuple[str, int]) -> Optional[str]:
    hit = _CACHE.get(key)
    if hit is None:
        return None
    ts, text = hit
    if time.time() - ts > _CACHE_TTL_SECONDS:
        del _CACHE[key]
        return None
    return text


def _cache_put(key: Tuple[str, int], text: str) -> None:
    if len(_CACHE) >= _MAX_CACHE_ENTRIES:
        # Drop the oldest entry (dict preserves insertion order).
        oldest = next(iter(_CACHE))
        del _CACHE[oldest]
    _CACHE[key] = (time.time(), text)


def _parse_results(html: str, query: str, num_results: int) -> str:
    """Extract title/url/snippet from DuckDuckGo's HTML results page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    results = soup.select(".result")[:num_results]

    if not results:
        return f"Search for '{query}': no results found."

    output = [f"Search results for: {query}\n"]
    for i, result in enumerate(results):
        title_el = result.select_one(".result__title") or result.select_one(".result__a")
        snippet_el = result.select_one(".result__snippet")
        url_el = result.select_one(".result__url")

        title = title_el.get_text(strip=True) if title_el else "No title"
        snippet = snippet_el.get_text(strip=True) if snippet_el else "No snippet"
        url = url_el.get_text(strip=True) if url_el else "No URL"

        output.append(f"{i + 1}. {title}")
        output.append(f"   URL: {url}")
        output.append(f"   {snippet}\n")

    return "\n".join(output)


async def search_duckduckgo(query: str, num_results: int = 5) -> str:
    """Search the web and return top results as formatted text.

    Returns a plain string suitable for the model or a tool-result error
    message. Disabled via LLM_WEB_SEARCH_ENABLED=false.
    """
    num_results = max(1, min(int(num_results or 5), 10))

    if not settings.web_search_enabled:
        logger.warning("web_search called but LLM_WEB_SEARCH_ENABLED=false")
        return ("Error: web search is disabled on this server "
                "(set LLM_WEB_SEARCH_ENABLED=true to enable)")

    cache_key = (query.strip().lower(), num_results)
    if not cache_key[0]:
        return "Error: empty search query"

    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info("web_search cache hit: query=%r num_results=%d", query, num_results)
        return cached

    try:
        params = httpx.QueryParams({"q": query})
        timeout = settings.web_search_timeout
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(SEARCH_URL, params=params, headers={"User-Agent": _UA})
            resp.raise_for_status()
            html = resp.text
    except httpx.TimeoutException:
        logger.warning("web_search timeout for query=%r", query)
        return f"Error: web search timed out after {settings.web_search_timeout:.0f}s for '{query}'"
    except httpx.HTTPStatusError as e:
        logger.warning("web_search HTTP error: %s", e)
        return f"Error: web search failed (HTTP {e.response.status_code})"
    except httpx.HTTPError as e:
        logger.warning("web_search request error: %s", e)
        return f"Error: web search failed: {type(e).__name__}"

    try:
        text = _parse_results(html, query, num_results)
    except Exception as e:
        logger.warning("web_search parse error: %s", e)
        text = f"Error: failed to parse search results for '{query}'"

    _cache_put(cache_key, text)
    logger.info("web_search ok: query=%r results_returned=%d", query, num_results)
    return text