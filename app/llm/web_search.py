"""
web_search.py

DuckDuckGo web search for the agent tool loop. No API key required.

Design:
  - Returns plain strings, never raises — tool errors are surfaced as
    text so the agent can react to them (matches tools.py convention).
  - Primary backend: the HTML endpoint over httpx. DuckDuckGo sometimes
    serves a 202 anti-bot challenge page (HTTP 200 body with "anomaly"
    markers) to programmatic clients. When that happens we fall back to
    curl-impersonate (`curl_chrome*`, a Chrome TLS-impersonating curl),
    which presents a real Chrome fingerprint and passes the challenge —
    no browser driver needed.
  - A backend is treated as *blocked* (unavailable) when it returns a
    non-200 status or a challenge page. Only a genuine 200 page that
    parses zero results produces "no results found." If every backend is
    blocked the tool returns a clear `Error:` string so the agent stops
    searching instead of mistaking an infra outage for a missing fact.
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

# curl-impersonate binaries ship as curl_chrome<major> / curl_firefox<major>.
# Tried in order; the first one present and unblocked serves the request.
_IMPERSONATE_BINS = [
    "curl_chrome136",
    "curl_chrome142",
    "curl_chrome146",
    "curl_chrome150",
]
_CURL_TIMEOUT_SECONDS = 15

# substring markers the DDG anti-bot/anomaly challenge page contains.
_CHALLENGE_MARKERS = ("anomaly", "challenge")

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


def _looks_blocked(html: str) -> bool:
    """True when the page is a DuckDuckGo anti-bot challenge, not results."""
    low = html.lower()
    return any(marker in low for marker in _CHALLENGE_MARKERS)


def _count_results(html: str) -> int:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    return len(soup.select(".result"))


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


async def _fetch_httpx(query: str, num_results: int) -> Optional[Tuple[str, int]]:
    """Primary backend. Returns (text, parsed_count) or None when blocked."""
    try:
        params = httpx.QueryParams({"q": query})
        timeout = settings.web_search_timeout
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(SEARCH_URL, params=params, headers={"User-Agent": _UA})
            resp.raise_for_status()
            html = resp.text
    except httpx.TimeoutException:
        logger.warning("web_search timeout for query=%r", query)
        return None
    except httpx.HTTPStatusError as e:
        logger.warning("web_search HTTP error: %s", e)
        return None
    except httpx.HTTPError as e:
        logger.warning("web_search request error: %s", e)
        return None

    if resp.status_code != 200:
        logger.warning(
            "web_search DDG html responded HTTP %d for query=%r (anti-bot block?)",
            resp.status_code, query,
        )
        return None

    count = _count_results(html)
    if count == 0 and _looks_blocked(html):
        logger.warning("web_search DDG html returned a challenge page for query=%r", query)
        return None

    return _parse_results(html, query, num_results), count


async def _fetch_curl_impersonate(query: str, num_results: int) -> Optional[Tuple[str, int]]:
    """Fallback backend: curl-impersonate with a real Chrome TLS fingerprint."""
    for bin_name in _IMPERSONATE_BINS:
        try:
            proc = await asyncio.create_subprocess_exec(
                bin_name,
                "-s", "--location",
                "--max-time", str(_CURL_TIMEOUT_SECONDS),
                "--get", "--data-urlencode", f"q={query}",
                "-A", _UA,
                SEARCH_URL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_CURL_TIMEOUT_SECONDS + 5
            )
            html = stdout.decode("utf-8", errors="replace")
        except Exception as e:  # missing binary, timeout, spawn error
            logger.warning("web_search impersonate %s failed for query=%r: %s", bin_name, query, e)
            continue

        if not html:
            continue
        count = _count_results(html)
        if count == 0 and _looks_blocked(html):
            logger.warning("web_search impersonate %s got a challenge page for query=%r", bin_name, query)
            continue

        logger.info("web_search primary blocked; served by %s", bin_name)
        return _parse_results(html, query, num_results), count

    return None


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

    result = await _fetch_httpx(query, num_results)
    backend = "duckduckgo-html"
    if result is None:
        result = await _fetch_curl_impersonate(query, num_results)
        backend = "curl-impersonate"

    if result is None:
        logger.warning(
            "web_search unavailable for query=%r (all providers blocked/unreachable)", query
        )
        return ("Error: web search is temporarily unavailable "
                "(all providers blocked or unreachable)")

    text, count = result
    _cache_put(cache_key, text)
    logger.info("web_search ok: query=%r backend=%r results_returned=%d", query, backend, count)
    return text