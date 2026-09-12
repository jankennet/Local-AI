"""
Unit tests for app.llm.web_search (DuckDuckGo search tool).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.llm import web_search
from app.llm.web_search import _parse_results, search_duckduckgo, _CACHE

FAKE_HTML = """<html><body>
<div class="result">
  <a class="result__a" href="http://example.com/cats">Cats Are Great</a>
  <a class="result__url">example.com/cats</a>
  <a class="result__snippet">Learn everything about domestic cats and their habits.</a>
</div>
<div class="result">
  <a class="result__a" href="http://example.com/dogs">Dog Care Guide</a>
  <a class="result__url">example.com/dogs</a>
  <a class="result__snippet">A practical guide to taking care of your dog.</a>
</div>
</body></html>"""

BLOCK_PAGE = """<html><head><title>duckduckgo - anomaly</title></head>
<body>Your search was flagged as an anomaly. Anomaly detection challenge. Please try again later.</body></html>"""

EMPTY_PAGE = "<html><body></body></html>"


class FakeResponse:
    headers = {}

    def __init__(self, status_code=200, text=FAKE_HTML):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        pass


class FakeClient:
    """Fake httpx.AsyncClient with an optional failure mode."""

    def __init__(self, timeout=None, follow_redirects=None, fail="none"):
        self.get_calls = 0
        self.fail = fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, *args, **kwargs):
        self.get_calls += 1
        if self.fail == "timeout":
            raise httpx.TimeoutException("timed out")
        if self.fail == "http_error":
            raise httpx.HTTPStatusError("500", request=httpx.Request("GET", "x"), response=FakeResponse())
        if self.fail == "blocked":
            return FakeResponse(status_code=202, text=BLOCK_PAGE)
        if self.fail == "empty":
            return FakeResponse(text=EMPTY_PAGE)
        return FakeResponse()


@pytest.fixture
def clear_cache():
    _CACHE.clear()
    yield
    _CACHE.clear()


@pytest.fixture
def fake_client_factory(monkeypatch):
    def install(fail="none"):
        instance = FakeClient(fail=fail)
        monkeypatch.setattr("app.llm.web_search.httpx.AsyncClient", lambda *a, **k: instance)
        return instance

    return install


class TestParse:
    def test_parse_results_extracts_fields(self):
        text = _parse_results(FAKE_HTML, "cats", 5)
        assert "Search results for: cats" in text
        assert "Cats Are Great" in text
        assert "example.com/cats" in text
        assert "domestic cats" in text

    def test_parse_results_no_results(self):
        text = _parse_results("<html><body></body></html>", "qq", 5)
        assert "no results found" in text


class TestSearchDuckDuckGo:
    def test_search_success(self, clear_cache, fake_client_factory):
        client = fake_client_factory()
        result = asyncio.run(search_duckduckgo("cats", 2))
        assert "Cats Are Great" in result
        assert "example.com/cats" in result
        assert client.get_calls == 1

    def test_result_is_numbered(self, clear_cache, fake_client_factory):
        fake_client_factory()
        result = asyncio.run(search_duckduckgo("pets", 5))
        assert "1. " in result
        assert "2. " in result

    def test_cache_hit_skips_network(self, clear_cache, fake_client_factory):
        client = fake_client_factory()
        first = asyncio.run(search_duckduckgo("cats", 2))
        second = asyncio.run(search_duckduckgo("cats", 2))
        assert first == second
        assert client.get_calls == 1

    def test_cache_keyed_by_query_and_count(self, clear_cache, fake_client_factory):
        client = fake_client_factory()
        asyncio.run(search_duckduckgo("cats", 2))
        asyncio.run(search_duckduckgo("cats", 5))
        asyncio.run(search_duckduckgo("dogs", 2))
        assert client.get_calls == 3

    def test_disabled_returns_error(self, monkeypatch, clear_cache):
        monkeypatch.setattr(
            web_search, "settings",
            SimpleNamespace(web_search_enabled=False, web_search_timeout=15.0),
        )
        result = asyncio.run(search_duckduckgo("cats"))
        assert result.startswith("Error:")
        assert "disabled" in result

    def test_timeout_returns_error(self, monkeypatch, clear_cache, fake_client_factory):
        fake_client_factory(fail="timeout")
        monkeypatch.setattr(web_search, "_fetch_curl_impersonate", AsyncMock(return_value=None))
        monkeypatch.setattr(
            web_search, "settings",
            SimpleNamespace(web_search_enabled=True, web_search_timeout=15.0),
        )
        result = asyncio.run(search_duckduckgo("cats"))
        assert result.startswith("Error:")
        assert "unavailable" in result

    def test_http_error_returns_error(self, clear_cache, fake_client_factory, monkeypatch):
        fake_client_factory(fail="http_error")
        monkeypatch.setattr(web_search, "_fetch_curl_impersonate", AsyncMock(return_value=None))
        result = asyncio.run(search_duckduckgo("cats"))
        assert result.startswith("Error:")
        assert "unavailable" in result

    def test_empty_query_returns_error(self, clear_cache, fake_client_factory):
        result = asyncio.run(search_duckduckgo("   "))
        assert result.startswith("Error:")


class TestBlockedFallback:
    def test_challenge_page_detected(self):
        assert web_search._looks_blocked(BLOCK_PAGE) is True
        assert web_search._looks_blocked(FAKE_HTML) is False

    def test_blocked_primary_falls_back_to_impersonate(self, clear_cache, fake_client_factory, monkeypatch):
        fake_client_factory(fail="blocked")
        monkeypatch.setattr(
            web_search, "_fetch_curl_impersonate",
            AsyncMock(return_value=(
                "Search results for: cats\n1. Cats Are Great\n"
                "   URL: example.com/cats\n   Learn about cats.\n", 1,
            )),
        )
        result = asyncio.run(search_duckduckgo("cats", 2))
        assert "Cats Are Great" in result
        assert web_search._fetch_curl_impersonate.await_count == 1

    def test_all_backends_blocked_returns_error(self, clear_cache, fake_client_factory, monkeypatch):
        fake_client_factory(fail="blocked")
        monkeypatch.setattr(web_search, "_fetch_curl_impersonate", AsyncMock(return_value=None))
        result = asyncio.run(search_duckduckgo("cats", 2))
        assert result.startswith("Error:")
        assert "unavailable" in result

    def test_genuine_empty_results_do_not_trigger_fallback(self, clear_cache, fake_client_factory, monkeypatch):
        fake_client_factory(fail="empty")
        mocked = AsyncMock()
        monkeypatch.setattr(web_search, "_fetch_curl_impersonate", mocked)
        result = asyncio.run(search_duckduckgo("qq", 2))
        assert "no results found" in result
        mocked.assert_not_awaited()