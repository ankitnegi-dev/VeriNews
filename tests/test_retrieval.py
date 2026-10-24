"""Offline tests for news retrieval.

Every test uses a saved response-shape fixture or a local fake provider. No test
needs an API key or a network connection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

import app.retrieval as retrieval
from app.retrieval import (
    DailyLimitError,
    GNewsProvider,
    MissingAPIKeyError,
    NewsAPIProvider,
    RequestBudget,
    RetrievalCache,
    SearchProvider,
    retrieve,
)
from app.schemas import Article

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


class StaticProvider(SearchProvider):
    name = "static"

    def __init__(self, articles: list[Article], name: str = "static") -> None:
        self.articles = articles
        self.name = name
        self.calls = 0

    def search(self, query: str, max_results: int = 10) -> list[Article]:
        self.calls += 1
        return self.articles[:max_results]


def _article(
    title: str,
    url: str,
    content: str = "A useful article body.",
    domain: str = "example.com",
) -> Article:
    return Article(
        title=title,
        source_domain=domain,
        url=url,
        published_at=None,
        content=content,
    )


def _budget(tmp_path: Path, limit: int = 80) -> RequestBudget:
    return RequestBudget(
        tmp_path / "request_count.json",
        limit=limit,
        today=lambda: "2026-10-24",
    )


def test_newsapi_normalizes_articles_and_request_params(monkeypatch, tmp_path):
    payload = _fixture("newsapi_tom_skerritt_1.json")
    captured: dict[str, object] = {}

    def fake_get(url, *, params, headers, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse(payload)

    monkeypatch.setattr(retrieval.requests, "get", fake_get)
    budget = _budget(tmp_path)
    provider = NewsAPIProvider(api_key="test-key", budget=budget)

    articles = provider.search("Tom Skerritt", max_results=20)

    assert captured["url"] == retrieval.config.NEWSAPI_BASE_URL
    assert captured["params"]["language"] == "en"
    assert captured["params"]["sortBy"] == "relevancy"
    assert captured["params"]["pageSize"] == 20
    assert "apiKey" not in captured["params"]
    assert captured["headers"] == {"X-Api-Key": "test-key"}
    assert budget.usage()[0] == 1

    # The provider returns the response articles. retrieve() deduplicates after
    # all queries are merged, so the provider layer keeps all three.
    assert len(articles) == 3
    assert articles[0].source_domain == "abcnews.com"
    assert articles[0].content.count("Veteran actor Tom Skerritt") == 1
    assert "natural ca" in articles[0].content
    assert "[+177 chars]" not in articles[0].content
    assert articles[2].source_domain == "economictimes.indiatimes.com"


def test_live_fixtures_are_sanitized():
    allowed_article_keys = {
        "source",
        "title",
        "description",
        "url",
        "publishedAt",
        "content",
    }
    for name in (
        "newsapi_tom_skerritt_1.json",
        "newsapi_tom_skerritt_2.json",
        "newsapi_tom_skerritt_3.json",
        "gnews_tom_skerritt.json",
    ):
        raw = (FIXTURES / name).read_text(encoding="utf-8")
        assert "\u2013" not in raw
        assert "\u2014" not in raw

        payload = _fixture(name)
        serialized = json.dumps(payload).lower()
        assert "apikey" not in serialized
        assert "api_key" not in serialized
        assert payload["articles"]
        for article in payload["articles"]:
            assert set(article) == allowed_article_keys
            assert set(article["source"]) == {"name"}
            assert article["source"]["name"]
            assert article["title"]
            assert article["url"].startswith("http")
            assert article["publishedAt"]
            assert article["content"]


def test_newsapi_empty_response_returns_empty_list(monkeypatch, tmp_path):
    empty_payload = {"status": "ok", "totalResults": 0, "articles": []}

    def fake_get(url, *, params, headers, timeout):
        return FakeResponse(empty_payload)

    monkeypatch.setattr(retrieval.requests, "get", fake_get)
    provider = NewsAPIProvider(api_key="test-key", budget=_budget(tmp_path))

    assert provider.search("no matching story", max_results=20) == []


def test_gnews_normalizes_articles_and_request_params(monkeypatch, tmp_path):
    payload = _fixture("gnews_tom_skerritt.json")
    captured: dict[str, object] = {}

    def fake_get(url, *, params, headers, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        return FakeResponse(payload)

    monkeypatch.setattr(retrieval.requests, "get", fake_get)
    provider = GNewsProvider(api_key="gnews-test-key", budget=_budget(tmp_path))

    articles = provider.search("Tom Skerritt", max_results=20)

    assert captured["url"] == retrieval.config.GNEWS_BASE_URL
    assert captured["params"]["q"] == "Tom Skerritt"
    assert captured["params"]["lang"] == "en"
    assert captured["params"]["max"] == 10
    assert captured["params"]["apikey"] == "gnews-test-key"
    assert captured["headers"] is None

    assert len(articles) == 3
    assert articles[0].source_domain == "mlive.com"
    assert articles[1].source_domain == "al.com"
    assert articles[2].source_domain == "nj.com"
    assert "Emmy-winning actor Tom Skerritt" in articles[0].content
    assert "[2332 chars]" not in articles[0].content
    assert not articles[0].content.endswith("...")


def test_missing_newsapi_key_raises_clear_error(tmp_path):
    provider = NewsAPIProvider(api_key="", budget=_budget(tmp_path))

    with pytest.raises(MissingAPIKeyError, match="NEWS_API_KEY is missing from .env"):
        provider.search("apple iphone", max_results=20)


def test_request_budget_refuses_above_limit_and_resets_next_day(tmp_path):
    day = ["2026-10-24"]
    budget = RequestBudget(
        tmp_path / "request_count.json",
        limit=1,
        today=lambda: day[0],
    )

    assert budget.usage() == (0, 1, "2026-10-24")
    assert budget.consume("newsapi") == 1

    with pytest.raises(DailyLimitError, match="Daily live request limit reached"):
        budget.consume("gnews")

    assert budget.usage() == (1, 1, "2026-10-24")
    day[0] = "2026-10-25"
    assert budget.usage() == (0, 1, "2026-10-25")
    assert budget.consume("newsapi") == 1


def test_retrieve_uses_cache_on_second_call(monkeypatch, tmp_path):
    calls = 0

    def fake_get(url, *, params, headers, timeout):
        nonlocal calls
        calls += 1
        return FakeResponse(_fixture("newsapi_tom_skerritt_1.json"))

    monkeypatch.setattr(retrieval.requests, "get", fake_get)
    cache = RetrievalCache(tmp_path / "cache")
    budget = _budget(tmp_path)
    provider = NewsAPIProvider(api_key="test-key", budget=budget)

    first = retrieve(
        ["Tom Skerritt"],
        providers=[provider],
        cache=cache,
        cache_only=False,
    )
    second = retrieve(
        ["Tom Skerritt"],
        providers=[provider],
        cache=cache,
        cache_only=False,
    )

    assert len(first) == 3
    assert [article.url for article in second] == [article.url for article in first]
    assert calls == 1
    assert budget.usage()[0] == 1


def test_retrieve_falls_back_when_primary_fails(monkeypatch, tmp_path):
    requested_urls: list[str] = []

    def fake_get(url, *, params, headers, timeout):
        requested_urls.append(url)
        if url == retrieval.config.NEWSAPI_BASE_URL:
            raise requests.ConnectionError("primary is down")
        return FakeResponse(_fixture("gnews_tom_skerritt.json"))

    monkeypatch.setattr(retrieval.requests, "get", fake_get)
    budget = _budget(tmp_path)
    providers = [
        NewsAPIProvider(api_key="newsapi-test-key", budget=budget),
        GNewsProvider(api_key="gnews-test-key", budget=budget),
    ]

    articles = retrieve(
        ["Tom Skerritt"],
        providers=providers,
        cache=RetrievalCache(tmp_path / "cache"),
        cache_only=False,
    )

    assert len(articles) == 2
    assert requested_urls == [
        retrieval.config.NEWSAPI_BASE_URL,
        retrieval.config.GNEWS_BASE_URL,
    ]
    assert budget.usage()[0] == 2


def test_retrieve_cache_only_never_calls_network(monkeypatch, tmp_path):
    cached = _article(
        "Cached launch story",
        "https://cache.example.com/story",
        "This article came from the local cache.",
    )

    def fake_get(*args, **kwargs):
        raise AssertionError("network must not be called in cache-only mode")

    monkeypatch.setattr(retrieval.requests, "get", fake_get)
    cache = RetrievalCache(tmp_path / "cache")
    cache.set("newsapi", "apple iphone", [cached])

    articles = retrieve(
        ["apple iphone"],
        providers=[NewsAPIProvider(api_key="", budget=_budget(tmp_path))],
        cache=cache,
        cache_only=True,
    )

    assert [article.url for article in articles] == [cached.url]


def test_retrieve_dedupes_urls_titles_and_empty_content(tmp_path):
    provider = StaticProvider(
        [
            _article(
                "A clear story headline",
                "https://www.example.com/story?utm_source=test",
                "First body.",
            ),
            _article(
                "A clear story headline",
                "https://example.com/story?utm_campaign=test",
                "Duplicate URL body.",
            ),
            _article(
                "A different story headline",
                "https://other.example.com/second",
                "Second body.",
            ),
            _article(
                "Near identical headline here",
                "https://third.example.com/third",
                "Third body.",
            ),
            _article(
                "Near identical headlines here",
                "https://fourth.example.com/fourth",
                "Fourth body.",
            ),
            _article(
                "Empty body story",
                "https://empty.example.com/story",
                "   ",
            ),
        ]
    )

    articles = retrieve(
        ["story"],
        providers=[provider],
        cache=RetrievalCache(tmp_path / "cache"),
        cache_only=False,
    )

    assert [article.title for article in articles] == [
        "A clear story headline",
        "A different story headline",
        "Near identical headline here",
    ]


def test_default_cache_ttl_is_fourteen_days():
    assert retrieval.config.CACHE_TTL_HOURS == 14 * 24
    assert RetrievalCache().ttl_seconds == 14 * 24 * 3600


def test_cache_ttl_expires_entries(tmp_path):
    cache = RetrievalCache(tmp_path / "cache", ttl_hours=-1)
    cached = _article("Cached story", "https://cache.example.com/story")
    cache.set("newsapi", "story", [cached])

    assert cache.get("newsapi", "story") is None
    assert cache.get("newsapi", "story", allow_stale=True) is not None


def test_retrieve_empty_queries_returns_empty(tmp_path):
    assert (
        retrieve(
            [],
            providers=[StaticProvider([_article("Unused", "https://x.example.com")])],
            cache=RetrievalCache(tmp_path / "cache"),
            cache_only=True,
        )
        == []
    )
