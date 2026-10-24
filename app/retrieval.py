"""Live news retrieval with disk caching and a daily request budget.

The pipeline calls :func:`retrieve` with the queries produced by claim analysis.
Each query is tried against NewsAPI first and GNews second. Results are
normalized into :class:`Article`, cached on disk, merged across queries and
deduplicated by URL and near-identical title.

No API key is ever written to the cache or to a log. Offline tests inject
providers and a temporary cache directory, so they need neither a key nor
network access.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

import config
from app.schemas import Article

logger = logging.getLogger(__name__)

TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
}
TRUNCATION_MARKER = re.compile(r"\s*\[\+?\d+\s+chars\]\s*$", re.IGNORECASE)
TRAILING_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|\u2026)\s*$")
HTML_TAG = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")


class RetrievalError(RuntimeError):
    """Base class for retrieval failures that should not crash the pipeline."""


class ProviderError(RetrievalError):
    """A single provider failed, so the next provider may be tried."""


class MissingAPIKeyError(ProviderError):
    """A live provider has no key in .env."""


class DailyLimitError(ProviderError):
    """The safety cap for live requests has been reached."""


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class RequestBudget:
    """A small persistent counter that caps live requests per UTC day.

    The cap is deliberately below the free NewsAPI allowance. A request is
    counted before it is sent, so failed requests still consume budget. The
    policy is shared by all providers to keep total usage predictable.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        limit: int | None = None,
        today: Callable[[], str] | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else config.REQUEST_COUNT_PATH
        self.limit = config.DAILY_REQUEST_LIMIT if limit is None else limit
        self._today = today or _utc_today

    def usage(self) -> tuple[int, int, str]:
        data = self._read()
        return data["count"], self.limit, data["date"]

    def consume(self, provider: str = "live") -> int:
        data = self._read()
        if data["count"] >= self.limit:
            message = (
                f"Daily live request limit reached ({data['count']}/{self.limit} "
                f"on {data['date']}); refusing {provider} call"
            )
            logger.warning(message)
            raise DailyLimitError(message)

        data["count"] += 1
        self._write(data)
        logger.info(
            "Live request %d/%d used on %s for %s",
            data["count"],
            self.limit,
            data["date"],
            provider,
        )
        return data["count"]

    def _read(self) -> dict[str, object]:
        today = self._today()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"date": today, "count": 0}
        except (OSError, ValueError, TypeError):
            logger.warning("Could not read request counter at %s; resetting it", self.path)
            return {"date": today, "count": 0}

        if (
            isinstance(payload, dict)
            and payload.get("date") == today
            and isinstance(payload.get("count"), int)
            and payload["count"] >= 0
        ):
            return {"date": today, "count": payload["count"]}
        return {"date": today, "count": 0}

    def _write(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=self.path.name,
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary_name, self.path)
            temporary_name = ""
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)


class RetrievalCache:
    """JSON cache keyed by provider and query, with a configurable TTL."""

    def __init__(
        self,
        directory: Path | str | None = None,
        ttl_hours: float | None = None,
    ) -> None:
        self.directory = (
            Path(directory) if directory is not None else config.RETRIEVAL_CACHE_DIR
        )
        hours = config.CACHE_TTL_HOURS if ttl_hours is None else ttl_hours
        self.ttl_seconds = float(hours) * 3600

    def _path(self, provider: str, query: str) -> Path:
        key = f"{provider.strip().lower()}|{query.strip().lower()}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{provider.strip().lower()}_{digest}.json"

    def get(
        self,
        provider: str,
        query: str,
        *,
        allow_stale: bool = False,
    ) -> list[Article] | None:
        path = self._path(provider, query)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError):
            logger.warning("Ignoring unreadable cache entry at %s", path)
            return None

        if not allow_stale:
            fetched_at = payload.get("fetched_at") if isinstance(payload, dict) else None
            try:
                fetched = datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - fetched).total_seconds()
            except (TypeError, ValueError):
                logger.warning("Ignoring cache entry with a bad timestamp at %s", path)
                return None
            if age > self.ttl_seconds:
                return None

        raw_articles = payload.get("articles") if isinstance(payload, dict) else None
        if not isinstance(raw_articles, list):
            return None

        try:
            return [Article(**item) for item in raw_articles]
        except (TypeError, KeyError):
            logger.warning("Ignoring cache entry with malformed articles at %s", path)
            return None

    def set(self, provider: str, query: str, articles: Sequence[Article]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(provider, query)
        payload = {
            "provider": provider,
            "query": query,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "articles": [asdict(article) for article in articles],
        }

        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.directory,
            prefix=path.name,
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary_name, path)
            temporary_name = ""
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)


def _clean_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    text = unescape(value)
    text = HTML_TAG.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def _strip_truncation_marker(value: object) -> str:
    text = _clean_text(value)
    text = TRUNCATION_MARKER.sub("", text).strip()
    text = TRAILING_ELLIPSIS.sub("", text).strip()
    return text


def _combine_text(description: object, content: object) -> str:
    """Combine description and content without repeating the same text."""
    description_text = _strip_truncation_marker(description)
    content_text = _strip_truncation_marker(content)

    if not content_text:
        return description_text
    if not description_text:
        return content_text
    if content_text.startswith(description_text):
        return content_text
    if description_text.startswith(content_text):
        return description_text
    return f"{description_text} {content_text}"


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _domain_from_url(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    host = host.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _source_domain(url: str) -> str:
    return _domain_from_url(url) or "unknown"


def _normalize_url(url: str) -> str:
    value = _string(url)
    if not value:
        return ""

    try:
        parsed = urlparse(value)
    except ValueError:
        return value.lower()

    host = _domain_from_url(value)
    if not host:
        return ""

    try:
        port = parsed.port
    except ValueError:
        port = None
    if port and not (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    ):
        host = f"{host}:{port}"

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")

    query_pairs: list[tuple[str, str]] = []
    for key, parameter_value in parse_qsl(parsed.query, keep_blank_values=True):
        lower_key = key.lower()
        if lower_key.startswith("utm_") or lower_key in TRACKING_QUERY_KEYS:
            continue
        query_pairs.append((key, parameter_value))

    return urlunparse(
        (
            parsed.scheme.lower() or "https",
            host,
            path,
            "",
            urlencode(sorted(query_pairs)),
            "",
        )
    )


def _normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).lower()
    text = re.sub(r"[^\w\s]|_", " ", text)
    return WHITESPACE.sub(" ", text).strip()


def _titles_near_identical(first: str, second: str) -> bool:
    if not first or not second:
        return False
    if first == second:
        return True
    if min(len(first), len(second)) < 15:
        return False
    return SequenceMatcher(None, first, second).ratio() >= 0.92


def _dedupe_articles(articles: Iterable[Article]) -> list[Article]:
    unique: list[Article] = []
    seen_urls: set[str] = set()
    seen_titles: list[str] = []

    for article in articles:
        if not article.content.strip():
            continue

        url_key = _normalize_url(article.url)
        if url_key and url_key in seen_urls:
            continue

        title_key = _normalize_title(article.title)
        if title_key and any(
            _titles_near_identical(title_key, existing) for existing in seen_titles
        ):
            continue

        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.append(title_key)
        unique.append(article)

    return unique


def _article_from_newsapi(item: object) -> Article | None:
    if not isinstance(item, dict):
        return None

    url = _string(item.get("url"))
    title = _clean_text(item.get("title"))
    if not url or not title or title.lower() == "[removed]":
        return None

    content = _combine_text(item.get("description"), item.get("content"))
    if not content:
        return None

    published_at = item.get("publishedAt")
    return Article(
        title=title,
        source_domain=_source_domain(url),
        url=url,
        published_at=published_at if isinstance(published_at, str) else None,
        content=content,
    )


def _article_from_gnews(item: object) -> Article | None:
    if not isinstance(item, dict):
        return None

    url = _string(item.get("url"))
    title = _clean_text(item.get("title"))
    if not url or not title:
        return None

    content = _combine_text(item.get("description"), item.get("content"))
    if not content:
        return None

    published_at = item.get("publishedAt")
    return Article(
        title=title,
        source_domain=_source_domain(url),
        url=url,
        published_at=published_at if isinstance(published_at, str) else None,
        content=content,
    )


class SearchProvider(abc.ABC):
    """One live news search backend."""

    name = "base"

    @abc.abstractmethod
    def search(self, query: str, max_results: int = 10) -> list[Article]:
        """Return normalized articles for one query."""
        raise NotImplementedError


def _get_json(
    url: str,
    *,
    params: dict[str, object],
    headers: dict[str, str] | None,
    timeout: float,
    provider_label: str,
) -> dict[str, object]:
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise ProviderError(
            f"{provider_label} request failed ({type(exc).__name__})"
        ) from exc

    if response.status_code == 429:
        raise ProviderError(f"{provider_label} rate limit reached (HTTP 429)")
    if response.status_code in (401, 403):
        raise ProviderError(
            f"{provider_label} authentication failed; check the API key in .env"
        )
    if response.status_code >= 400:
        raise ProviderError(
            f"{provider_label} returned HTTP {response.status_code}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError(f"{provider_label} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ProviderError(f"{provider_label} returned an unexpected payload")
    return payload


class NewsAPIProvider(SearchProvider):
    """Primary provider using NewsAPI's everything endpoint."""

    name = "newsapi"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        budget: RequestBudget | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = config.NEWS_API_KEY if api_key is None else api_key
        self.budget = budget or RequestBudget()
        self.base_url = base_url or config.NEWSAPI_BASE_URL
        self.timeout = (
            config.RETRIEVAL_TIMEOUT_SECONDS if timeout is None else timeout
        )

    def _require_key(self) -> None:
        if not self.api_key.strip():
            raise MissingAPIKeyError(
                "NEWS_API_KEY is missing from .env. Add it to .env before live retrieval."
            )

    def search(self, query: str, max_results: int = 10) -> list[Article]:
        cleaned_query = query.strip()
        if not cleaned_query or max_results <= 0:
            return []

        self._require_key()
        self.budget.consume(self.name)

        payload = _get_json(
            self.base_url,
            params={
                "q": cleaned_query,
                "language": config.NEWS_LANGUAGE,
                "sortBy": "relevancy",
                "pageSize": config.NEWSAPI_PAGE_SIZE,
                "page": 1,
            },
            headers={"X-Api-Key": self.api_key},
            timeout=self.timeout,
            provider_label="NewsAPI",
        )

        if payload.get("status") != "ok":
            raise ProviderError("NewsAPI returned a non-ok status")

        articles = payload.get("articles")
        if not isinstance(articles, list):
            return []

        normalized = [
            article
            for item in articles
            if (article := _article_from_newsapi(item)) is not None
        ]
        return normalized[:max_results]


class GNewsProvider(SearchProvider):
    """Fallback provider using the GNews search endpoint."""

    name = "gnews"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        budget: RequestBudget | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = config.GNEWS_API_KEY if api_key is None else api_key
        self.budget = budget or RequestBudget()
        self.base_url = base_url or config.GNEWS_BASE_URL
        self.timeout = (
            config.RETRIEVAL_TIMEOUT_SECONDS if timeout is None else timeout
        )

    def _require_key(self) -> None:
        if not self.api_key.strip():
            raise MissingAPIKeyError(
                "GNEWS_API_KEY is missing from .env. Add it to .env before live retrieval."
            )

    def search(self, query: str, max_results: int = 10) -> list[Article]:
        cleaned_query = query.strip()
        if not cleaned_query or max_results <= 0:
            return []

        self._require_key()
        self.budget.consume(self.name)

        payload = _get_json(
            self.base_url,
            params={
                "q": cleaned_query,
                "lang": config.NEWS_LANGUAGE,
                "max": min(max_results, config.GNEWS_MAX_RESULTS),
                "apikey": self.api_key,
            },
            headers=None,
            timeout=self.timeout,
            provider_label="GNews",
        )

        if payload.get("errors"):
            raise ProviderError("GNews returned an error response")

        articles = payload.get("articles")
        if not isinstance(articles, list):
            return []

        normalized = [
            article
            for item in articles
            if (article := _article_from_gnews(item)) is not None
        ]
        return normalized[:max_results]


def _default_providers() -> list[SearchProvider]:
    budget = RequestBudget()
    return [
        NewsAPIProvider(budget=budget),
        GNewsProvider(budget=budget),
    ]


def retrieve(
    queries: Iterable[str],
    *,
    providers: Sequence[SearchProvider] | None = None,
    cache: RetrievalCache | None = None,
    cache_only: bool | None = None,
    max_results: int | None = None,
) -> list[Article]:
    """Retrieve and deduplicate articles for a list of search queries.

    Providers are tried in order. A provider failure falls through to the next
    provider. In cache-only mode no network call is made and stale cache entries
    are accepted, which keeps an offline demo working.
    """

    query_list: list[str] = []
    seen_queries: set[str] = set()
    for query in queries:
        if not isinstance(query, str):
            continue
        cleaned = query.strip()
        key = cleaned.lower()
        if not cleaned or key in seen_queries:
            continue
        seen_queries.add(key)
        query_list.append(cleaned)

    if not query_list:
        return []

    active_providers = list(providers) if providers is not None else _default_providers()
    if not active_providers:
        return []

    active_cache = cache or RetrievalCache()
    use_cache_only = config.CACHE_ONLY if cache_only is None else cache_only
    effective_max_results = (
        config.MAX_RESULTS_PER_QUERY if max_results is None else max_results
    )

    collected: list[Article] = []
    for query in query_list:
        for provider in active_providers:
            cached = active_cache.get(
                provider.name,
                query,
                allow_stale=use_cache_only,
            )
            if cached is not None:
                logger.info(
                    "Cache hit for %s (%d articles, stale allowed: %s)",
                    provider.name,
                    len(cached),
                    use_cache_only,
                )
                articles = cached
            elif use_cache_only:
                logger.info(
                    "Cache-only mode: no cached result for %s query", provider.name
                )
                continue
            else:
                try:
                    articles = provider.search(query, max_results=effective_max_results)
                except ProviderError as exc:
                    logger.warning("%s", exc)
                    articles = []
                except Exception:
                    logger.exception(
                        "Unexpected failure in provider %s", provider.name
                    )
                    articles = []
                else:
                    active_cache.set(provider.name, query, articles)

            if articles:
                collected.extend(articles)
                break

    unique = _dedupe_articles(collected)
    logger.info(
        "Retrieved %d unique articles from %d raw results",
        len(unique),
        len(collected),
    )
    if not unique and not use_cache_only:
        logger.error(
            "No articles retrieved. Check NEWS_API_KEY and GNEWS_API_KEY in .env "
            "and the provider warnings above."
        )
    return unique
