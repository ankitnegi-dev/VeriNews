"""Full-article text extraction and caching for the v2 passage experiment.

The cache stores one JSON record per URL under ``.cache/fulltext/``.  Article
text is never committed: ``.cache/`` is gitignored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.robotparser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

import config
import trafilatura

logger = logging.getLogger(__name__)

USER_AGENT = "VeriNews/1.0 (+https://github.com/ankitnegi-dev/VeriNews; passage-NLI experiment)"
FULLTEXT_CACHE_DIR = config.FULLTEXT_CACHE_DIR

_LAST_REQUEST: dict[str, float] = {}
_ROBOTS_CACHE: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def cache_key(url: str) -> str:
    """Return the SHA-256 cache key for a URL."""
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def cache_path(url: str, directory: Path | str | None = None) -> Path:
    """Return the cache file for a URL."""
    base = Path(directory) if directory is not None else FULLTEXT_CACHE_DIR
    return base / f"{cache_key(url)}.json"


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _rate_limit(domain: str, delay: float = 1.0) -> None:
    last = _LAST_REQUEST.get(domain, 0.0)
    wait = delay - (time.monotonic() - last)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST[domain] = time.monotonic()


def _robots_allowed(domain: str, url: str, timeout: float) -> bool:
    """Fetch and cache robots.txt for a domain."""
    if domain not in _ROBOTS_CACHE:
        _rate_limit(domain)
        robots_url = f"https://{domain}/robots.txt"
        parser: urllib.robotparser.RobotFileParser | None = None
        try:
            response = requests.get(
                robots_url,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            if response.status_code == 200:
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(response.text.splitlines())
        except requests.RequestException:
            parser = None
        _ROBOTS_CACHE[domain] = parser

    parser = _ROBOTS_CACHE[domain]
    if parser is None:
        return True
    try:
        return bool(parser.can_fetch(USER_AGENT, url))
    except Exception:
        return True


def extract_text(html: str, url: str | None = None) -> str:
    """Extract article text from an HTML document using trafilatura."""
    if not isinstance(html, str) or not html.strip():
        return ""
    try:
        text = trafilatura.extract(
            html,
            url=url,
            output_format="txt",
            include_comments=False,
            include_tables=False,
            include_images=False,
            include_links=False,
            no_fallback=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("trafilatura extraction failed: %s", exc)
        text = None
    if not text:
        return ""
    return " ".join(text.split())


def read_cached_text(
    url: str,
    *,
    directory: Path | str | None = None,
) -> str | None:
    """Return cached full text, or None when no cache record exists."""
    path = cache_path(url, directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError):
        logger.warning("Could not read full-text cache %s", path)
        return None
    if not isinstance(payload, dict):
        return None
    text = payload.get("text")
    return text if isinstance(text, str) else None


def write_cached_text(
    url: str,
    text: str,
    *,
    directory: Path | str | None = None,
    **metadata: Any,
) -> Path:
    """Write a full-text cache record."""
    path = cache_path(url, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": url,
        "text": text,
        "text_length": len(text),
        **metadata,
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return path


def fetch_full_text(
    url: str,
    *,
    timeout: float | None = None,
    user_agent: str = USER_AGENT,
    respect_robots: bool = True,
    cache: bool = True,
    force: bool = False,
    directory: Path | str | None = None,
) -> str:
    """Fetch and cache full text for a URL. Returns an empty string on failure."""
    if not force:
        cached = read_cached_text(url, directory=directory)
        if cached is not None:
            return cached

    active_timeout = config.FULLTEXT_TIMEOUT_SECONDS if timeout is None else timeout
    domain = _domain(url)

    if respect_robots and not _robots_allowed(domain, url, active_timeout):
        if cache:
            write_cached_text(
                url,
                "",
                directory=directory,
                robots_allowed=False,
                error="robots_disallowed",
            )
        return ""

    _rate_limit(domain)
    try:
        response = requests.get(
            url,
            headers={"User-Agent": user_agent},
            timeout=active_timeout,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        logger.warning("Full-text fetch failed for %s: %s", url, exc)
        if cache:
            write_cached_text(
                url,
                "",
                directory=directory,
                status_code=None,
                error=f"{type(exc).__name__}: {str(exc)[:120]}",
            )
        return ""

    if response.status_code >= 400:
        if cache:
            write_cached_text(
                url,
                "",
                directory=directory,
                status_code=response.status_code,
                error=f"http_{response.status_code}",
            )
        return ""

    text = extract_text(response.text, url=response.url)
    if cache:
        write_cached_text(
            url,
            text,
            directory=directory,
            status_code=response.status_code,
            error=None if text else "extraction_empty",
        )
    return text
