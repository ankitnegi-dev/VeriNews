"""Read the local full-text cache produced by the passage experiment.

The v3 experiment never fetches article text. It uses the cache written by the
``feat/passage-nli`` branch under ``.cache/fulltext``; a missing or empty record
means the caller must fall back to the ranked snippet.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def cache_key(url: str) -> str:
    """Return the SHA-256 cache key for a URL."""
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def cache_path(url: str, directory: Path | str | None = None) -> Path:
    """Return the cache file for a URL."""
    base = Path(directory) if directory is not None else config.FULLTEXT_CACHE_DIR
    return base / f"{cache_key(url)}.json"


def read_cached_text(
    url: str,
    *,
    directory: Path | str | None = None,
) -> str | None:
    """Return cached full text, or None when no usable record exists."""
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
    return text if isinstance(text, str) and text.strip() else None
