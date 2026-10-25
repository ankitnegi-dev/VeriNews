"""Rank retrieved articles by claim relevance and recency.

The API text is short and often truncated, so article-level scoring uses the
whole title plus content. A best-unit score is also computed from the title and
content sentences, and the larger of the two similarities is used. The final
score multiplies that similarity by a recency factor.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Protocol, Sequence

import numpy as np

import config
from app.schemas import Article, RankedArticle

logger = logging.getLogger(__name__)

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_EMBEDDER = None


class Embedder(Protocol):
    """Minimal interface shared by SentenceTransformer and test doubles."""

    def encode(self, sentences: Sequence[str], **kwargs: object) -> object:
        ...


def _get_embedder() -> Embedder:
    """Load the MiniLM model once, on first use."""
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        logger.info("Loading embedding model %s", config.EMBEDDING_MODEL_NAME)
        try:
            _EMBEDDER = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load embedding model {config.EMBEDDING_MODEL_NAME!r}. "
                "Check the model cache or download it with sentence-transformers."
            ) from exc
    return _EMBEDDER


def _article_text(article: Article) -> str:
    title = article.title.strip()
    content = article.content.strip()
    if title and content:
        return f"{title}. {content}"
    return title or content


def _split_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return [part.strip() for part in SENTENCE_BOUNDARY.split(cleaned) if part.strip()]


def _candidate_units(article: Article) -> list[str]:
    units: list[str] = []
    for value in [article.title, *_split_sentences(article.content)]:
        cleaned = " ".join(value.split()).strip()
        if cleaned and cleaned not in units:
            units.append(cleaned)
    return units


def _normalize_rows(values: object) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _encode(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=float)
    encoded = embedder.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return _normalize_rows(encoded)


def _recency_factor(published_at: str | None, now: datetime) -> float:
    if not published_at:
        return config.RECENCY_UNKNOWN_DATE_FACTOR

    try:
        published = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return config.RECENCY_UNKNOWN_DATE_FACTOR
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)

    age_days = max(0.0, (now - published).total_seconds() / 86400.0)
    if age_days <= config.RECENCY_FULL_CREDIT_DAYS:
        return 1.0
    if age_days >= config.RECENCY_ZERO_CREDIT_DAYS:
        return config.RECENCY_MIN_FACTOR

    span = config.RECENCY_ZERO_CREDIT_DAYS - config.RECENCY_FULL_CREDIT_DAYS
    fraction = (age_days - config.RECENCY_FULL_CREDIT_DAYS) / span
    return 1.0 - (1.0 - config.RECENCY_MIN_FACTOR) * fraction


def _snippet(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= config.SNIPPET_MAX_CHARS:
        return cleaned
    return cleaned[: config.SNIPPET_MAX_CHARS - 3].rstrip() + "..."


def score_articles(
    claim: str,
    articles: Sequence[Article],
    *,
    embedder: Embedder | None = None,
    now: datetime | None = None,
) -> list[RankedArticle]:
    """Score every article without applying the similarity threshold."""
    cleaned_claim = " ".join(claim.split()).strip()
    article_list = list(articles)
    if not cleaned_claim or not article_list:
        return []

    active_embedder = embedder or _get_embedder()
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)

    article_texts = [_article_text(article) for article in article_list]
    embeddings = _encode(active_embedder, [cleaned_claim, *article_texts])
    claim_vector = embeddings[0]
    article_vectors = embeddings[1:]
    full_similarities = article_vectors @ claim_vector

    units_by_article = [_candidate_units(article) for article in article_list]
    unit_records: list[tuple[int, str]] = []
    for index, units in enumerate(units_by_article):
        unit_records.extend((index, unit) for unit in units)

    best_unit_text = [""] * len(article_list)
    best_unit_similarity = np.zeros(len(article_list), dtype=float)
    if unit_records:
        unit_embeddings = _encode(active_embedder, [unit for _, unit in unit_records])
        unit_similarities = unit_embeddings @ claim_vector
        for (index, unit), similarity in zip(unit_records, unit_similarities):
            if similarity > best_unit_similarity[index]:
                best_unit_similarity[index] = similarity
                best_unit_text[index] = unit

    ranked: list[RankedArticle] = []
    for index, article in enumerate(article_list):
        full_similarity = float(full_similarities[index])
        unit_similarity = float(best_unit_similarity[index])
        similarity = max(0.0, min(1.0, max(full_similarity, unit_similarity)))
        recency = _recency_factor(article.published_at, current)
        snippet_text = best_unit_text[index] or article_texts[index]
        ranked.append(
            RankedArticle(
                title=article.title,
                source_domain=article.source_domain,
                url=article.url,
                published_at=article.published_at,
                snippet=_snippet(snippet_text),
                similarity=similarity,
                recency_factor=recency,
                score=similarity * recency,
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def _domain_key(item: RankedArticle) -> str:
    domain = item.source_domain.strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain or item.url.strip().lower()


def _diversify(items: Sequence[RankedArticle], limit: int) -> list[RankedArticle]:
    """Keep the best-scoring article from each domain, up to the limit."""
    selected: list[RankedArticle] = []
    seen_domains: set[str] = set()
    for item in items:
        domain = _domain_key(item)
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def rank_articles(
    claim: str,
    articles: Sequence[Article],
    *,
    top_k: int | None = None,
    min_similarity: float | None = None,
    embedder: Embedder | None = None,
    now: datetime | None = None,
) -> list[RankedArticle]:
    """Return the top articles whose raw similarity meets the threshold.

    The final list has at most one article per source domain. Syndicated copies
    on different domains can still both appear; domain diversity is not a
    duplicate detector.
    """
    limit = config.TOP_K if top_k is None else top_k
    threshold = config.MIN_SIMILARITY if min_similarity is None else min_similarity
    if limit <= 0:
        return []

    scored = score_articles(claim, articles, embedder=embedder, now=now)
    filtered = [item for item in scored if item.similarity >= threshold]
    return _diversify(filtered, limit)
