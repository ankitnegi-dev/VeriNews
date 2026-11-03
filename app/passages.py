"""Sentence and passage helpers for the v2 passage-level NLI experiment."""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

import config
from app.nli import choose_stance
from app.schemas import NLIResult

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
WHITESPACE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return " ".join(str(text).split()).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple punctuation boundaries."""
    cleaned = _clean(text)
    if not cleaned:
        return []
    return [part.strip() for part in SENTENCE_BOUNDARY.split(cleaned) if part.strip()]


def sliding_windows(sentences: Sequence[str], size: int = 2) -> list[str]:
    """Build sliding windows of ``size`` consecutive sentences."""
    cleaned = [_clean(sentence) for sentence in sentences if _clean(sentence)]
    if not cleaned:
        return []
    size = max(1, int(size))
    if len(cleaned) <= size:
        return [" ".join(cleaned)]
    return [
        " ".join(cleaned[index : index + size])
        for index in range(len(cleaned) - size + 1)
    ]


def build_passages(
    text: str,
    *,
    title: str | None = None,
    window_size: int | None = None,
) -> list[str]:
    """Build passage candidates, including the title and 2-sentence windows."""
    size = config.PREMISE_WINDOW_SIZE if window_size is None else window_size
    passages: list[str] = []
    title_text = _clean(title or "")
    if title_text:
        passages.append(title_text)

    for window in sliding_windows(split_sentences(text), size):
        cleaned = _clean(window)
        if cleaned and cleaned not in passages:
            passages.append(cleaned)

    if not passages and _clean(text):
        passages.append(_clean(text))
    return passages


def _normalize_rows(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def rank_passages(
    claim: str,
    passages: Sequence[str],
    *,
    embedder: Any = None,
    top_k: int | None = None,
) -> list[str]:
    """Rank passages by cosine similarity to the claim and return the top ones."""
    cleaned_claim = _clean(claim)
    cleaned_passages = [_clean(passage) for passage in passages if _clean(passage)]
    if not cleaned_claim or not cleaned_passages:
        return []

    limit = config.PREMISE_PASSAGE_COUNT if top_k is None else int(top_k)
    if limit <= 0:
        return []
    if len(cleaned_passages) <= limit:
        return cleaned_passages

    if embedder is None:
        from app.ranking import _get_embedder

        embedder = _get_embedder()

    texts = [cleaned_claim, *cleaned_passages]
    try:
        encoded = embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except TypeError:
        encoded = embedder.encode(texts)
    matrix = _normalize_rows(encoded)
    claim_vector = matrix[0]
    scores = matrix[1:] @ claim_vector

    order = sorted(range(len(cleaned_passages)), key=lambda i: float(scores[i]), reverse=True)
    return [cleaned_passages[i] for i in order[:limit]]


def top_passages(
    claim: str,
    text: str,
    *,
    title: str | None = None,
    embedder: Any = None,
    top_k: int | None = None,
    window_size: int | None = None,
) -> list[str]:
    """Build and rank passages for one article."""
    passages = build_passages(text, title=title, window_size=window_size)
    return rank_passages(claim, passages, embedder=embedder, top_k=top_k)


def premises_for_article(
    claim: str,
    text: str,
    *,
    title: str | None = None,
    snippet: str | None = None,
    embedder: Any = None,
    top_k: int | None = None,
    window_size: int | None = None,
) -> list[str]:
    """Return the NLI premises for one article, falling back to the snippet.

    Full text is preferred. When extraction failed or returned nothing, the
    ranked snippet is the premise, exactly as in snippet mode.
    """
    if not _clean(text):
        fallback = _clean(snippet or "")
        return [fallback] if fallback else []

    passages = top_passages(
        claim,
        text,
        title=title,
        embedder=embedder,
        top_k=top_k,
        window_size=window_size,
    )
    if passages:
        return passages
    fallback = _clean(snippet or "")
    return [fallback] if fallback else []


def aggregate_passage_results(results: Sequence[NLIResult]) -> NLIResult:
    """Aggregate passage-level NLI results into one article-level result."""
    cleaned = [result for result in results if result is not None]
    if not cleaned:
        return NLIResult(0.0, 1.0, 0.0, "NEUTRAL", cached=False)

    max_entail = max(result.p_entail for result in cleaned)
    max_contra = max(result.p_contra for result in cleaned)
    max_neutral = max(result.p_neutral for result in cleaned)

    if max_entail > 0.5 and max_contra > 0.5:
        stance = "NEUTRAL"
    else:
        stance = choose_stance(max_entail, max_neutral, max_contra)

    return NLIResult(
        p_entail=max_entail,
        p_neutral=max_neutral,
        p_contra=max_contra,
        stance=stance,
        cached=all(result.cached for result in cleaned),
    )
