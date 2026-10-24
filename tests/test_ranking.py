"""Offline tests for article ranking."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

import config
from app.ranking import rank_articles, score_articles
from app.schemas import Article

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


class KeywordEmbedder:
    """Tiny deterministic embedder for relevance and tie-break tests."""

    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            if "tom skerritt" in sentence.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=float)


class DictEmbedder:
    """Return exact vectors for exact input strings."""

    def __init__(self, mapping: dict[str, list[float]], default=(0.0, 1.0)) -> None:
        self.mapping = mapping
        self.default = default

    def encode(self, sentences, **kwargs):
        return np.asarray(
            [self.mapping.get(sentence, self.default) for sentence in sentences],
            dtype=float,
        )


def _article(
    title: str,
    content: str,
    url: str,
    published_at: str | None = "2026-10-23T12:00:00Z",
    domain: str = "example.com",
) -> Article:
    return Article(
        title=title,
        source_domain=domain,
        url=url,
        published_at=published_at,
        content=content,
    )


def test_relevant_article_outranks_and_unrelated_is_filtered():
    claim = "Tom Skerritt died."
    relevant = _article(
        "Tom Skerritt dies at 93",
        "Veteran actor Tom Skerritt has died.",
        "https://example.com/relevant",
    )
    unrelated = _article(
        "Earthquake strikes Tibet",
        "A strong earthquake struck Tibet.",
        "https://example.com/unrelated",
    )

    scored = score_articles(
        claim,
        [unrelated, relevant],
        embedder=KeywordEmbedder(),
        now=NOW,
    )
    assert [item.title for item in scored] == [
        "Tom Skerritt dies at 93",
        "Earthquake strikes Tibet",
    ]
    assert scored[0].similarity == pytest.approx(1.0)
    assert scored[1].similarity == pytest.approx(0.0)

    ranked = rank_articles(
        claim,
        [unrelated, relevant],
        embedder=KeywordEmbedder(),
        now=NOW,
    )
    assert [item.title for item in ranked] == ["Tom Skerritt dies at 93"]


def test_best_sentence_sets_snippet_and_article_similarity():
    claim = "Tom Skerritt died."
    article = _article(
        "Daily briefing",
        "The weather was mild. Tom Skerritt died at 93.",
        "https://example.com/best-sentence",
    )
    mapping = {
        claim: [1.0, 0.0],
        "Daily briefing. The weather was mild. Tom Skerritt died at 93.": [0.6, 0.8],
        "Daily briefing": [0.0, 1.0],
        "The weather was mild.": [0.0, 1.0],
        "Tom Skerritt died at 93.": [1.0, 0.0],
    }

    scored = score_articles(
        claim,
        [article],
        embedder=DictEmbedder(mapping),
        now=NOW,
    )

    assert scored[0].similarity == pytest.approx(1.0)
    assert scored[0].snippet == "Tom Skerritt died at 93."


def test_recency_factor_breaks_a_similarity_tie():
    claim = "Tom Skerritt died."
    recent = _article(
        "Tom Skerritt dies",
        "Tom Skerritt has died.",
        "https://example.com/recent",
        published_at="2026-10-23T12:00:00Z",
    )
    older = _article(
        "Tom Skerritt dies",
        "Tom Skerritt has died.",
        "https://example.com/older",
        published_at="2026-09-01T12:00:00Z",
    )

    scored = score_articles(
        claim,
        [older, recent],
        embedder=KeywordEmbedder(),
        now=NOW,
    )

    assert scored[0].url == "https://example.com/recent"
    assert scored[1].url == "https://example.com/older"
    assert scored[0].recency_factor == pytest.approx(1.0)
    assert scored[1].recency_factor == pytest.approx(0.6)
    assert scored[0].score > scored[1].score


def test_unknown_date_uses_configured_factor():
    article = _article(
        "Tom Skerritt dies",
        "Tom Skerritt has died.",
        "https://example.com/unknown-date",
        published_at=None,
    )

    scored = score_articles(
        "Tom Skerritt died.",
        [article],
        embedder=KeywordEmbedder(),
        now=NOW,
    )

    assert scored[0].recency_factor == pytest.approx(config.RECENCY_UNKNOWN_DATE_FACTOR)


def test_rank_articles_respects_top_k():
    articles = [
        _article(
            f"Tom Skerritt story {index}",
            "Tom Skerritt has died.",
            f"https://example.com/story-{index}",
        )
        for index in range(3)
    ]

    ranked = rank_articles(
        "Tom Skerritt died.",
        articles,
        embedder=KeywordEmbedder(),
        now=NOW,
        top_k=1,
    )

    assert len(ranked) == 1


def test_snippet_is_truncated_to_configured_limit():
    long_sentence = "Tom Skerritt " + ("very " * 200) + "died."
    article = _article(
        "Daily briefing",
        long_sentence,
        "https://example.com/long-snippet",
    )

    scored = score_articles(
        "Tom Skerritt died.",
        [article],
        embedder=KeywordEmbedder(),
        now=NOW,
    )

    assert len(scored[0].snippet) <= config.SNIPPET_MAX_CHARS


def test_empty_inputs_return_empty_list():
    assert rank_articles("Tom Skerritt died.", [], embedder=KeywordEmbedder()) == []
    assert (
        score_articles(
            "",
            [_article("Tom Skerritt dies", "Tom Skerritt has died.", "https://x.example.com")],
            embedder=KeywordEmbedder(),
        )
        == []
    )
