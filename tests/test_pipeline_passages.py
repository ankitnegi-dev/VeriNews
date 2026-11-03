"""Offline tests for passage-mode wiring, snippet fallback, and the question guard."""

from __future__ import annotations

import numpy as np

from app.pipeline import VerificationPipeline
from app.schemas import Article, Claim, NLIResult, RankedArticle


def _claim(text: str) -> Claim:
    return Claim(text=text, entities=[], queries=["query one", "query two"])


def _article(domain: str, title: str) -> Article:
    return Article(
        title=title,
        source_domain=domain,
        url=f"https://{domain}/story",
        published_at=None,
        content="Article body.",
    )


def _ranked(domain: str, title: str, snippet: str) -> RankedArticle:
    return RankedArticle(
        title=title,
        source_domain=domain,
        url=f"https://{domain}/story",
        published_at=None,
        snippet=snippet,
        similarity=0.9,
        recency_factor=1.0,
        score=0.9,
    )


class ClaimLikeEmbedder:
    """The claim and article body point at [1, 0]; titles point at [0, 1]."""

    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            lowered = sentence.lower()
            if "council approves" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=float)


class RecordingClassifier:
    def __init__(self, stance: str = "SUPPORTS") -> None:
        self.calls: list[list[str]] = []
        self.stance = stance

    def __call__(self, claim, snippets):
        self.calls.append(list(snippets))
        if self.stance == "REFUTES":
            return [NLIResult(0.02, 0.08, 0.90, "REFUTES") for _ in snippets]
        return [NLIResult(0.90, 0.08, 0.02, "SUPPORTS") for _ in snippets]


def test_question_guard_returns_exact_message_without_running_stages():
    def fail(*args, **kwargs):
        raise AssertionError("stages must not run for a question")

    pipeline = VerificationPipeline(
        analyzer=fail,
        retriever=fail,
        ranker=fail,
        classifier=fail,
    )

    verdict = pipeline.verify("Is this true?")

    assert verdict.verdict == "UNVERIFIED"
    assert verdict.explanation == (
        "This looks like a question. Enter one statement of fact."
    )
    assert verdict.queries == []
    assert verdict.evidence == []
    assert verdict.latency_ms >= 0


def test_passage_mode_uses_ranked_windows_and_aggregates_per_article():
    classifier = RecordingClassifier()
    full_text = (
        "The council approved the river plan on Tuesday. "
        "The plan pays for flood defences. "
        "Work starts in March."
    )
    pipeline = VerificationPipeline(
        cache_only=True,
        analyzer=lambda claim: _claim(claim),
        retriever=lambda queries, *, cache_only=False: [
            _article("reuters.com", "Council approves river plan"),
            _article("bbc.com", "Council approves river plan"),
        ],
        ranker=lambda claim, articles: [
            _ranked("reuters.com", "Council approves river plan", "Snippet one."),
            _ranked("bbc.com", "Council approves river plan", "Snippet two."),
        ],
        classifier=classifier,
        premise_mode="passage",
        embedder=ClaimLikeEmbedder(),
        full_text_getter=lambda url: full_text,
    )

    verdict = pipeline.verify("The council approved the river plan.")

    assert verdict.verdict == "SUPPORTED"
    assert len(classifier.calls) == 1
    premises = classifier.calls[0]
    assert len(premises) == 4
    assert all("council approves river plan" not in item.lower() for item in premises)
    assert all("flood defences" in item or "Work starts" in item for item in premises)
    assert len(verdict.evidence) == 2


def test_passage_mode_falls_back_to_the_snippet_when_text_is_missing():
    classifier = RecordingClassifier()
    pipeline = VerificationPipeline(
        cache_only=True,
        analyzer=lambda claim: _claim(claim),
        retriever=lambda queries, *, cache_only=False: [
            _article("reuters.com", "Council approves river plan"),
            _article("bbc.com", "Council approves river plan"),
        ],
        ranker=lambda claim, articles: [
            _ranked("reuters.com", "Council approves river plan", "Snippet one."),
            _ranked("bbc.com", "Council approves river plan", "Snippet two."),
        ],
        classifier=classifier,
        premise_mode="passage",
        embedder=ClaimLikeEmbedder(),
        full_text_getter=lambda url: "",
    )

    verdict = pipeline.verify("The council approved the river plan.")

    assert verdict.verdict == "SUPPORTED"
    assert classifier.calls == [["Snippet one.", "Snippet two."]]
