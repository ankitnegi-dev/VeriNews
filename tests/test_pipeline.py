"""Offline tests for pipeline wiring and per-stage degradation."""

from __future__ import annotations

import app.pipeline as pipeline_module
from app.pipeline import VerificationPipeline
from app.schemas import Article, Claim, NLIResult, RankedArticle


def _claim(text: str) -> Claim:
    return Claim(text=text, entities=[], queries=["query one", "query two"])


def _article(domain: str) -> Article:
    return Article(
        title=f"Article from {domain}",
        source_domain=domain,
        url=f"https://{domain}/story",
        published_at=None,
        content="The article contains relevant evidence.",
    )


def _ranked(domain: str) -> RankedArticle:
    return RankedArticle(
        title=f"Article from {domain}",
        source_domain=domain,
        url=f"https://{domain}/story",
        published_at=None,
        snippet="The article supports the claim.",
        similarity=0.9,
        recency_factor=1.0,
        score=0.9,
    )


def _nli(stance: str = "SUPPORTS") -> NLIResult:
    if stance == "SUPPORTS":
        return NLIResult(0.9, 0.08, 0.02, stance)
    if stance == "REFUTES":
        return NLIResult(0.02, 0.08, 0.9, stance)
    return NLIResult(0.02, 0.96, 0.02, stance)


class GoodStages:
    def __init__(self) -> None:
        self.retrieved: tuple[list[str], bool] | None = None

    def analyze(self, claim: str) -> Claim:
        return _claim(claim)

    def retrieve(self, queries, *, cache_only=False):
        self.retrieved = (list(queries), cache_only)
        return [_article("reuters.com"), _article("bbc.com")]

    def rank(self, claim, articles):
        return [_ranked("reuters.com"), _ranked("bbc.com")]

    def classify(self, claim, snippets):
        return [_nli("SUPPORTS"), _nli("SUPPORTS")]


def test_successful_pipeline_returns_a_verdict():
    stages = GoodStages()
    pipeline = VerificationPipeline(
        cache_only=True,
        analyzer=stages.analyze,
        retriever=stages.retrieve,
        ranker=stages.rank,
        classifier=stages.classify,
    )

    verdict = pipeline.verify("Test claim.")

    assert verdict.verdict == "SUPPORTED"
    assert verdict.queries == ["query one", "query two"]
    assert len(verdict.evidence) == 2
    assert verdict.latency_ms >= 0
    assert stages.retrieved == (["query one", "query two"], True)


def test_empty_claim_degrades_without_calling_stages():
    pipeline = VerificationPipeline(
        analyzer=lambda claim: (_ for _ in ()).throw(AssertionError("must not run"))
    )

    verdict = pipeline.verify("   ")

    assert verdict.verdict == "UNVERIFIED"
    assert "Claim is empty" in verdict.explanation


def test_claim_analysis_failure_degrades():
    pipeline = VerificationPipeline(
        analyzer=lambda claim: (_ for _ in ()).throw(RuntimeError("bad analysis"))
    )

    verdict = pipeline.verify("Test claim.")

    assert verdict.verdict == "UNVERIFIED"
    assert "Claim analysis failed" in verdict.explanation
    assert "bad analysis" in verdict.explanation
    assert verdict.latency_ms >= 0


def test_retrieval_failure_degrades_and_keeps_queries():
    def analyze(claim: str) -> Claim:
        return _claim(claim)

    def retrieve(queries, *, cache_only=False):
        raise RuntimeError("retrieval down")

    pipeline = VerificationPipeline(
        cache_only=True,
        analyzer=analyze,
        retriever=retrieve,
    )

    verdict = pipeline.verify("Test claim.")

    assert verdict.verdict == "UNVERIFIED"
    assert "Retrieval failed" in verdict.explanation
    assert verdict.queries == ["query one", "query two"]
    assert verdict.evidence == []


def test_ranking_failure_degrades():
    def retrieve(queries, *, cache_only=False):
        return [_article("reuters.com")]

    def rank(claim, articles):
        raise RuntimeError("ranking down")

    pipeline = VerificationPipeline(
        analyzer=lambda claim: _claim(claim),
        retriever=retrieve,
        ranker=rank,
    )

    verdict = pipeline.verify("Test claim.")

    assert verdict.verdict == "UNVERIFIED"
    assert "Ranking failed" in verdict.explanation


def test_nli_failure_degrades():
    def retrieve(queries, *, cache_only=False):
        return [_article("reuters.com")]

    def rank(claim, articles):
        return [_ranked("reuters.com")]

    def classify(claim, snippets):
        raise RuntimeError("NLI down")

    pipeline = VerificationPipeline(
        analyzer=lambda claim: _claim(claim),
        retriever=retrieve,
        ranker=rank,
        classifier=classify,
    )

    verdict = pipeline.verify("Test claim.")

    assert verdict.verdict == "UNVERIFIED"
    assert "NLI verification failed" in verdict.explanation


def test_decision_failure_degrades(monkeypatch):
    stages = GoodStages()
    pipeline = VerificationPipeline(
        analyzer=stages.analyze,
        retriever=stages.retrieve,
        ranker=stages.rank,
        classifier=stages.classify,
    )

    def fail_decide(*args, **kwargs):
        raise RuntimeError("decision down")

    monkeypatch.setattr(pipeline_module, "decide", fail_decide)
    verdict = pipeline.verify("Test claim.")

    assert verdict.verdict == "UNVERIFIED"
    assert "Decision failed" in verdict.explanation


def test_error_messages_redact_api_keys():
    def retrieve(queries, *, cache_only=False):
        raise RuntimeError("https://gnews.io/api/v4/search?apikey=SECRET123&q=test")

    pipeline = VerificationPipeline(
        analyzer=lambda claim: _claim(claim),
        retriever=retrieve,
    )

    verdict = pipeline.verify("Test claim.")

    assert "SECRET123" not in verdict.explanation
    assert "apikey=[redacted]" in verdict.explanation


def test_warmup_calls_all_three_loaders(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(pipeline_module, "_get_nlp", lambda: calls.append("nlp"))
    monkeypatch.setattr(
        pipeline_module,
        "_get_embedder",
        lambda: calls.append("embedding"),
    )
    monkeypatch.setattr(pipeline_module, "_get_model", lambda: calls.append("nli"))

    VerificationPipeline().warmup()

    assert calls == ["nlp", "embedding", "nli"]
