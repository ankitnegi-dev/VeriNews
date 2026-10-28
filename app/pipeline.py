"""End-to-end verification pipeline with per-stage degradation.

Each stage is wrapped separately. A failure in claim analysis, retrieval,
ranking, NLI, evidence assembly, or decision returns an UNVERIFIED verdict with
an explanation instead of raising to the API layer. The API has a final safety
net as well, so an unexpected error still becomes a 200 response with
UNVERIFIED rather than a 500.

Set ``CACHE_ONLY=true`` in the environment to make retrieval read only from the
local cache. The same flag is exposed by the pipeline constructor for tests and
for the offline demo.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable, Sequence

import config
from app.claim_analysis import _get_nlp, analyze
from app.credibility import CredibilityTable, load_credibility
from app.decision import build_evidence, decide
from app.nli import _get_model, classify
from app.ranking import _get_embedder, rank_articles
from app.retrieval import retrieve
from app.schemas import Article, Claim, NLIResult, RankedArticle, Verdict

logger = logging.getLogger(__name__)

Analyzer = Callable[[str], Claim]
Retriever = Callable[..., list[Article]]
Ranker = Callable[..., list[RankedArticle]]
Classifier = Callable[..., list[NLIResult]]

_SECRET_PATTERN = re.compile(r"(?i)\b(apikey|api_key|key|token)=([^&\s]+)")


def _safe_error(exc: Exception) -> str:
    """Keep an error readable without leaking API keys or query credentials."""
    message = " ".join(str(exc).split())
    message = _SECRET_PATTERN.sub(r"\1=[redacted]", message)
    if len(message) > 240:
        message = message[:237].rstrip() + "..."
    name = type(exc).__name__
    return f"{name}: {message}" if message else name


def failure_verdict(
    claim: str,
    reason: str,
    *,
    queries: Sequence[str] | None = None,
    latency_ms: int = 0,
) -> Verdict:
    """Build a safe UNVERIFIED response for a stage failure."""
    cleaned_reason = reason.strip().rstrip(".")
    return Verdict(
        claim=" ".join(str(claim).split()),
        verdict="UNVERIFIED",
        confidence=0.0,
        support_score=0.0,
        refute_score=0.0,
        total_weight=0.0,
        queries=list(queries or []),
        evidence=[],
        explanation=f"UNVERIFIED: {cleaned_reason}.",
        latency_ms=max(0, int(latency_ms)),
    )


class VerificationPipeline:
    """Wire the claim, retrieval, ranking, NLI, and decision stages together."""

    def __init__(
        self,
        *,
        cache_only: bool | None = None,
        credibility_table: CredibilityTable | None = None,
        analyzer: Analyzer = analyze,
        retriever: Retriever = retrieve,
        ranker: Ranker = rank_articles,
        classifier: Classifier = classify,
    ) -> None:
        self.cache_only = config.CACHE_ONLY if cache_only is None else cache_only
        self._credibility_table = credibility_table
        self._analyze = analyzer
        self._retrieve = retriever
        self._rank = ranker
        self._classify = classifier

    @property
    def credibility_table(self) -> CredibilityTable:
        if self._credibility_table is None:
            self._credibility_table = load_credibility()
        return self._credibility_table

    def warmup(self) -> None:
        """Load the three local models once at application startup."""
        loaders = (
            ("spaCy", _get_nlp),
            ("embedding model", _get_embedder),
            ("NLI model", _get_model),
        )
        for name, loader in loaders:
            try:
                loader()
                logger.info("Warmed up %s", name)
            except Exception:
                logger.exception("Could not warm up %s; requests will degrade", name)

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, round((time.perf_counter() - started) * 1000))

    def verify(self, claim_text: str) -> Verdict:
        started = time.perf_counter()
        claim = " ".join(str(claim_text).split())
        if not claim:
            return failure_verdict(claim, "Claim is empty")

        try:
            analysis = self._analyze(claim)
        except Exception as exc:
            logger.warning("Claim analysis failed: %s", _safe_error(exc))
            return failure_verdict(
                claim,
                f"Claim analysis failed ({_safe_error(exc)})",
                latency_ms=self._elapsed_ms(started),
            )

        try:
            articles = self._retrieve(
                analysis.queries,
                cache_only=self.cache_only,
            )
        except Exception as exc:
            logger.warning("Retrieval failed: %s", _safe_error(exc))
            return failure_verdict(
                analysis.text,
                f"Retrieval failed ({_safe_error(exc)})",
                queries=analysis.queries,
                latency_ms=self._elapsed_ms(started),
            )

        try:
            ranked = self._rank(analysis.text, articles)
        except Exception as exc:
            logger.warning("Ranking failed: %s", _safe_error(exc))
            return failure_verdict(
                analysis.text,
                f"Ranking failed ({_safe_error(exc)})",
                queries=analysis.queries,
                latency_ms=self._elapsed_ms(started),
            )

        try:
            snippets = [item.snippet for item in ranked]
            results = self._classify(analysis.text, snippets) if snippets else []
        except Exception as exc:
            logger.warning("NLI failed: %s", _safe_error(exc))
            return failure_verdict(
                analysis.text,
                f"NLI verification failed ({_safe_error(exc)})",
                queries=analysis.queries,
                latency_ms=self._elapsed_ms(started),
            )

        try:
            evidence = build_evidence(
                ranked,
                results,
                credibility_table=self.credibility_table,
            )
        except Exception as exc:
            logger.warning("Evidence assembly failed: %s", _safe_error(exc))
            return failure_verdict(
                analysis.text,
                f"Evidence assembly failed ({_safe_error(exc)})",
                queries=analysis.queries,
                latency_ms=self._elapsed_ms(started),
            )

        try:
            verdict = decide(
                analysis.text,
                evidence,
                queries=analysis.queries,
                latency_ms=self._elapsed_ms(started),
            )
        except Exception as exc:
            logger.warning("Decision failed: %s", _safe_error(exc))
            return failure_verdict(
                analysis.text,
                f"Decision failed ({_safe_error(exc)})",
                queries=analysis.queries,
                latency_ms=self._elapsed_ms(started),
            )

        logger.info(
            "Verdict %s for %r with %d evidence item(s) in %d ms",
            verdict.verdict,
            claim,
            len(verdict.evidence),
            verdict.latency_ms,
        )
        return verdict


_DEFAULT_PIPELINE: VerificationPipeline | None = None


def get_pipeline() -> VerificationPipeline:
    """Return the process-wide pipeline used by the API."""
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = VerificationPipeline()
    return _DEFAULT_PIPELINE


def verify(claim_text: str) -> Verdict:
    """Verify one claim with the process-wide pipeline."""
    return get_pipeline().verify(claim_text)
