"""Orchestration for the v3 claim-aware experiment.

The module keeps the v1 pipeline intact. A caller opts into one or more
components by passing :class:`V3Options`; with all flags false the pipeline is
byte-for-byte the v1 path.

Component B ranks 1- and 2-sentence windows, optionally per coordinated clause.
Component C applies the two evidence guards to snippets or passages. The
generic decision engine is still used for ordinary claims; a coordinated claim
uses the explicit clause rule requested by the experiment, with the same
two-article safety floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import config
from app.claim_aware import (
    ClaimFeatures,
    GuardStats,
    aggregate_results,
    apply_guards,
    best_passage,
    build_passages,
    claim_features,
    clause_combine,
    passage_diagnostics,
    rank_passages,
)
from app.credibility import CredibilityTable, load_credibility
from app.fulltext import read_cached_text
from app.schemas import Evidence, NLIResult, RankedArticle, Verdict

Classifier = Callable[..., list[NLIResult]]


@dataclass(frozen=True)
class V3Options:
    """Per-component switches for the v3 experiment."""

    retrieval: bool = False
    passages: bool = False
    guards: bool = False
    guard_term_coverage: float = config.V3_GUARD_TERM_COVERAGE
    guard_contradiction_max: float = config.V3_GUARD_CONTRADICTION_MAX
    guard_support_prob: float = config.V3_GUARD_SUPPORT_PROB
    passage_top_k: int = config.V3_PASSAGE_TOP_K
    passage_coverage_weight: float = config.V3_PASSAGE_COVERAGE_WEIGHT


@dataclass
class ClauseResult:
    """The per-clause evidence counts and the resulting clause verdict."""

    coordinated: bool
    clauses: list[str]
    counts: list[dict[str, object]]
    verdict: str
    explanation: str

    def as_dict(self) -> dict[str, object]:
        return {
            "coordinated": self.coordinated,
            "clauses": self.clauses,
            "counts": self.counts,
            "verdict": self.verdict,
            "explanation": self.explanation,
        }


@dataclass
class V3EvidenceResult:
    """Evidence plus the diagnostics needed by the ablation report."""

    evidence: list[Evidence]
    clause_result: ClauseResult | None
    guard_stats: GuardStats
    diagnostics: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "guard_stats": self.guard_stats.as_dict(),
            "clause_result": (
                self.clause_result.as_dict() if self.clause_result else None
            ),
            "articles": self.diagnostics,
        }


def _result_list(
    classifier: Classifier,
    claim: str,
    passages: Sequence[str],
) -> list[NLIResult]:
    results = list(classifier(claim, list(passages)))
    if len(results) != len(passages):
        raise ValueError(
            "classifier returned a different number of results "
            f"({len(results)} != {len(passages)})"
        )
    return results


def _article_text(
    article: RankedArticle,
    fulltext_reader: Callable[[str], str | None],
) -> tuple[str, str]:
    text = fulltext_reader(article.url)
    if text and text.strip():
        return text, "fulltext"
    return article.snippet, "snippet"


def _guarded(
    options: V3Options,
    features: ClaimFeatures,
    passage: str,
    result: NLIResult,
    stats: GuardStats,
) -> NLIResult:
    if not options.guards:
        return result
    return apply_guards(
        features,
        passage,
        result,
        min_term_coverage=options.guard_term_coverage,
        contradiction_max=options.guard_contradiction_max,
        support_prob=options.guard_support_prob,
        stats=stats,
    )


def build_v3_evidence(
    claim: str,
    ranked_articles: Sequence[RankedArticle],
    *,
    options: V3Options,
    classifier: Classifier,
    credibility_table: CredibilityTable | None = None,
    fulltext_reader: Callable[[str], str | None] = read_cached_text,
    embedder: object | None = None,
) -> V3EvidenceResult:
    """Build article evidence with B and/or C enabled."""
    if not options.passages and not options.guards:
        raise ValueError("build_v3_evidence called without B or C enabled")

    table = credibility_table or load_credibility()
    features = claim_features(claim)
    coordinated = bool(options.passages and features.coordinated)
    clauses = list(features.clauses) if coordinated else [claim]
    clause_counts: list[dict[str, object]] = [
        {
            "clause": clause,
            "supports": 0,
            "refutes": 0,
            "neutrals": 0,
            "articles": [],
        }
        for clause in clauses
    ]
    evidence: list[Evidence] = []
    diagnostics: list[dict[str, object]] = []
    total_guard_stats = GuardStats()

    for article in ranked_articles:
        text, source = _article_text(article, fulltext_reader)
        article_clause_results: list[NLIResult] = []
        article_clause_diagnostics: list[dict[str, object]] = []

        if options.passages:
            for clause_index, clause in enumerate(clauses):
                clause_features = (
                    claim_features(clause) if coordinated else features
                )
                candidates = build_passages(
                    text,
                    title=article.title,
                    sizes=config.V3_PASSAGE_WINDOW_SIZES,
                )
                passages = rank_passages(
                    clause,
                    candidates,
                    features=clause_features,
                    embedder=embedder,
                    top_k=options.passage_top_k,
                    coverage_weight=options.passage_coverage_weight,
                )
                if not passages:
                    passages = [article.snippet]
                raw_results = _result_list(classifier, clause, passages)
                guarded_results = [
                    _guarded(
                        options,
                        clause_features,
                        passage,
                        result,
                        total_guard_stats,
                    )
                    for passage, result in zip(passages, raw_results)
                ]
                aggregate = aggregate_results(guarded_results)
                article_clause_results.append(aggregate)

                if coordinated:
                    bucket = clause_counts[clause_index]
                    key = {
                        "SUPPORTS": "supports",
                        "REFUTES": "refutes",
                        "NEUTRAL": "neutrals",
                    }[aggregate.stance]
                    bucket[key] = int(bucket[key]) + 1
                    articles = bucket["articles"]
                    assert isinstance(articles, list)
                    articles.append(article.source_domain)

                article_clause_diagnostics.append(
                    {
                        "clause": clause,
                        "source": source,
                        "passages": passage_diagnostics(passages, guarded_results),
                    }
                )

            if coordinated:
                # Evidence is for display only; the clause rule supplies the
                # verdict. A conflicting article is shown as neutral.
                aggregate = aggregate_results(article_clause_results)
            else:
                aggregate = article_clause_results[0]
        else:
            snippet = article.snippet
            raw = _result_list(classifier, claim, [snippet])[0]
            aggregate = _guarded(
                options,
                features,
                snippet,
                raw,
                total_guard_stats,
            )
            article_clause_diagnostics.append(
                {
                    "clause": claim,
                    "source": "snippet",
                    "passages": passage_diagnostics([snippet], [aggregate]),
                }
            )

        best_text, _ = best_passage(
            [
                passage
                for clause_diag in article_clause_diagnostics
                for passage in [
                    item["passage"]
                    for item in clause_diag["passages"]  # type: ignore[index]
                ]
            ],
            [
                NLIResult(
                    item["p_entail"],  # type: ignore[index]
                    item["p_neutral"],  # type: ignore[index]
                    item["p_contra"],  # type: ignore[index]
                    item["stance"],  # type: ignore[index]
                )
                for clause_diag in article_clause_diagnostics
                for item in clause_diag["passages"]  # type: ignore[index]
            ],
        )
        evidence.append(
            Evidence(
                title=article.title,
                source_domain=article.source_domain,
                url=article.url,
                published_at=article.published_at,
                snippet=best_text or article.snippet,
                similarity=article.similarity,
                credibility=table.score_for(article.source_domain),
                p_entail=aggregate.p_entail,
                p_neutral=aggregate.p_neutral,
                p_contra=aggregate.p_contra,
                stance=aggregate.stance,
            )
        )
        diagnostics.append(
            {
                "domain": article.source_domain,
                "url": article.url,
                "title": article.title,
                "source": source,
                "similarity": article.similarity,
                "clauses": article_clause_diagnostics,
                "aggregate": {
                    "stance": aggregate.stance,
                    "p_entail": aggregate.p_entail,
                    "p_neutral": aggregate.p_neutral,
                    "p_contra": aggregate.p_contra,
                },
            }
        )

    clause_result = None
    if coordinated:
        verdict = clause_combine(
            [
                {
                    "supports": int(count["supports"]),
                    "refutes": int(count["refutes"]),
                }
                for count in clause_counts
            ]
        )
        summary = "; ".join(
            f"clause {index + 1} supports {count['supports']} / "
            f"refutes {count['refutes']}"
            for index, count in enumerate(clause_counts)
        )
        clause_result = ClauseResult(
            coordinated=True,
            clauses=clauses,
            counts=clause_counts,
            verdict=verdict,
            explanation=(
                f"Coordinated claim ({len(clauses)} clauses): {summary}; "
                f"clause rule -> {verdict}"
            ),
        )

    return V3EvidenceResult(
        evidence=evidence,
        clause_result=clause_result,
        guard_stats=total_guard_stats,
        diagnostics=diagnostics,
    )


def clause_verdict(
    claim: str,
    clause_result: ClauseResult,
    evidence: Sequence[Evidence],
    *,
    queries: Sequence[str],
    latency_ms: int,
) -> Verdict:
    """Build a Verdict from the explicit coordinated-clause rule."""
    supports = sum(int(count["supports"]) for count in clause_result.counts)
    refutes = sum(int(count["refutes"]) for count in clause_result.counts)
    verdict = clause_result.verdict
    if verdict == "SUPPORTED":
        support_score = float(supports)
        refute_score = float(refutes)
    elif verdict == "REFUTED":
        support_score = float(supports)
        refute_score = float(refutes)
    else:
        support_score = float(supports)
        refute_score = float(refutes)
    return Verdict(
        claim=claim,
        verdict=verdict,
        confidence=0.0,
        support_score=support_score,
        refute_score=refute_score,
        total_weight=float(supports + refutes),
        queries=list(queries),
        evidence=list(evidence),
        explanation=f"{verdict}: {clause_result.explanation}.",
        latency_ms=int(latency_ms),
    )
