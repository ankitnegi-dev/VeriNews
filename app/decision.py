"""Turn ranked NLI evidence into a safety-gated verdict.

Decision rules, in order:

1. Fewer than ``MIN_EVIDENCE`` kept articles means UNVERIFIED.
2. Articles below ``REL_MIN`` are irrelevant to the decision, even if NLI gave
   them a confident stance.
3. NEUTRAL articles are no information and carry no decision weight.
4. At least ``MIN_EVIDENCE`` relevant articles must agree on the deciding side,
   so a single article can never decide a verdict.
5. The weighted support and refute totals must clear ``MIN_TOTAL_WEIGHT``,
   ``DECISION_RATIO`` and ``MARGIN``.
6. Otherwise the result is UNVERIFIED.

Only ``REL_MIN``, ``MIN_TOTAL_WEIGHT``, ``DECISION_RATIO`` and ``MARGIN`` are
tunable. They are tuned on split=dev only.
"""

from __future__ import annotations

import logging
from typing import Sequence

import config
from app.credibility import CredibilityTable, load_credibility, normalize_domain
from app.schemas import Evidence, NLIResult, RankedArticle, Verdict

logger = logging.getLogger(__name__)

SUPPORTS = "SUPPORTS"
REFUTES = "REFUTES"
NEUTRAL = "NEUTRAL"
VALID_STANCES = frozenset({SUPPORTS, REFUTES, NEUTRAL})


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _weight(item: Evidence) -> float:
    return _clamp(item.similarity) * _clamp(item.credibility)


def _stance(item: Evidence) -> str:
    stance = str(item.stance).strip().upper()
    if stance not in VALID_STANCES:
        logger.warning("Unknown stance %r; treating it as NEUTRAL", item.stance)
        return NEUTRAL
    return stance


def _top_domains(items: Sequence[Evidence], limit: int = 2) -> list[str]:
    ordered = sorted(items, key=_weight, reverse=True)
    domains: list[str] = []
    for item in ordered:
        domain = normalize_domain(item.source_domain) or "unknown"
        if domain in domains:
            continue
        domains.append(domain)
        if len(domains) >= limit:
            break
    return domains


def _domain_phrase(items: Sequence[Evidence]) -> str:
    domains = _top_domains(items)
    return ", ".join(domains) if domains else "none"


def build_evidence(
    ranked_articles: Sequence[RankedArticle],
    nli_results: Sequence[NLIResult],
    *,
    credibility_table: CredibilityTable | None = None,
) -> list[Evidence]:
    """Attach stance probabilities and a credibility score to each article."""
    ranked = list(ranked_articles)
    results = list(nli_results)
    if len(ranked) != len(results):
        raise ValueError(
            "ranked_articles and nli_results must have the same length "
            f"({len(ranked)} != {len(results)})"
        )

    table = credibility_table or load_credibility()
    return [
        Evidence(
            title=item.title,
            source_domain=item.source_domain,
            url=item.url,
            published_at=item.published_at,
            snippet=item.snippet,
            similarity=item.similarity,
            credibility=table.score_for(item.source_domain),
            p_entail=result.p_entail,
            p_neutral=result.p_neutral,
            p_contra=result.p_contra,
            stance=result.stance,
        )
        for item, result in zip(ranked, results)
    ]


def _make_verdict(
    *,
    claim: str,
    verdict: str,
    confidence: float,
    support_score: float,
    refute_score: float,
    total_weight: float,
    queries: Sequence[str],
    evidence: Sequence[Evidence],
    explanation: str,
    latency_ms: int,
) -> Verdict:
    return Verdict(
        claim=claim,
        verdict=verdict,
        confidence=_clamp(confidence),
        support_score=support_score,
        refute_score=refute_score,
        total_weight=total_weight,
        queries=list(queries),
        evidence=list(evidence),
        explanation=explanation,
        latency_ms=int(latency_ms),
    )


def _unverified(
    *,
    claim: str,
    reason: str,
    confidence: float,
    support_score: float,
    refute_score: float,
    total_weight: float,
    queries: Sequence[str],
    evidence: Sequence[Evidence],
    supports: Sequence[Evidence],
    refutes: Sequence[Evidence],
    neutral_count: int,
    latency_ms: int,
) -> Verdict:
    parts = [f"UNVERIFIED: {reason}"]
    if supports:
        parts.append(f"top supporting: {_domain_phrase(supports)}")
    if refutes:
        parts.append(f"top refuting: {_domain_phrase(refutes)}")
    if neutral_count:
        parts.append(
            f"{neutral_count} relevant neutral article(s) treated as no information"
        )
    return _make_verdict(
        claim=claim,
        verdict="UNVERIFIED",
        confidence=confidence,
        support_score=support_score,
        refute_score=refute_score,
        total_weight=total_weight,
        queries=queries,
        evidence=evidence,
        explanation="; ".join(parts) + ".",
        latency_ms=latency_ms,
    )


def decide(
    claim: str,
    evidence: Sequence[Evidence],
    *,
    queries: Sequence[str] | None = None,
    latency_ms: int = 0,
) -> Verdict:
    """Aggregate evidence into SUPPORTED, REFUTED, or UNVERIFIED."""
    cleaned_claim = " ".join(claim.split()).strip()
    if not cleaned_claim:
        raise ValueError("Claim is empty.")

    items = list(evidence)
    query_list = list(queries or [])

    if len(items) < config.MIN_EVIDENCE:
        return _unverified(
            claim=cleaned_claim,
            reason=(
                f"only {len(items)} kept article(s); at least "
                f"{config.MIN_EVIDENCE} are required"
            ),
            confidence=0.0,
            support_score=0.0,
            refute_score=0.0,
            total_weight=0.0,
            queries=query_list,
            evidence=items,
            supports=[],
            refutes=[],
            neutral_count=0,
            latency_ms=latency_ms,
        )

    relevant = [
        item for item in items if _clamp(item.similarity) >= config.REL_MIN
    ]
    if not relevant:
        return _unverified(
            claim=cleaned_claim,
            reason=(
                f"none of the {len(items)} kept article(s) reaches the relevance "
                f"gate (similarity >= {config.REL_MIN:.2f})"
            ),
            confidence=0.0,
            support_score=0.0,
            refute_score=0.0,
            total_weight=0.0,
            queries=query_list,
            evidence=items,
            supports=[],
            refutes=[],
            neutral_count=0,
            latency_ms=latency_ms,
        )

    supports = [item for item in relevant if _stance(item) == SUPPORTS]
    refutes = [item for item in relevant if _stance(item) == REFUTES]
    neutrals = [item for item in relevant if _stance(item) == NEUTRAL]

    support_score = sum(
        _weight(item) * _clamp(item.p_entail) for item in supports
    )
    refute_score = sum(
        _weight(item) * _clamp(item.p_contra) for item in refutes
    )
    total_weight = sum(_weight(item) for item in [*supports, *refutes])

    if not supports and not refutes:
        return _unverified(
            claim=cleaned_claim,
            reason=(
                "no relevant article supports or refutes the claim; neutral "
                "evidence is treated as no information"
            ),
            confidence=0.0,
            support_score=0.0,
            refute_score=0.0,
            total_weight=0.0,
            queries=query_list,
            evidence=items,
            supports=[],
            refutes=[],
            neutral_count=len(neutrals),
            latency_ms=latency_ms,
        )

    if len(supports) < config.MIN_EVIDENCE and len(refutes) < config.MIN_EVIDENCE:
        if supports and not refutes:
            reason = (
                "only one relevant article supports the claim; a single article "
                "cannot decide a verdict"
            )
        elif refutes and not supports:
            reason = (
                "only one relevant article refutes the claim; a single article "
                "cannot decide a verdict"
            )
        else:
            reason = (
                f"only {len(supports)} relevant article(s) support and "
                f"{len(refutes)} refute the claim; at least "
                f"{config.MIN_EVIDENCE} relevant articles are required on the "
                "deciding side"
            )
        return _unverified(
            claim=cleaned_claim,
            reason=reason,
            confidence=0.0,
            support_score=support_score,
            refute_score=refute_score,
            total_weight=total_weight,
            queries=query_list,
            evidence=items,
            supports=supports,
            refutes=refutes,
            neutral_count=len(neutrals),
            latency_ms=latency_ms,
        )

    if total_weight < config.MIN_TOTAL_WEIGHT:
        return _unverified(
            claim=cleaned_claim,
            reason=(
                f"weighted relevant evidence is too low ({total_weight:.3f} < "
                f"{config.MIN_TOTAL_WEIGHT:.3f})"
            ),
            confidence=0.0,
            support_score=support_score,
            refute_score=refute_score,
            total_weight=total_weight,
            queries=query_list,
            evidence=items,
            supports=supports,
            refutes=refutes,
            neutral_count=len(neutrals),
            latency_ms=latency_ms,
        )

    ratio = max(support_score, refute_score) / total_weight
    confidence = abs(support_score - refute_score) / total_weight

    if ratio < config.DECISION_RATIO:
        return _unverified(
            claim=cleaned_claim,
            reason=(
                f"evidence is too evenly split (support {support_score:.3f} vs "
                f"refute {refute_score:.3f}; ratio {ratio:.3f} < "
                f"{config.DECISION_RATIO:.3f})"
            ),
            confidence=confidence,
            support_score=support_score,
            refute_score=refute_score,
            total_weight=total_weight,
            queries=query_list,
            evidence=items,
            supports=supports,
            refutes=refutes,
            neutral_count=len(neutrals),
            latency_ms=latency_ms,
        )

    if support_score > refute_score * config.MARGIN:
        verdict = "SUPPORTED"
        side = supports
    elif refute_score > support_score * config.MARGIN:
        verdict = "REFUTED"
        side = refutes
    else:
        return _unverified(
            claim=cleaned_claim,
            reason=(
                f"no side clears the decision margin (support "
                f"{support_score:.3f} vs refute {refute_score:.3f}; "
                f"margin {config.MARGIN:.2f})"
            ),
            confidence=confidence,
            support_score=support_score,
            refute_score=refute_score,
            total_weight=total_weight,
            queries=query_list,
            evidence=items,
            supports=supports,
            refutes=refutes,
            neutral_count=len(neutrals),
            latency_ms=latency_ms,
        )

    side_name = "support" if verdict == "SUPPORTED" else "refute"
    parts = [
        f"{verdict}: {len(side)} relevant sources {side_name} the claim "
        f"(top: {_domain_phrase(side)})",
        f"support {support_score:.3f} vs refute {refute_score:.3f}",
    ]
    if neutrals:
        parts.append(
            f"{len(neutrals)} relevant neutral article(s) treated as no information"
        )

    return _make_verdict(
        claim=cleaned_claim,
        verdict=verdict,
        confidence=confidence,
        support_score=support_score,
        refute_score=refute_score,
        total_weight=total_weight,
        queries=query_list,
        evidence=items,
        explanation="; ".join(parts) + ".",
        latency_ms=latency_ms,
    )
