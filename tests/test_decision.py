"""Hand-built decision-engine tests for the required safety patterns."""

from __future__ import annotations

import pytest

from app.decision import build_evidence, decide
from app.schemas import Evidence, NLIResult, RankedArticle


def _evidence(
    domain: str,
    stance: str,
    *,
    similarity: float = 0.9,
    credibility: float = 1.0,
    p_entail: float | None = None,
    p_neutral: float | None = None,
    p_contra: float | None = None,
) -> Evidence:
    if stance == "SUPPORTS":
        default_entail, default_neutral, default_contra = 0.90, 0.08, 0.02
    elif stance == "REFUTES":
        default_entail, default_neutral, default_contra = 0.02, 0.08, 0.90
    else:
        default_entail, default_neutral, default_contra = 0.02, 0.96, 0.02

    return Evidence(
        title=f"Article from {domain}",
        source_domain=domain,
        url=f"https://{domain}/article",
        published_at=None,
        snippet=f"Snippet from {domain}.",
        similarity=similarity,
        credibility=credibility,
        p_entail=default_entail if p_entail is None else p_entail,
        p_neutral=default_neutral if p_neutral is None else p_neutral,
        p_contra=default_contra if p_contra is None else p_contra,
        stance=stance,
    )


def test_clear_support_is_supported():
    evidence = [
        _evidence("reuters.com", "SUPPORTS"),
        _evidence("bbc.com", "SUPPORTS"),
        _evidence("npr.org", "SUPPORTS"),
    ]

    verdict = decide("The test claim is true.", evidence, queries=["test claim"])

    assert verdict.verdict == "SUPPORTED"
    assert verdict.support_score > verdict.refute_score
    assert verdict.confidence > 0.8
    assert "SUPPORTED" in verdict.explanation
    assert "3 relevant sources support" in verdict.explanation
    assert "reuters.com, bbc.com" in verdict.explanation
    assert verdict.queries == ["test claim"]
    assert len(verdict.evidence) == 3


def test_clear_refutation_is_refuted():
    evidence = [
        _evidence("reuters.com", "REFUTES", similarity=0.8, credibility=0.7),
        _evidence("apnews.com", "REFUTES", similarity=0.8, credibility=0.7),
    ]

    verdict = decide("The test claim is false.", evidence)

    assert verdict.verdict == "REFUTED"
    assert verdict.refute_score > verdict.support_score
    assert verdict.confidence > 0.8
    assert "REFUTED" in verdict.explanation
    assert "2 relevant sources refute" in verdict.explanation


def test_conflict_is_unverified():
    evidence = [
        _evidence("reuters.com", "SUPPORTS", similarity=0.8, credibility=0.7),
        _evidence("bbc.com", "SUPPORTS", similarity=0.8, credibility=0.7),
        _evidence("apnews.com", "REFUTES", similarity=0.8, credibility=0.7),
        _evidence("npr.org", "REFUTES", similarity=0.8, credibility=0.7),
    ]

    verdict = decide("The test claim is disputed.", evidence)

    assert verdict.verdict == "UNVERIFIED"
    assert verdict.support_score == pytest.approx(verdict.refute_score)
    assert verdict.confidence == pytest.approx(0.0)
    assert "decision margin" in verdict.explanation


def test_too_little_evidence_is_unverified():
    verdict = decide("The test claim has one source.", [_evidence("reuters.com", "SUPPORTS")])

    assert verdict.verdict == "UNVERIFIED"
    assert verdict.support_score == 0.0
    assert verdict.total_weight == 0.0
    assert "only 1 kept article" in verdict.explanation


def test_no_evidence_is_unverified():
    verdict = decide("The test claim has no sources.", [])

    assert verdict.verdict == "UNVERIFIED"
    assert "only 0 kept article" in verdict.explanation


def test_irrelevant_evidence_is_unverified():
    evidence = [
        _evidence("reuters.com", "SUPPORTS", similarity=0.45),
        _evidence("bbc.com", "SUPPORTS", similarity=0.45),
    ]

    verdict = decide("The test claim has unrelated coverage.", evidence)

    assert verdict.verdict == "UNVERIFIED"
    assert verdict.support_score == 0.0
    assert "relevance gate" in verdict.explanation


def test_bollywood_pattern_is_unverified():
    evidence = [
        _evidence(
            "economictimes.indiatimes.com",
            "REFUTES",
            similarity=0.403,
            credibility=0.7,
            p_entail=0.0,
            p_neutral=0.0,
            p_contra=1.0,
        )
    ]

    verdict = decide(
        "An anonymous source says a famous Bollywood actor will announce retirement.",
        evidence,
    )

    assert verdict.verdict == "UNVERIFIED"
    assert verdict.refute_score == 0.0
    assert "only 1 kept article" in verdict.explanation


def test_one_support_with_neutrals_is_unverified():
    evidence = [
        _evidence("reuters.com", "SUPPORTS"),
        _evidence("bbc.com", "NEUTRAL"),
        _evidence("npr.org", "NEUTRAL"),
    ]

    verdict = decide("The test claim has one supporting article.", evidence)

    assert verdict.verdict == "UNVERIFIED"
    assert verdict.support_score > 0.0
    assert verdict.confidence == 0.0
    assert "single article cannot decide" in verdict.explanation


def test_one_refute_with_neutrals_is_unverified():
    evidence = [
        _evidence("cbc.ca", "REFUTES", similarity=0.779, credibility=0.4),
        _evidence("dw.com", "NEUTRAL", similarity=0.802, credibility=1.0),
        _evidence("business-standard.com", "NEUTRAL", similarity=0.770, credibility=0.7),
    ]

    verdict = decide("The test claim has one refuting article.", evidence)

    assert verdict.verdict == "UNVERIFIED"
    assert verdict.refute_score > 0.0
    assert verdict.confidence == 0.0
    assert "only one relevant article refutes" in verdict.explanation


def test_low_signal_conflict_fails_the_ratio_gate():
    evidence = [
        _evidence(
            "reuters.com",
            "SUPPORTS",
            similarity=0.8,
            credibility=0.7,
            p_entail=0.55,
            p_neutral=0.35,
            p_contra=0.10,
        ),
        _evidence(
            "bbc.com",
            "SUPPORTS",
            similarity=0.8,
            credibility=0.7,
            p_entail=0.55,
            p_neutral=0.35,
            p_contra=0.10,
        ),
        _evidence(
            "apnews.com",
            "REFUTES",
            similarity=0.8,
            credibility=0.7,
            p_entail=0.10,
            p_neutral=0.35,
            p_contra=0.55,
        ),
        _evidence(
            "npr.org",
            "REFUTES",
            similarity=0.8,
            credibility=0.7,
            p_entail=0.10,
            p_neutral=0.35,
            p_contra=0.55,
        ),
    ]

    verdict = decide("The test claim has weak conflicting signals.", evidence)

    assert verdict.verdict == "UNVERIFIED"
    assert "too evenly split" in verdict.explanation


def test_one_relevant_support_with_a_below_gate_support_is_unverified():
    evidence = [
        _evidence("reuters.com", "SUPPORTS", similarity=0.9),
        _evidence("bbc.com", "SUPPORTS", similarity=0.45),
    ]

    verdict = decide("The test claim has one relevant supporter.", evidence)

    assert verdict.verdict == "UNVERIFIED"
    assert verdict.support_score == pytest.approx(0.81)
    assert "single article cannot decide" in verdict.explanation


def test_neutral_evidence_is_no_information():
    evidence = [
        _evidence("reuters.com", "NEUTRAL"),
        _evidence("bbc.com", "NEUTRAL"),
        _evidence("npr.org", "NEUTRAL"),
    ]

    verdict = decide("The test claim is not addressed by the coverage.", evidence)

    assert verdict.verdict == "UNVERIFIED"
    assert verdict.support_score == 0.0
    assert verdict.refute_score == 0.0
    assert verdict.total_weight == 0.0
    assert "neutral evidence is treated as no information" in verdict.explanation


def test_neutral_evidence_does_not_block_a_supported_verdict():
    evidence = [
        _evidence("reuters.com", "SUPPORTS"),
        _evidence("bbc.com", "SUPPORTS"),
        _evidence("npr.org", "NEUTRAL"),
    ]

    verdict = decide("The test claim has two supporting articles.", evidence)

    assert verdict.verdict == "SUPPORTED"
    assert verdict.total_weight == pytest.approx(1.8)


def test_low_weight_evidence_is_unverified():
    evidence = [
        _evidence("unknown-one.example", "SUPPORTS", similarity=0.6, credibility=0.4),
        _evidence("unknown-two.example", "SUPPORTS", similarity=0.6, credibility=0.4),
    ]

    verdict = decide("The test claim has weak sourcing.", evidence)

    assert verdict.verdict == "UNVERIFIED"
    assert verdict.total_weight < 0.5
    assert "weighted relevant evidence is too low" in verdict.explanation


def test_build_evidence_attaches_credibility_and_nli_fields():
    ranked = RankedArticle(
        title="Reuters report",
        source_domain="www.reuters.com",
        url="https://www.reuters.com/world/story",
        published_at=None,
        snippet="The report confirms the claim.",
        similarity=0.82,
        recency_factor=1.0,
        score=0.82,
    )
    result = NLIResult(0.91, 0.06, 0.03, "SUPPORTS")

    evidence = build_evidence([ranked], [result])

    assert len(evidence) == 1
    assert evidence[0].credibility == 1.0
    assert evidence[0].p_entail == 0.91
    assert evidence[0].stance == "SUPPORTS"
    assert evidence[0].source_domain == "www.reuters.com"


def test_build_evidence_rejects_mismatched_lengths():
    ranked = RankedArticle(
        title="Title",
        source_domain="example.com",
        url="https://example.com/story",
        published_at=None,
        snippet="Snippet.",
        similarity=0.8,
        recency_factor=1.0,
        score=0.8,
    )

    with pytest.raises(ValueError, match="same length"):
        build_evidence([ranked], [])
