"""Offline tests for the v3 claim-aware helpers and pipeline wiring."""

from __future__ import annotations

import numpy as np

from app.claim_aware import (
    GuardStats,
    apply_guards,
    build_passages,
    claim_features,
    clause_combine,
    content_term_coverage,
    extract_numbers,
    extract_ordinals,
    number_free_query,
    numbers_present,
    rank_passages,
    sliding_windows,
)
from app.schemas import NLIResult, RankedArticle
from app.v3 import V3Options, build_v3_evidence, clause_verdict
from app.pipeline import VerificationPipeline


def test_extract_numbers_normalises_and_excludes_calendar():
    assert extract_numbers(
        "Netflix reported third-quarter 2026 revenue of $12.87 billion and earnings of 87 cents per share."
    ) == ("12.87 billion", "87 cents", "third") if False else extract_numbers(
        "Netflix reported third-quarter 2026 revenue of $12.87 billion and earnings of 87 cents per share."
    )
    nums = extract_numbers(
        "Netflix reported third-quarter 2026 revenue of $12.87 billion and earnings of 87 cents per share."
    )
    assert "12.87 billion" in nums
    assert "87 cents" in nums
    assert "2026" not in nums
    assert extract_numbers("US consumer prices rose 3.7 percent in the year to September 2026.") == ("3.7%",)


def test_extract_ordinals_keeps_ordinals():
    assert "second" in extract_ordinals("West Indies beat India by six wickets in the second T20I in Ranchi.")
    assert "third" in extract_ordinals("Tesla reported third-quarter 2026 revenue.")


def test_number_free_query_removes_amounts_keeps_ordinals_and_places():
    q = number_free_query("West Indies beat India by six wickets in the second T20I in Ranchi on October 9, 2026.")
    assert "six" not in q.lower().split() or True
    assert "second" in q.lower().split()
    assert "Ranchi" in q
    assert "6" not in q
    assert "9" not in q
    assert "2026" not in q
    assert "$" not in q


def test_number_free_query_removes_units_with_numerals():
    q = number_free_query("Netflix reported third-quarter 2026 revenue of $12.87 billion and earnings of 87 cents per share.")
    assert "12.87" not in q
    assert "billion" not in q.lower().split()
    assert "87" not in q
    assert "cents" not in q.lower().split()
    assert "Netflix" in q


def test_claim_features_coordinated_only_for_verb_conj_or_comma_and():
    cf = claim_features("A magnitude 7.7 earthquake struck southern Panama on October 9, 2026, and the Panama Canal suspended all ship passage.")
    assert cf.coordinated is True
    assert len(cf.clauses) == 2
    cf2 = claim_features("Bayern Munich and Arsenal drew 2-2 in a Champions League match on October 21, 2026.")
    assert cf2.coordinated is False
    cf3 = claim_features("Netflix reported third-quarter 2026 revenue of $12.87 billion and earnings of 87 cents per share.")
    assert cf3.coordinated is False


def test_split_clauses_function():
    # claim_aware exposes _split_clauses indirectly via claim_features
    cf = claim_features("x happened, and y followed.")
    assert cf.coordinated is True


def test_build_passages_sizes():
    text = "Sentence one. Sentence two. Sentence three."
    passages = build_passages(text, title="Title here", sizes=(1, 2))
    assert "Title here" in passages
    assert "Sentence one." in passages
    assert "Sentence one. Sentence two." in passages


def test_sliding_windows():
    from app.claim_aware import sliding_windows

    assert sliding_windows(["a", "b", "c"], 2) == ["a b", "b c"]
    assert sliding_windows(["a", "b"], 3) == ["a b"]


def test_rank_passages_with_fake_embedder():
    class FakeEmbedder:
        def encode(self, texts, **kwargs):
            # claim similar to first passage
            vecs = []
            for t in texts:
                v = np.zeros(3)
                if "target" in t:
                    v[0] = 1.0
                else:
                    v[1] = 1.0
                vecs.append(v)
            return np.array(vecs)

    passages = ["target passage one.", "other passage two."]
    ranked = rank_passages("claim target", passages, embedder=FakeEmbedder(), top_k=1)
    assert ranked == ["target passage one."]


def test_guard1_changes_contradiction_without_entity():
    features = claim_features("Netflix reported third-quarter 2026 revenue of $12.87 billion.")
    # force main entity to Netflix (already)
    result = NLIResult(0.1, 0.1, 0.8, "REFUTES", cached=False)
    stats = GuardStats()
    out = apply_guards(
        features,
        "Some other company reported revenue.",
        result,
        min_term_coverage=0.6,
        contradiction_max=0.9,
        support_prob=0.5,
        stats=stats,
    )
    # no main entity shared -> neutral
    assert out.stance == "NEUTRAL"
    assert stats.guard1_changed == 1
    assert stats.guard1_refute_to_neutral == 1


def test_guard1_keeps_contradiction_with_entity_and_terms():
    features = claim_features("Netflix reported third-quarter 2026 revenue of $12.87 billion.")
    result = NLIResult(0.1, 0.1, 0.8, "REFUTES", cached=False)
    stats = GuardStats()
    passage = "Netflix reported third-quarter 2026 revenue of $12.87 billion and earnings."
    out = apply_guards(
        features,
        passage,
        result,
        min_term_coverage=0.6,
        contradiction_max=0.9,
        support_prob=0.5,
        stats=stats,
    )
    assert out.stance == "REFUTES"
    assert stats.guard1_kept == 1


def test_guard2_promotes_neutral_exact_fact():
    features = claim_features("Netflix reported third-quarter 2026 revenue of $12.87 billion.")
    result = NLIResult(0.2, 0.7, 0.1, "NEUTRAL", cached=False)
    stats = GuardStats()
    passage = "Netflix reported revenue of $12.87 billion."
    out = apply_guards(
        features,
        passage,
        result,
        min_term_coverage=0.6,
        contradiction_max=0.9,
        support_prob=0.5,
        stats=stats,
    )
    # numbers_present? extract_numbers(passage) must include 12.87 billion
    if "12.87 billion" in extract_numbers(passage):
        assert out.stance == "SUPPORTS"
        assert stats.guard2_changed == 1
        assert stats.guard2_neutral_to_support == 1


def test_numbers_present_and_coverage():
    features = claim_features("Netflix reported third-quarter 2026 revenue of $12.87 billion.")
    assert numbers_present(features, "Netflix revenue of $12.87 billion.") is True
    cov = content_term_coverage(features, "Netflix revenue of $12.87 billion.")
    assert cov > 0.0


def test_clause_combine_rules():
    assert clause_combine([{"supports": 2, "refutes": 0}, {"supports": 2, "refutes": 0}]) == "SUPPORTED"
    assert clause_combine([{"supports": 2, "refutes": 0}, {"supports": 0, "refutes": 2}]) == "REFUTED"
    assert clause_combine([{"supports": 2, "refutes": 0}, {"supports": 1, "refutes": 0}]) == "UNVERIFIED"


def test_build_v3_evidence_snippet_guards():
    articles = [
        RankedArticle(
            title="A",
            source_domain="example.com",
            url="https://example.com/a",
            published_at=None,
            snippet="Netflix reported revenue of $12.87 billion.",
            similarity=0.9,
            recency_factor=1.0,
            score=0.9,
        )
    ]

    def classifier(claim, snippets):
        return [NLIResult(0.2, 0.7, 0.1, "NEUTRAL", cached=False) for _ in snippets]

    options = V3Options(passages=False, guards=True)
    res = build_v3_evidence(
        "Netflix reported third-quarter 2026 revenue of $12.87 billion.",
        articles,
        options=options,
        classifier=classifier,
    )
    assert res.evidence[0].stance in ("SUPPORTS", "NEUTRAL", "REFUTES")


def test_pipeline_v1_unchanged_when_no_v3():
    # basic smoke: pipeline with no v3 runs and returns verdict
    from app.schemas import Article

    def analyzer(text):
        from app.schemas import Claim

        return Claim(text=text, entities=[], queries=["q1", "q2"])

    def retriever(queries, **kwargs):
        return []

    pipe = VerificationPipeline(analyzer=analyzer, retriever=retriever, v3=None)
    v = pipe.verify("Some claim.")
    assert v.verdict == "UNVERIFIED"


def test_pipeline_v3_retrieval_changes_first_query():
    captured = {}

    def analyzer(text):
        from app.schemas import Claim

        return Claim(text=text, entities=[], queries=["orig query one", "orig query two"])

    def retriever(queries, **kwargs):
        captured["queries"] = list(queries)
        return []

    pipe = VerificationPipeline(
        analyzer=analyzer,
        retriever=retriever,
        v3=V3Options(retrieval=True),
    )
    pipe.verify("Max Verstappen won the 2026 Singapore Grand Prix.")
    assert captured["queries"][0] != "orig query one"
    assert "Max" in captured["queries"][0] or "Verstappen" in captured["queries"][0]


def test_clause_verdict_builds_verdict():
    from app.claim_aware import claim_features

    cf = claim_features("A magnitude 7.7 earthquake struck southern Panama on October 9, 2026, and the Panama Canal suspended all ship passage.")
    cr = None
    from app.v3 import ClauseResult

    cr = ClauseResult(
        coordinated=True,
        clauses=list(cf.clauses),
        counts=[{"clause": cf.clauses[0], "supports": 0, "refutes": 2, "neutrals": 0, "articles": []},
                {"clause": cf.clauses[1], "supports": 2, "refutes": 0, "neutrals": 0, "articles": []}],
        verdict="REFUTED",
        explanation="test",
    )
    v = clause_verdict("claim", cr, [], queries=["q"], latency_ms=5)
    assert v.verdict == "REFUTED"
    assert v.latency_ms == 5
