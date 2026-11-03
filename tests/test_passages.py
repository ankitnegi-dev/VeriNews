"""Offline tests for sentence windows, passage ranking, and aggregation."""

from __future__ import annotations

import numpy as np

from app.nli import NLICache
from app.passages import (
    aggregate_passage_results,
    build_passages,
    premises_for_article,
    rank_passages,
    sliding_windows,
    split_sentences,
)
from app.schemas import NLIResult


class MarkerEmbedder:
    """Claim-like text points at [1, 0]; filler points at [0, 1]."""

    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            lowered = sentence.lower()
            if "claim" in lowered or "river" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=float)


def test_split_sentences_and_sliding_windows():
    text = "First sentence. Second sentence! Third sentence?"

    sentences = split_sentences(text)
    assert sentences == ["First sentence.", "Second sentence!", "Third sentence?"]
    assert sliding_windows(sentences, 2) == [
        "First sentence. Second sentence!",
        "Second sentence! Third sentence?",
    ]
    assert sliding_windows(["Only one."], 2) == ["Only one."]


def test_build_passages_includes_title_and_two_sentence_windows():
    passages = build_passages(
        "The river rose quickly. The council met on Tuesday. Work starts in March.",
        title="Council approves river plan",
    )

    assert passages[0] == "Council approves river plan"
    assert "The river rose quickly. The council met on Tuesday." in passages
    assert "The council met on Tuesday. Work starts in March." in passages
    assert len(passages) == 3


def test_rank_passages_returns_the_most_claim_like_windows():
    passages = [
        "Weather report for the weekend.",
        "The claim about the river is discussed here.",
        "The river plan was approved by the council.",
        "Sports results from the weekend.",
    ]

    selected = rank_passages(
        "claim about the river",
        passages,
        embedder=MarkerEmbedder(),
        top_k=2,
    )

    assert selected == [
        "The claim about the river is discussed here.",
        "The river plan was approved by the council.",
    ]


def test_premises_for_article_falls_back_to_snippet_without_text():
    assert premises_for_article(
        "The river plan was approved.",
        "",
        title="Council approves river plan",
        snippet="The council approved the plan.",
        embedder=MarkerEmbedder(),
    ) == ["The council approved the plan."]


def test_nli_cache_key_includes_the_passage_text():
    cache = NLICache(directory="unused")
    first = cache.path_for("A claim.", "Passage one.")
    second = cache.path_for("A claim.", "Passage two.")

    assert first != second


def test_aggregate_uses_maximum_entailment_and_contradiction():
    results = [
        NLIResult(0.82, 0.10, 0.08, "SUPPORTS"),
        NLIResult(0.05, 0.50, 0.45, "NEUTRAL"),
        NLIResult(0.10, 0.85, 0.05, "NEUTRAL"),
    ]

    aggregated = aggregate_passage_results(results)

    assert aggregated.p_entail == 0.82
    assert aggregated.p_contra == 0.45
    assert aggregated.p_neutral == 0.85
    assert aggregated.stance == "SUPPORTS"


def test_aggregate_marks_conflicting_high_scores_as_neutral():
    results = [
        NLIResult(0.90, 0.05, 0.05, "SUPPORTS"),
        NLIResult(0.05, 0.05, 0.90, "REFUTES"),
    ]

    aggregated = aggregate_passage_results(results)

    assert aggregated.p_entail == 0.90
    assert aggregated.p_contra == 0.90
    assert aggregated.stance == "NEUTRAL"
