"""Tests for Phase 2 claim analysis.

The tests use the real small spaCy model (no network once installed). They check
entity extraction, query shape, deduplication and clear input validation.
"""

from __future__ import annotations

import pytest

from app.claim_analysis import ENTITY_LABELS, analyze, build_queries
from app.schemas import Claim


SAMPLES = [
    (
        "The Eiffel Tower is located in Paris, France.",
        {"Paris", "France"},
    ),
    (
        "Apple announced the iPhone 15 at an event in Cupertino on September 12, 2023.",
        {"Apple", "Cupertino", "September 12, 2023"},
    ),
    (
        "Narendra Modi met Joe Biden in New Delhi on Friday.",
        {"Narendra Modi", "Joe Biden", "New Delhi", "Friday"},
    ),
    (
        "The World Health Organization declared COVID-19 a pandemic in March 2020.",
        {"The World Health Organization", "March 2020"},
    ),
    (
        "Tesla reported a profit of $2.5 billion in the fourth quarter of 2023.",
        {"Tesla", "$2.5 billion", "the fourth quarter of 2023"},
    ),
]


@pytest.mark.parametrize("text,expected_entities", SAMPLES)
def test_analyze_extracts_entities_and_queries(text, expected_entities):
    claim = analyze(text)

    assert isinstance(claim, Claim)
    assert claim.text == text
    entity_texts = {entity["text"] for entity in claim.entities}
    assert expected_entities <= entity_texts
    assert all(entity["label"] in ENTITY_LABELS for entity in claim.entities)

    assert 2 <= len(claim.queries) <= 3
    assert len(claim.queries) == len({query.lower() for query in claim.queries})
    assert all(2 <= len(query.split()) <= 12 for query in claim.queries)
    assert any(
        expected.lower() in query.lower()
        for query in claim.queries
        for expected in expected_entities
    )


@pytest.mark.parametrize("bad_claim", ["", "   ", "Too short"])
def test_empty_or_short_claim_raises_value_error(bad_claim):
    with pytest.raises(ValueError, match="Claim is (empty|too short)"):
        analyze(bad_claim)


def test_non_string_claim_raises_value_error():
    with pytest.raises(ValueError, match="must be a string"):
        analyze(None)


def test_claim_without_entities_still_builds_queries():
    claim = analyze("Inflation rose sharply.")
    assert claim.entities == []
    assert len(claim.queries) >= 2


def test_build_queries_is_public_and_returns_unique_queries():
    claim = analyze("Narendra Modi met Joe Biden in New Delhi on Friday.")
    queries = build_queries(claim.text, claim.entities)
    assert 2 <= len(queries) <= 3
    assert len(queries) == len({query.lower() for query in queries})
