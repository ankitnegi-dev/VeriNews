"""Tests for Phase 2 claim analysis.

The tests use the real small spaCy model (no network once installed). They check
entity extraction, query shape, deduplication and clear input validation.
"""

from __future__ import annotations

import pytest

from app.claim_analysis import (
    _add_query,
    _is_low_value_query_entity,
    _query_entity_text,
    analyze,
    build_queries,
)
from app.schemas import Claim
from config import ENTITY_LABELS


SAMPLES = [
    (
        "The Eiffel Tower is located in Paris, France.",
        {"The Eiffel Tower", "Paris", "France"},
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


def test_query_entity_text_strips_leading_articles():
    assert _query_entity_text("The Eiffel Tower") == "Eiffel Tower"
    assert _query_entity_text("a Tesla Model 3") == "Tesla Model 3"
    assert _query_entity_text("an Apple event") == "Apple event"
    assert _query_entity_text("Paris") == "Paris"


def test_age_phrase_is_marked_as_low_value():
    assert _is_low_value_query_entity("the age of 93")
    assert _is_low_value_query_entity("age of 42")
    assert not _is_low_value_query_entity("March 2020")
    assert not _is_low_value_query_entity("Tom Skerritt")


def test_skerritt_claim_does_not_query_the_age_phrase():
    claim = analyze(
        "Tom Skerritt, the actor known for Top Gun and Alien, has died at the age of 93."
    )

    assert 2 <= len(claim.queries) <= 3
    assert any("Tom Skerritt" in query for query in claim.queries)
    assert not any("age of 93" in query.lower() for query in claim.queries)


def test_entity_queries_strip_leading_the():
    claim = analyze("The Eiffel Tower is located in Paris, France.")

    assert any("Eiffel Tower" in query for query in claim.queries)
    assert not any("The Eiffel Tower" in query for query in claim.queries)


def test_build_queries_is_public_and_returns_unique_queries():
    claim = analyze("Narendra Modi met Joe Biden in New Delhi on Friday.")
    queries = build_queries(claim.text, claim.entities)
    assert 2 <= len(queries) <= 3
    assert len(queries) == len({query.lower() for query in queries})


def test_landmark_entity_appears_in_entities_and_queries():
    claim = analyze("The Eiffel Tower is in Paris")
    entity_texts = [entity["text"] for entity in claim.entities]
    assert any("Eiffel Tower" in text for text in entity_texts)
    assert any("Eiffel Tower" in query for query in claim.queries)


def test_money_query_preserves_the_original_span():
    claim = analyze(
        "Tesla reported a profit of $2.5 billion in the fourth quarter of 2023."
    )
    assert any(entity["text"] == "$2.5 billion" for entity in claim.entities)
    assert any("$2.5 billion" in query for query in claim.queries)
    assert not any("$ 2.5" in query for query in claim.queries)


def test_query_dedupe_ignores_word_order():
    queries: list[str] = []
    seen: set[str] = set()
    _add_query(queries, seen, "Narendra Modi met Joe Biden")
    _add_query(queries, seen, "Joe Biden met Narendra Modi")
    assert queries == ["Narendra Modi met Joe Biden"]
