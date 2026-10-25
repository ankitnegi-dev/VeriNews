"""Offline tests for NLI label mapping, stance thresholds, and caching."""

from __future__ import annotations

import pytest

import app.nli as nli
from app.schemas import NLIResult


def test_find_label_index_does_not_assume_order():
    id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

    assert nli.find_label_index(id2label, "entail") == 2
    assert nli.find_label_index(id2label, "neutral") == 1
    assert nli.find_label_index(id2label, "contra") == 0


def test_choose_stance_uses_probabilities():
    assert nli.choose_stance(0.6, 0.2, 0.2) == "SUPPORTS"
    assert nli.choose_stance(0.2, 0.2, 0.6) == "REFUTES"
    assert nli.choose_stance(0.3, 0.4, 0.3) == "NEUTRAL"


def test_cache_roundtrip_and_model_mismatch(tmp_path):
    cache = nli.NLICache(tmp_path)
    result = NLIResult(0.81, 0.12, 0.07, "SUPPORTS")

    cache.set("Claim text.", "Snippet text.", result)
    loaded = cache.get("Claim text.", "Snippet text.")

    assert loaded is not None
    assert loaded.cached is True
    assert loaded.p_entail == pytest.approx(0.81)
    assert loaded.stance == "SUPPORTS"

    other_model = nli.NLICache(tmp_path, model_name="other/model")
    assert other_model.get("Claim text.", "Snippet text.") is None


def test_cache_key_ignores_extra_whitespace():
    first = nli.cache_key("Claim   text.", "Snippet  text.")
    second = nli.cache_key(" Claim text. ", "Snippet text. ")

    assert first == second


def test_classify_uses_cache_without_loading_the_model(tmp_path, monkeypatch):
    cache = nli.NLICache(tmp_path)
    claim = "Claim text."
    snippets = ["Snippet one.", "Snippet two."]
    for snippet in snippets:
        cache.set(claim, snippet, NLIResult(0.9, 0.05, 0.05, "SUPPORTS"))

    def fail_if_loaded():
        raise AssertionError("The model should not load when every pair is cached.")

    monkeypatch.setattr(nli, "_get_model", fail_if_loaded)
    results = nli.classify(claim, snippets, cache=cache)

    assert len(results) == 2
    assert all(result.cached for result in results)
    assert all(result.stance == "SUPPORTS" for result in results)
