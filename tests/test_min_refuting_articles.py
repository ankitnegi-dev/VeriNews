import csv
import json

import config
from app.decision import decide
from app.schemas import Evidence


def _to_evidence(ev):
    return [
        Evidence(
            title=e.get("title", ""),
            source_domain=e.get("source_domain", e.get("domain", "")),
            url=e.get("url", ""),
            published_at=e.get("published_at"),
            snippet=e.get("snippet", ""),
            similarity=e.get("similarity", 1.0),
            credibility=e.get("credibility", 0.5),
            p_entail=e["p_entail"],
            p_neutral=e["p_neutral"],
            p_contra=e["p_contra"],
            stance=e["stance"],
        )
        for e in ev
    ]


def _load_claims():
    claims = []
    for path in ("results/eval_dev.json", "results/eval_test.json"):
        d = json.load(open(path, encoding="utf-8"))
        for r in d["records"]:
            claims.append((r["claim"], r["expected"], r["predicted"], r["evidence"]))
    fresh = json.load(open("notes/fresh20_v1.json", encoding="utf-8"))
    rows = list(csv.DictReader(open("tests/data/fresh_claims_20.csv", encoding="utf-8", newline="")))
    for r, row in zip(fresh, rows):
        claims.append((r["claim"], row["label"], r["verdict"], r["evidence"]))
    return claims


def test_default_two_reproduces_cached_v1():
    claims = _load_claims()
    config.MIN_REFUTING_ARTICLES = 2
    mismatches = 0
    for claim, gold, cached, ev in claims:
        v = decide(claim, _to_evidence(ev), queries=[], latency_ms=0)
        if v.verdict != cached:
            mismatches += 1
    assert mismatches == 0, f"{mismatches} mismatches with MIN_REFUTING_ARTICLES=2"


def test_three_reproduces_gate_row():
    claims = _load_claims()
    config.MIN_REFUTING_ARTICLES = 3
    kept = removed = lost = 0
    correct_sr = 0
    for claim, gold, cached, ev in claims:
        v = decide(claim, _to_evidence(ev), queries=[], latency_ms=0)
        if gold == "REFUTED" and cached == "REFUTED" and v.verdict == "REFUTED":
            kept += 1
        if gold != "REFUTED" and cached == "REFUTED" and v.verdict != "REFUTED":
            removed += 1
        if gold == "REFUTED" and cached == "REFUTED" and v.verdict != "REFUTED":
            lost += 1
        if gold in ("SUPPORTED", "REFUTED") and v.verdict == gold:
            correct_sr += 1
    assert kept == 9, kept
    assert removed == 4, removed
    assert lost == 1, lost
    assert correct_sr == 12, correct_sr
    # restore default
    config.MIN_REFUTING_ARTICLES = 2
