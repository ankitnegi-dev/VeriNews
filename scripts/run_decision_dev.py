"""Run the decision engine on the ten dev claims using cached retrieval only.

This script never calls a live news API. It filters ``split=dev`` and does not
touch the test split. NLI uses its local pair cache and may load the local model
only if a pair is missing.

Usage:
    python scripts/run_decision_dev.py
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from app.claim_analysis import analyze
from app.credibility import load_credibility
from app.decision import build_evidence, decide
from app.nli import classify
from app.ranking import rank_articles
from app.retrieval import retrieve

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

CLAIMS_PATH = PROJECT_ROOT / "tests" / "data" / "claims.csv"
SPLIT = "dev"


def load_claims(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if (row.get("split") or "").strip() == split]


def shorten(text: str, limit: int = 58) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def main() -> int:
    claims = load_claims(CLAIMS_PATH, SPLIT)
    if not claims:
        print(f"No {SPLIT} claims found in {CLAIMS_PATH}")
        return 1

    table = load_credibility()

    print("Decision dev run: cached retrieval only, split=dev")
    print(f"NLI model: {config.NLI_MODEL_NAME}")
    print(f"MIN_EVIDENCE: {config.MIN_EVIDENCE} (fixed safety floor)")
    print(
        "Tunables: "
        f"REL_MIN={config.REL_MIN} "
        f"MIN_TOTAL_WEIGHT={config.MIN_TOTAL_WEIGHT} "
        f"DECISION_RATIO={config.DECISION_RATIO} "
        f"MARGIN={config.MARGIN}"
    )
    print(f"Claims: {len(claims)}")
    print()

    records: list[dict[str, object]] = []
    for number, row in enumerate(claims, start=1):
        claim = " ".join((row.get("claim") or "").split())
        expected = (row.get("label") or "").strip()
        started = time.perf_counter()

        try:
            analysis = analyze(claim)
            articles = retrieve(analysis.queries, cache_only=True)
            ranked = rank_articles(claim, articles)
            snippets = [item.snippet for item in ranked]
            results = classify(claim, snippets) if snippets else []
            evidence = build_evidence(
                ranked,
                results,
                credibility_table=table,
            )
            verdict = decide(
                claim,
                evidence,
                queries=analysis.queries,
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            print(f"[{number}] ERROR: {type(exc).__name__}: {exc}")
            continue

        records.append(
            {
                "number": number,
                "claim": claim,
                "expected": expected,
                "verdict": verdict,
                "kept": len(ranked),
            }
        )

    print(
        "| # | claim | expected | predicted | confidence | kept | "
        "support | refute |"
    )
    print("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for record in records:
        verdict = record["verdict"]
        print(
            f"| {record['number']} | {shorten(str(record['claim']))} | "
            f"{record['expected']} | {verdict.verdict} | "
            f"{verdict.confidence:.3f} | {record['kept']} | "
            f"{verdict.support_score:.3f} | {verdict.refute_score:.3f} |"
        )

    print()
    for record in records:
        verdict = record["verdict"]
        print(
            f"{record['number']}. expected={record['expected']} "
            f"predicted={verdict.verdict} confidence={verdict.confidence:.3f}"
        )
        print(f"   {verdict.explanation}")
        for item in verdict.evidence:
            print(
                f"   - {item.stance:<8} sim={item.similarity:.3f} "
                f"cred={item.credibility:.2f} {item.source_domain}"
            )

    if records:
        agreement = sum(
            1
            for record in records
            if record["verdict"].verdict == record["expected"]
        )
        print()
        print(f"Dev exact agreement: {agreement}/{len(records)}")
        print("No test-split claims were loaded or run.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
