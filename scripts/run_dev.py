"""Run the dev claims through cached retrieval, ranking, and NLI.

This is a diagnostic script, not a decision engine. It never calls a live news
API: retrieval runs with ``cache_only=True``, so a missing cache entry appears as
zero evidence. NLI may load the local model and use its own pair cache.

Usage:
    python scripts/run_dev.py
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from app.claim_analysis import analyze
from app.nli import classify
from app.ranking import rank_articles
from app.retrieval import retrieve

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

CLAIMS_PATH = PROJECT_ROOT / "tests" / "data" / "claims.csv"
SPLIT = "dev"


def load_claims(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if (row.get("split") or "").strip() == SPLIT]


def shorten(text: str, limit: int = 180) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def main() -> int:
    claims = load_claims(CLAIMS_PATH)
    if not claims:
        print(f"No {SPLIT} claims found in {CLAIMS_PATH}")
        return 1

    print("Dev run: cached retrieval only, no live API calls")
    print(f"NLI model: {config.NLI_MODEL_NAME}")
    print(f"MIN_SIMILARITY: {config.MIN_SIMILARITY} (PROVISIONAL)")
    print(f"Claims: {len(claims)}")
    print()

    for number, row in enumerate(claims, start=1):
        claim = " ".join((row.get("claim") or "").split())
        expected = (row.get("label") or "").strip()
        print("=" * 110)
        print(f"Claim {number}: {claim}")
        print(f"Expected label: {expected}")

        try:
            analysis = analyze(claim)
            articles = retrieve(analysis.queries, cache_only=True)
            ranked = rank_articles(claim, articles)
            snippets = [item.snippet for item in ranked]
            results = classify(claim, snippets) if snippets else []
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
            print()
            continue

        sum_entail = sum(result.p_entail for result in results)
        sum_contra = sum(result.p_contra for result in results)
        print(
            f"Evidence kept: {len(ranked)} | "
            f"sum P(entail): {sum_entail:.3f} | "
            f"sum P(contra): {sum_contra:.3f}"
        )

        if not ranked:
            print("  No cached evidence passed MIN_SIMILARITY.")
            print()
            continue

        for index, (item, result) in enumerate(zip(ranked, results), start=1):
            print(
                f"  [{index}] {result.stance} "
                f"p_e={result.p_entail:.3f} "
                f"p_n={result.p_neutral:.3f} "
                f"p_c={result.p_contra:.3f} | "
                f"{item.source_domain} | sim={item.similarity:.3f} "
                f"score={item.score:.3f}"
            )
            print(f"      title: {shorten(item.title)}")
            print(f"      snippet: {shorten(item.snippet)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
