"""Run the offline demo through the same pipeline the API uses.

Cache-only mode never calls a live news API. Retrieval reads ``.cache/retrieval``
and accepts stale entries. NLI pairs are also served from the local cache once
warmed, so repeated demos are fast.

Usage:
    python scripts/run_demo_cache.py
    python scripts/run_demo_cache.py --split dev
    python scripts/run_demo_cache.py --split all --limit 5
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import VerificationPipeline

CLAIMS_PATH = PROJECT_ROOT / "tests" / "data" / "claims.csv"
VALID_SPLITS = ("dev", "test")
VERDICT_LABELS = {
    "SUPPORTED": "Supported",
    "REFUTED": "Refuted",
    "UNVERIFIED": "Not enough evidence",
}
FAILURE_MARKERS = (" failed (", "Unexpected pipeline failure", "Claim is empty")


def load_claims(path: Path, split: str) -> list[dict[str, str]]:
    """Read the claim text and split without using the label for evaluation."""
    if not path.exists():
        raise SystemExit(f"Claims file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"claim", "split"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Claims file is missing columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    selected: list[dict[str, str]] = []
    for row in rows:
        row_split = (row.get("split") or "").strip().lower()
        claim = " ".join((row.get("claim") or "").split())
        if not claim or row_split not in VALID_SPLITS:
            continue
        if split != "all" and row_split != split:
            continue
        selected.append({"claim": claim, "split": row_split})
    return selected


def shorten(text: str, limit: int = 64) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the VeriNews demo with cache-only retrieval."
    )
    parser.add_argument(
        "--split",
        choices=("all", "dev", "test"),
        default="all",
        help="Which claims to run. Labels are never evaluated here.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N selected claims (0 means all).",
    )
    parser.add_argument(
        "--claims-path",
        type=Path,
        default=CLAIMS_PATH,
        help="CSV file with claim and split columns.",
    )
    args = parser.parse_args()

    claims = load_claims(args.claims_path, args.split)
    if args.limit > 0:
        claims = claims[: args.limit]

    if not claims:
        print(f"No claims found for split={args.split}.")
        return 1

    pipeline = VerificationPipeline(cache_only=True)
    print("VeriNews cache-only demo")
    print("Network calls: none (retrieval cache, local NLI model only)")
    print(f"Claims selected: {len(claims)}")
    print()

    rows: list[tuple[int, str, str, str, float, int, int]] = []
    failures = 0
    wall_started = time.perf_counter()

    for number, item in enumerate(claims, start=1):
        verdict = pipeline.verify(item["claim"])
        label = VERDICT_LABELS.get(verdict.verdict, verdict.verdict)
        if any(marker in verdict.explanation for marker in FAILURE_MARKERS):
            failures += 1
        rows.append(
            (
                number,
                item["split"],
                shorten(item["claim"]),
                label,
                verdict.confidence,
                len(verdict.evidence),
                verdict.latency_ms,
            )
        )

    wall_ms = round((time.perf_counter() - wall_started) * 1000)
    print("| # | split | claim | result | confidence | evidence | ms |")
    print("| ---: | --- | --- | --- | ---: | ---: | ---: |")
    for number, split, claim, label, confidence, evidence, latency in rows:
        print(
            f"| {number} | {split} | {claim} | {label} | "
            f"{confidence:.3f} | {evidence} | {latency} |"
        )

    latencies = [row[6] for row in rows]
    print()
    print(f"Wall time: {wall_ms} ms")
    print(f"Pipeline latency: median={statistics.median(latencies):.0f} ms, "
          f"max={max(latencies)} ms")
    if failures:
        print(f"Degraded responses: {failures} (check the explanation text)")
    else:
        print("Degraded responses: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
