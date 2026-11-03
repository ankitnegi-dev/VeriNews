"""Re-measure the passage cold latency for one claim after a suspended run.

The original claim 10 cold measurement was 25 minutes because the process was
suspended during a server restart. Warm was 5 seconds, so the algorithm itself
was not the cause. This script clears that claim's NLI cache and measures cold
and warm again, then updates the resumable results file.

Usage:
    python scripts/remeasure_passage_claim.py 10
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import VerificationPipeline
from scripts.finish_premise_comparison import (
    RESUME_PATH,
    batched_classify,
    cache_paths_by_claim,
    cached_text,
    clear_claim,
    digest,
    load_claims,
)


def main() -> int:
    number = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    claims = load_claims()
    claim = claims[number - 1]["claim"]
    resume = json.loads(RESUME_PATH.read_text(encoding="utf-8"))
    if claim not in resume:
        raise SystemExit(f"Claim {number} is not in {RESUME_PATH}")

    pipeline = VerificationPipeline(
        cache_only=True,
        premise_mode="passage",
        classifier=batched_classify,
        full_text_getter=cached_text,
    )
    pipeline.warmup()
    cache_paths = cache_paths_by_claim()
    removed = clear_claim(cache_paths, claim)
    cold = pipeline.verify(claim)
    warm = pipeline.verify(claim)

    resume[claim] = {
        "verdict": cold.verdict,
        "explanation": cold.explanation,
        "evidence": digest(cold),
        "latency_cold_ms": cold.latency_ms,
        "latency_warm_ms": warm.latency_ms,
        "cold_equals_warm": cold.verdict == warm.verdict,
    }
    RESUME_PATH.write_text(
        json.dumps(resume, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"claim {number}: {cold.verdict} cold={cold.latency_ms}ms "
        f"warm={warm.latency_ms}ms cleared={removed}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
