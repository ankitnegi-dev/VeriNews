"""Compare snippet and passage NLI premises on all 30 development claims.

Retrieval and full text are cache-only. Cold latency means the NLI pair cache
for these claims was cleared first; warm latency is a second pass over the same
claims. This is development evidence, not held-out evaluation.

Usage:
    python scripts/compare_premise_modes.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.nli import NLICache, classify
from app.pipeline import VerificationPipeline
from scripts.evaluate import compute_metrics

CLAIMS_PATH = PROJECT_ROOT / "tests" / "data" / "claims.csv"
PROGRESS_PATH = PROJECT_ROOT / ".cache" / "premise_comparison_progress.json"
REPORT_PATH = PROJECT_ROOT / "notes" / "premise_comparison.md"
JSON_PATH = PROJECT_ROOT / "notes" / "premise_comparison.json"
NLI_BATCH = 16


def load_all_claims() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with CLAIMS_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            claim = " ".join((row.get("claim") or "").split())
            label = (row.get("label") or "").strip().upper()
            if claim:
                rows.append({"claim": claim, "label": label})
    return rows


def clear_claim_cache(claims: list[dict[str, str]]) -> int:
    """Delete every NLI pair cache entry belonging to these claims."""
    wanted = {row["claim"] for row in claims}
    cache = NLICache()
    removed = 0
    for path in cache.directory.glob("nli_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        claim = payload.get("claim")
        if isinstance(claim, str) and " ".join(claim.split()) in wanted:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def batched_classify(claim: str, snippets):
    return classify(claim, snippets, batch_size=NLI_BATCH)


def run_mode(mode: str, claims: list[dict[str, str]]) -> dict[str, object]:
    removed = clear_claim_cache(claims)
    print(f"[{mode}] cleared {removed} NLI cache entries", flush=True)

    pipeline = VerificationPipeline(
        cache_only=True,
        premise_mode=mode,
        classifier=batched_classify,
    )
    pipeline.warmup()

    cold_verdicts = []
    cold_latencies: list[int] = []
    for number, row in enumerate(claims, start=1):
        started = time.perf_counter()
        verdict = pipeline.verify(row["claim"])
        cold_verdicts.append(verdict)
        cold_latencies.append(verdict.latency_ms)
        print(
            f"[{mode}] cold {number}/{len(claims)} {verdict.verdict} "
            f"{verdict.latency_ms}ms wall={round((time.perf_counter() - started) * 1000)}ms",
            flush=True,
        )

    warm_verdicts = []
    warm_latencies: list[int] = []
    for number, row in enumerate(claims, start=1):
        verdict = pipeline.verify(row["claim"])
        warm_verdicts.append(verdict)
        warm_latencies.append(verdict.latency_ms)
        print(
            f"[{mode}] warm {number}/{len(claims)} {verdict.verdict} "
            f"{verdict.latency_ms}ms",
            flush=True,
        )

    expected = [row["label"] for row in claims]
    predicted = [verdict.verdict for verdict in cold_verdicts]
    metrics = compute_metrics(expected, predicted)
    metrics["mean_latency_cold_ms"] = round(statistics.mean(cold_latencies))
    metrics["mean_latency_warm_ms"] = round(statistics.mean(warm_latencies))
    metrics["max_latency_cold_ms"] = max(cold_latencies)
    metrics["max_latency_warm_ms"] = max(warm_latencies)

    return {
        "mode": mode,
        "cold": cold_verdicts,
        "warm": warm_verdicts,
        "metrics": metrics,
    }


def evidence_digest(verdict) -> list[str]:
    return [
        f"{item.source_domain}: {item.stance} "
        f"(entail {item.p_entail:.2f}, contra {item.p_contra:.2f})"
        for item in verdict.evidence
    ]


def main() -> int:
    claims = load_all_claims()
    print(f"claims={len(claims)}", flush=True)

    results: dict[str, dict[str, object]] = {}
    for mode in ("snippet", "passage"):
        results[mode] = run_mode(mode, claims)
        PROGRESS_PATH.write_text(
            json.dumps(
                {
                    "completed": list(results),
                    "metrics": {
                        name: data["metrics"] for name, data in results.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    snippet = results["snippet"]
    passage = results["passage"]
    sm = snippet["metrics"]
    pm = passage["metrics"]

    changed = []
    regressions = []
    for index, row in enumerate(claims):
        snippet_verdict = snippet["cold"][index]
        passage_verdict = passage["cold"][index]
        if snippet_verdict.verdict != passage_verdict.verdict:
            changed.append((row, snippet_verdict, passage_verdict))
        if (
            row["label"] == "UNVERIFIED"
            and snippet_verdict.verdict == "UNVERIFIED"
            and passage_verdict.verdict in {"SUPPORTED", "REFUTED"}
        ):
            regressions.append((row, passage_verdict))

    lines: list[str] = []
    lines.append("# Passage-level NLI comparison")
    lines.append("")
    lines.append("**Development evidence, not held-out.**")
    lines.append("")
    lines.append(
        "All 30 claims in `tests/data/claims.csv` (dev + test rows) were run "
        "cache-only. Retrieval and full text came from local caches; only the "
        "NLI pair cache was cleared for the cold pass. This is development "
        "evidence, not held-out evaluation."
    )
    lines.append("")
    lines.append("| Metric | Snippet mode | Passage mode |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| Exact accuracy | {sm['exact_accuracy']:.3f} "
        f"({sm['correct']}/{sm['total']}) | {pm['exact_accuracy']:.3f} "
        f"({pm['correct']}/{pm['total']}) |"
    )
    lines.append(
        f"| Wrong-verdict rate | {sm['wrong_verdict_rate']:.3f} "
        f"({sm['wrong_verdict_flips']}/{sm['directional_gold']}) | "
        f"{pm['wrong_verdict_rate']:.3f} "
        f"({pm['wrong_verdict_flips']}/{pm['directional_gold']}) |"
    )
    lines.append(
        f"| Coverage | {sm['coverage']:.3f} "
        f"({sm['committed_directional']}/{sm['directional_gold']}) | "
        f"{pm['coverage']:.3f} "
        f"({pm['committed_directional']}/{pm['directional_gold']}) |"
    )
    lines.append(
        f"| Mean latency, cold (ms) | {sm['mean_latency_cold_ms']} | "
        f"{pm['mean_latency_cold_ms']} |"
    )
    lines.append(
        f"| Mean latency, warm (ms) | {sm['mean_latency_warm_ms']} | "
        f"{pm['mean_latency_warm_ms']} |"
    )
    lines.append(
        f"| Max latency, cold (ms) | {sm['max_latency_cold_ms']} | "
        f"{pm['max_latency_cold_ms']} |"
    )
    lines.append(
        f"| Max latency, warm (ms) | {sm['max_latency_warm_ms']} | "
        f"{pm['max_latency_warm_ms']} |"
    )
    lines.append("")
    lines.append("## Claims whose verdict changed")
    lines.append("")
    if not changed:
        lines.append("No claim changed verdict between snippet and passage mode.")
    else:
        for row, snippet_verdict, passage_verdict in changed:
            lines.append(f"### {row['claim']}")
            lines.append("")
            lines.append(f"- Gold: `{row['label']}`")
            lines.append(
                f"- Snippet: `{snippet_verdict.verdict}` — "
                f"{snippet_verdict.explanation}"
            )
            lines.append(
                f"- Passage: `{passage_verdict.verdict}` — "
                f"{passage_verdict.explanation}"
            )
            snippet_evidence = evidence_digest(snippet_verdict)
            passage_evidence = evidence_digest(passage_verdict)
            if snippet_evidence or passage_evidence:
                lines.append("- Snippet evidence:")
                lines.extend(f"  - {item}" for item in snippet_evidence)
                lines.append("- Passage evidence:")
                lines.extend(f"  - {item}" for item in passage_evidence)
            lines.append("")
    lines.append("## Correct abstention to wrong verdict")
    lines.append("")
    if regressions:
        for row, verdict in regressions:
            lines.append(
                f"- {row['claim']} -> `{verdict.verdict}`: "
                f"{verdict.explanation}"
            )
    else:
        lines.append(
            "None. No claim went from a correct abstention to a wrong verdict."
        )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "note": "Development evidence, not held-out.",
        "claims": len(claims),
        "snippet": sm,
        "passage": pm,
        "changed": [
            {
                "claim": row["claim"],
                "gold": row["label"],
                "snippet": snippet_verdict.verdict,
                "passage": passage_verdict.verdict,
                "snippet_explanation": snippet_verdict.explanation,
                "passage_explanation": passage_verdict.explanation,
            }
            for row, snippet_verdict, passage_verdict in changed
        ],
        "regressions": [
            {"claim": row["claim"], "passage": verdict.verdict}
            for row, verdict in regressions
        ],
    }
    JSON_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"changed={len(changed)} regressions={len(regressions)}", flush=True)
    print(json.dumps({"snippet": sm, "passage": pm}, indent=2), flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
