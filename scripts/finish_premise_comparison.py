"""Finish the snippet-vs-passage comparison without re-running snippet mode.

Snippet predictions and metrics are recovered from the completed snippet pass in
``.cache/compare_premise_modes.log`` and
``.cache/premise_comparison_progress.json``. Passage mode is resumable: each
claim's verdict is written to ``.cache/passage_resume.json`` after it finishes,
so a restart does not lose completed claims.

Usage:
    python scripts/finish_premise_comparison.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.fulltext import read_cached_text
from app.nli import NLICache, classify
from app.pipeline import VerificationPipeline
from scripts.evaluate import compute_metrics

CLAIMS_PATH = PROJECT_ROOT / "tests" / "data" / "claims.csv"
LOG_PATH = PROJECT_ROOT / ".cache" / "compare_premise_modes.log"
SNIPPET_PROGRESS_PATH = PROJECT_ROOT / ".cache" / "premise_comparison_progress.json"
RESUME_PATH = PROJECT_ROOT / ".cache" / "passage_resume.json"
REPORT_PATH = PROJECT_ROOT / "notes" / "premise_comparison.md"
JSON_PATH = PROJECT_ROOT / "notes" / "premise_comparison.json"
NLI_BATCH = 16


def load_claims() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with CLAIMS_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            claim = " ".join((row.get("claim") or "").split())
            label = (row.get("label") or "").strip().upper()
            if claim:
                rows.append({"claim": claim, "label": label})
    return rows


def read_log_text(path: Path) -> str:
    """Read a PowerShell redirect log, which may be UTF-16 or UTF-8."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-8"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "[snippet] cold" in text:
            return text
    return raw.decode("utf-8", errors="replace")


def load_snippet_results() -> tuple[list[str], dict[str, object]]:
    pattern = re.compile(
        r"\[snippet\] cold (\d+)/30 (SUPPORTED|REFUTED|UNVERIFIED) (\d+)ms"
    )
    predictions: dict[int, str] = {}
    for line in read_log_text(LOG_PATH).splitlines():
        match = pattern.search(line)
        if match:
            predictions[int(match.group(1))] = match.group(2)
    if len(predictions) != 30:
        raise SystemExit(
            f"Expected 30 snippet predictions in {LOG_PATH}, found {len(predictions)}"
        )

    progress = json.loads(SNIPPET_PROGRESS_PATH.read_text(encoding="utf-8"))
    metrics = progress["metrics"]["snippet"]
    return [predictions[index] for index in range(1, 31)], metrics


def cache_paths_by_claim() -> dict[str, list[Path]]:
    cache = NLICache()
    paths: dict[str, list[Path]] = {}
    for path in cache.directory.glob("nli_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        claim = payload.get("claim")
        if isinstance(claim, str):
            key = " ".join(claim.split())
            paths.setdefault(key, []).append(path)
    return paths


def clear_claim(cache_paths: dict[str, list[Path]], claim: str) -> int:
    removed = 0
    for path in cache_paths.pop(claim, []):
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def batched_classify(claim: str, snippets):
    return classify(claim, snippets, batch_size=NLI_BATCH)


def cached_text(url: str) -> str:
    return read_cached_text(url) or ""


def digest(verdict) -> list[str]:
    return [
        f"{item.source_domain}: {item.stance} "
        f"(entail {item.p_entail:.2f}, contra {item.p_contra:.2f})"
        for item in verdict.evidence
    ]


def serialise(verdict, latency_cold: int, latency_warm: int) -> dict[str, object]:
    return {
        "verdict": verdict.verdict,
        "explanation": verdict.explanation,
        "evidence": digest(verdict),
        "latency_cold_ms": latency_cold,
        "latency_warm_ms": latency_warm,
    }


def main() -> int:
    claims = load_claims()
    snippet_predictions, snippet_metrics = load_snippet_results()
    print(f"claims={len(claims)} snippet predictions recovered", flush=True)

    resume: dict[str, dict[str, object]] = {}
    if RESUME_PATH.exists():
        try:
            resume = json.loads(RESUME_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            resume = {}

    pipeline = VerificationPipeline(
        cache_only=True,
        premise_mode="passage",
        classifier=batched_classify,
        full_text_getter=cached_text,
    )
    pipeline.warmup()
    cache_paths = cache_paths_by_claim()

    for number, row in enumerate(claims, start=1):
        claim = row["claim"]
        if claim in resume:
            print(f"[passage] {number}/30 already done", flush=True)
            continue
        removed = clear_claim(cache_paths, claim)
        cold = pipeline.verify(claim)
        warm = pipeline.verify(claim)
        resume[claim] = serialise(cold, cold.latency_ms, warm.latency_ms)
        resume[claim]["cold_equals_warm"] = cold.verdict == warm.verdict
        RESUME_PATH.write_text(
            json.dumps(resume, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[passage] {number}/30 {cold.verdict} cold={cold.latency_ms}ms "
            f"warm={warm.latency_ms}ms cleared={removed}",
            flush=True,
        )

    passage_records = [resume[row["claim"]] for row in claims]
    passage_predictions = [record["verdict"] for record in passage_records]
    expected = [row["label"] for row in claims]
    passage_metrics = compute_metrics(expected, passage_predictions)
    cold_latencies = [int(record["latency_cold_ms"]) for record in passage_records]
    warm_latencies = [int(record["latency_warm_ms"]) for record in passage_records]
    passage_metrics["mean_latency_cold_ms"] = round(statistics.mean(cold_latencies))
    passage_metrics["mean_latency_warm_ms"] = round(statistics.mean(warm_latencies))
    passage_metrics["max_latency_cold_ms"] = max(cold_latencies)
    passage_metrics["max_latency_warm_ms"] = max(warm_latencies)

    changed_indices = [
        index
        for index, (snippet_prediction, passage_prediction) in enumerate(
            zip(snippet_predictions, passage_predictions)
        )
        if snippet_prediction != passage_prediction
    ]
    regressions = [
        index
        for index, row in enumerate(claims)
        if row["label"] == "UNVERIFIED"
        and snippet_predictions[index] == "UNVERIFIED"
        and passage_predictions[index] in {"SUPPORTED", "REFUTED"}
    ]

    # Recompute snippet verdicts only for changed claims so the report can
    # explain the snippet side. This runs after every passage claim is saved.
    snippet_pipeline = VerificationPipeline(
        cache_only=True,
        premise_mode="snippet",
        classifier=batched_classify,
    )
    snippet_explanations: dict[int, dict[str, object]] = {}
    for index in changed_indices:
        claim = claims[index]["claim"]
        clear_claim(cache_paths, claim)
        verdict = snippet_pipeline.verify(claim)
        snippet_explanations[index] = {
            "verdict": verdict.verdict,
            "explanation": verdict.explanation,
            "evidence": digest(verdict),
        }
        print(
            f"[snippet replay] {index + 1}/30 {verdict.verdict} "
            f"(expected {snippet_predictions[index]})",
            flush=True,
        )

    sm = snippet_metrics
    pm = passage_metrics
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
    mismatches = [
        index + 1
        for index, record in enumerate(passage_records)
        if not record.get("cold_equals_warm", True)
    ]
    lines.append(
        "Cold and warm passage verdicts agreed on all 30 claims."
        if not mismatches
        else f"Cold and warm passage verdicts differed on claims: {mismatches}."
    )
    lines.append("")
    lines.append("## Claims whose verdict changed")
    lines.append("")
    if not changed_indices:
        lines.append("No claim changed verdict between snippet and passage mode.")
    else:
        for index in changed_indices:
            row = claims[index]
            snippet = snippet_explanations[index]
            passage = passage_records[index]
            lines.append(f"### {row['claim']}")
            lines.append("")
            lines.append(f"- Gold: `{row['label']}`")
            lines.append(
                f"- Snippet: `{snippet['verdict']}` — {snippet['explanation']}"
            )
            lines.append(
                f"- Passage: `{passage['verdict']}` — {passage['explanation']}"
            )
            if snippet["evidence"] or passage["evidence"]:
                lines.append("- Snippet evidence:")
                lines.extend(f"  - {item}" for item in snippet["evidence"])
                lines.append("- Passage evidence:")
                lines.extend(f"  - {item}" for item in passage["evidence"])
            lines.append("")
    lines.append("## Correct abstention to wrong verdict")
    lines.append("")
    if regressions:
        for index in regressions:
            row = claims[index]
            record = passage_records[index]
            lines.append(
                f"- {row['claim']} -> `{record['verdict']}`: "
                f"{record['explanation']}"
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
        "cold_warm_mismatches": mismatches,
        "changed": [
            {
                "claim": claims[index]["claim"],
                "gold": claims[index]["label"],
                "snippet": snippet_explanations[index]["verdict"],
                "passage": passage_records[index]["verdict"],
                "snippet_explanation": snippet_explanations[index]["explanation"],
                "passage_explanation": passage_records[index]["explanation"],
            }
            for index in changed_indices
        ],
        "regressions": [
            {
                "claim": claims[index]["claim"],
                "passage": passage_records[index]["verdict"],
            }
            for index in regressions
        ],
    }
    JSON_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"changed={len(changed_indices)} regressions={len(regressions)} "
        f"cold_warm_mismatches={len(mismatches)}",
        flush=True,
    )
    print(json.dumps({"snippet": sm, "passage": pm}, indent=2), flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
