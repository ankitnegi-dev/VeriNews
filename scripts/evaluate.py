"""Evaluate the VeriNews pipeline on a labeled claim split.

The default split is ``dev``. Retrieval is cache-only and no live news request is
made. The script writes ``results/eval_<split>.json`` and
``results/eval_<split>_report.md``.

The test split is only used when it is requested explicitly. Do not run the test
split until the held-out evaluation is scheduled.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --split dev
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from app.decision import decide
from app.pipeline import VerificationPipeline
from app.schemas import Verdict

logger = logging.getLogger(__name__)

DEFAULT_CLAIMS_PATH = PROJECT_ROOT / "tests" / "data" / "claims.csv"
LABELS = ("SUPPORTED", "REFUTED", "UNVERIFIED")
DIRECTIONAL = frozenset({"SUPPORTED", "REFUTED"})
VALID_SPLITS = ("dev", "test")


def load_claims(path: Path, split: str) -> list[dict[str, str]]:
    """Read one split without loading or changing any other split's rows."""
    if split not in VALID_SPLITS:
        raise SystemExit(f"Invalid split {split!r}; expected dev or test.")
    if not path.exists():
        raise SystemExit(f"Claims file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"claim", "label", "split"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Claims file is missing columns: {', '.join(sorted(missing))}"
            )

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            row_split = (row.get("split") or "").strip().lower()
            if row_split != split:
                continue
            claim = " ".join((row.get("claim") or "").split())
            label = (row.get("label") or "").strip().upper()
            if not claim:
                logger.warning("Skipping empty claim on row %d", row_number)
                continue
            if label not in LABELS:
                raise SystemExit(
                    f"Row {row_number} has invalid label {label!r}; "
                    f"expected one of {', '.join(LABELS)}"
                )
            rows.append(
                {
                    "row": str(row_number),
                    "claim": claim,
                    "label": label,
                }
            )

    if not rows:
        raise SystemExit(f"No claims found for split={split} in {path}")
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, object]:
    """Return accuracy, per-class scores, confusion matrix, and safety rates."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")

    matrix = {
        gold: {prediction: 0 for prediction in LABELS}
        for gold in LABELS
    }
    for gold, prediction in zip(expected, predicted):
        if gold not in LABELS or prediction not in LABELS:
            raise ValueError(f"Unknown label pair: {gold!r}, {prediction!r}")
        matrix[gold][prediction] += 1

    total = len(expected)
    correct = sum(matrix[label][label] for label in LABELS)

    per_class: dict[str, dict[str, object]] = {}
    for label in LABELS:
        true_positive = matrix[label][label]
        false_positive = sum(
            matrix[gold][label] for gold in LABELS if gold != label
        )
        false_negative = sum(
            matrix[label][prediction]
            for prediction in LABELS
            if prediction != label
        )
        per_class[label] = {
            "precision": _rate(true_positive, true_positive + false_positive),
            "recall": _rate(true_positive, true_positive + false_negative),
            "support": true_positive + false_negative,
            "predicted": true_positive + false_positive,
        }

    directional_gold = sum(
        matrix[gold][prediction]
        for gold in DIRECTIONAL
        for prediction in LABELS
    )
    committed = sum(
        matrix[gold][prediction]
        for gold in DIRECTIONAL
        for prediction in DIRECTIONAL
    )
    flips = matrix["SUPPORTED"]["REFUTED"] + matrix["REFUTED"]["SUPPORTED"]

    return {
        "total": total,
        "correct": correct,
        "exact_accuracy": _rate(correct, total),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "wrong_verdict_rate": _rate(flips, directional_gold),
        "wrong_verdict_flips": flips,
        "directional_gold": directional_gold,
        "coverage": _rate(committed, directional_gold),
        "committed_directional": committed,
    }


def ablation_verdict(verdict: Verdict) -> Verdict:
    """Re-run only the decision with every evidence credibility set to 1.0."""
    evidence = [
        replace(item, credibility=1.0)
        for item in verdict.evidence
    ]
    return decide(
        verdict.claim,
        evidence,
        queries=verdict.queries,
        latency_ms=verdict.latency_ms,
    )


def run_split(
    claims: Sequence[dict[str, str]],
    pipeline: VerificationPipeline,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for number, row in enumerate(claims, start=1):
        claim = row["claim"]
        started = time.perf_counter()
        verdict = pipeline.verify(claim)
        elapsed = round((time.perf_counter() - started) * 1000)
        ablation = ablation_verdict(verdict)
        records.append(
            {
                "row": row["row"],
                "claim": claim,
                "expected": row["label"],
                "predicted": verdict.verdict,
                "confidence": verdict.confidence,
                "support_score": verdict.support_score,
                "refute_score": verdict.refute_score,
                "total_weight": verdict.total_weight,
                "queries": list(verdict.queries),
                "explanation": verdict.explanation,
                "latency_ms": verdict.latency_ms,
                "wall_ms": elapsed,
                "evidence": [asdict(item) for item in verdict.evidence],
                "ablation_predicted": ablation.verdict,
                "ablation_confidence": ablation.confidence,
                "ablation_explanation": ablation.explanation,
            }
        )
        print(
            f"[{number}/{len(claims)}] expected={row['label']} "
            f"predicted={verdict.verdict} latency={verdict.latency_ms} ms "
            f"{claim[:72]}",
            flush=True,
        )
    return records


def _fmt_rate(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_fraction(value: object, numerator: int, denominator: int) -> str:
    return f"{_fmt_rate(value)} ({numerator}/{denominator})"


def _shorten(text: str, limit: int = 88) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _metric_rows(
    metrics: dict[str, object],
    *,
    mean_latency_ms: int | None = None,
    max_latency_ms: int | None = None,
) -> list[str]:
    lines = [
        f"| Claims | {metrics['total']} |",
        (
            "| Exact accuracy | "
            f"{_fmt_fraction(metrics['exact_accuracy'], metrics['correct'], metrics['total'])} |"
        ),
        (
            "| Wrong-verdict rate (SUPPORTED vs REFUTED flips) | "
            f"{_fmt_fraction(metrics['wrong_verdict_rate'], metrics['wrong_verdict_flips'], metrics['directional_gold'])} |"
        ),
        (
            "| Coverage on directional claims | "
            f"{_fmt_fraction(metrics['coverage'], metrics['committed_directional'], metrics['directional_gold'])} |"
        ),
    ]
    if mean_latency_ms is not None:
        lines.append(f"| Mean latency | {mean_latency_ms} ms |")
    if max_latency_ms is not None:
        lines.append(f"| Max latency | {max_latency_ms} ms |")
    return lines


def _evidence_lines(evidence: Sequence[dict[str, object]]) -> list[str]:
    if not evidence:
        return ["- No evidence was kept."]

    lines: list[str] = []
    for item in evidence:
        title = str(item.get("title") or "")
        url = str(item.get("url") or "")
        link = f"[{title}]({url})" if url else title
        lines.append(
            "  - "
            f"`{item.get('stance')}` | {item.get('source_domain')} | "
            f"relevance {float(item.get('similarity') or 0.0):.3f} | "
            f"credibility {float(item.get('credibility') or 0.0):.2f} | "
            f"P(entail) {float(item.get('p_entail') or 0.0):.3f} | "
            f"P(contra) {float(item.get('p_contra') or 0.0):.3f} | {link}"
        )
    return lines


def build_report(
    *,
    split: str,
    records: Sequence[dict[str, object]],
    metrics: dict[str, object],
    ablation_metrics: dict[str, object],
    mean_latency_ms: int,
    max_latency_ms: int,
    generated_at: str,
) -> str:
    lines = [
        f"# VeriNews evaluation - {split}",
        "",
        f"Generated: {generated_at}",
        "",
        "Retrieval: cache-only. Models were warmed before timing.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        *_metric_rows(
            metrics,
            mean_latency_ms=mean_latency_ms,
            max_latency_ms=max_latency_ms,
        ),
        "",
        "Wrong-verdict rate counts only claims whose expected label is SUPPORTED or REFUTED. Coverage uses the same denominator and counts predictions that committed to either direction.",
        "",
        "## Per-class precision and recall",
        "",
        "| Class | Precision | Recall | Gold support | Predicted |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    per_class = metrics["per_class"]
    assert isinstance(per_class, dict)
    for label in LABELS:
        row = per_class[label]
        assert isinstance(row, dict)
        lines.append(
            f"| {label} | {_fmt_rate(row['precision'])} | "
            f"{_fmt_rate(row['recall'])} | {row['support']} | "
            f"{row['predicted']} |"
        )

    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "Rows are the expected label. Columns are the predicted label.",
            "",
            "| Expected \\ Predicted | SUPPORTED | REFUTED | UNVERIFIED |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    matrix = metrics["confusion_matrix"]
    assert isinstance(matrix, dict)
    for gold in LABELS:
        row = matrix[gold]
        assert isinstance(row, dict)
        lines.append(
            f"| {gold} | {row['SUPPORTED']} | {row['REFUTED']} | "
            f"{row['UNVERIFIED']} |"
        )

    lines.extend(["", "## Mistakes", ""])
    mistakes = [
        record
        for record in records
        if record["expected"] != record["predicted"]
    ]
    if not mistakes:
        lines.append("No mistakes on this split.")
    else:
        for number, record in enumerate(mistakes, start=1):
            lines.extend(
                [
                    f"### {number}. {record['claim']}",
                    "",
                    f"- Expected: `{record['expected']}`",
                    f"- Predicted: `{record['predicted']}`",
                    f"- Confidence: {float(record['confidence']):.3f}",
                    f"- Support score: {float(record['support_score']):.3f}",
                    f"- Refute score: {float(record['refute_score']):.3f}",
                    f"- Total weight: {float(record['total_weight']):.3f}",
                    f"- Latency: {record['latency_ms']} ms",
                    f"- Explanation: {record['explanation']}",
                    "- Evidence:",
                    *_evidence_lines(record["evidence"]),  # type: ignore[arg-type]
                    "",
                ]
            )

    lines.extend(
        [
            "## Credibility ablation",
            "",
            "All evidence credibility scores were set to 1.0 and the decision was "
            "re-run with the same NLI probabilities. Retrieval, ranking, and NLI "
            "were not re-run.",
            "",
            "| Metric | Real tiers | All weights 1.0 |",
            "| --- | ---: | ---: |",
            (
                "| Exact accuracy | "
                f"{_fmt_fraction(metrics['exact_accuracy'], metrics['correct'], metrics['total'])} | "
                f"{_fmt_fraction(ablation_metrics['exact_accuracy'], ablation_metrics['correct'], ablation_metrics['total'])} |"
            ),
            (
                "| Wrong-verdict rate | "
                f"{_fmt_fraction(metrics['wrong_verdict_rate'], metrics['wrong_verdict_flips'], metrics['directional_gold'])} | "
                f"{_fmt_fraction(ablation_metrics['wrong_verdict_rate'], ablation_metrics['wrong_verdict_flips'], ablation_metrics['directional_gold'])} |"
            ),
            (
                "| Coverage | "
                f"{_fmt_fraction(metrics['coverage'], metrics['committed_directional'], metrics['directional_gold'])} | "
                f"{_fmt_fraction(ablation_metrics['coverage'], ablation_metrics['committed_directional'], ablation_metrics['directional_gold'])} |"
            ),
            "",
            "| Claim | Expected | Real tiers | All weights 1.0 | Changed |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for record in records:
        lines.append(
            f"| {_shorten(str(record['claim']))} | {record['expected']} | "
            f"{record['predicted']} | {record['ablation_predicted']} | "
            f"{'yes' if record['predicted'] != record['ablation_predicted'] else 'no'} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate cache-only pipeline verdicts on a labeled split."
    )
    parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="dev",
        help="Claim split to evaluate. Default: dev.",
    )
    parser.add_argument(
        "--claims-path",
        type=Path,
        default=DEFAULT_CLAIMS_PATH,
        help="CSV file with claim, label, and split columns.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=config.RESULTS_DIR,
        help="Directory for the JSON and markdown outputs.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if args.split == "test":
        print(
            "WARNING: evaluating the held-out test split. Use this result only "
            "for the final evaluation."
        )

    claims = load_claims(args.claims_path, args.split)
    pipeline = VerificationPipeline(cache_only=True)
    print(f"Evaluating split={args.split} with {len(claims)} claim(s), cache-only.")
    print("Warming models before timing...")
    pipeline.warmup()

    records = run_split(claims, pipeline)
    expected = [str(record["expected"]) for record in records]
    predicted = [str(record["predicted"]) for record in records]
    ablation_predicted = [
        str(record["ablation_predicted"]) for record in records
    ]
    latencies = [int(record["latency_ms"]) for record in records]

    metrics = compute_metrics(expected, predicted)
    metrics["mean_latency_ms"] = round(statistics.mean(latencies))
    metrics["max_latency_ms"] = max(latencies)
    ablation_metrics = compute_metrics(expected, ablation_predicted)

    mistakes = [
        record
        for record in records
        if record["expected"] != record["predicted"]
    ]
    changed = [
        record
        for record in records
        if record["predicted"] != record["ablation_predicted"]
    ]

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "split": args.split,
        "generated_at": generated_at,
        "claims_path": str(args.claims_path),
        "cache_only": True,
        "model": {
            "nli": config.NLI_MODEL_NAME,
            "embedding": config.EMBEDDING_MODEL_NAME,
            "spacy": config.SPACY_MODEL_NAME,
        },
        "decision_config": {
            "REL_MIN": config.REL_MIN,
            "MIN_TOTAL_WEIGHT": config.MIN_TOTAL_WEIGHT,
            "DECISION_RATIO": config.DECISION_RATIO,
            "MARGIN": config.MARGIN,
            "MIN_EVIDENCE": config.MIN_EVIDENCE,
        },
        "metrics": metrics,
        "mistakes": mistakes,
        "records": records,
        "credibility_ablation": {
            "description": (
                "Evidence credibility set to 1.0; decision re-run with the same "
                "NLI probabilities."
            ),
            "metrics": ablation_metrics,
            "changed_count": len(changed),
            "changed_claims": [
                {
                    "claim": record["claim"],
                    "expected": record["expected"],
                    "real_tiers": record["predicted"],
                    "all_weights_one": record["ablation_predicted"],
                }
                for record in changed
            ],
            "records": [
                {
                    "claim": record["claim"],
                    "expected": record["expected"],
                    "real_tiers": record["predicted"],
                    "all_weights_one": record["ablation_predicted"],
                    "changed": record["predicted"] != record["ablation_predicted"],
                }
                for record in records
            ],
        },
    }

    report = build_report(
        split=args.split,
        records=records,
        metrics=metrics,
        ablation_metrics=ablation_metrics,
        mean_latency_ms=int(metrics["mean_latency_ms"]),
        max_latency_ms=int(metrics["max_latency_ms"]),
        generated_at=generated_at,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / f"eval_{args.split}.json"
    report_path = args.results_dir / f"eval_{args.split}_report.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(report + "\n", encoding="utf-8")

    print()
    print(
        f"Exact accuracy: {metrics['exact_accuracy']:.3f} "
        f"({metrics['correct']}/{metrics['total']})"
    )
    print(
        "Wrong-verdict rate: "
        f"{_fmt_fraction(metrics['wrong_verdict_rate'], metrics['wrong_verdict_flips'], metrics['directional_gold'])}"
    )
    print(
        "Coverage: "
        f"{_fmt_fraction(metrics['coverage'], metrics['committed_directional'], metrics['directional_gold'])}"
    )
    print(f"Mean latency: {metrics['mean_latency_ms']} ms")
    print(f"Max latency: {metrics['max_latency_ms']} ms")
    print(f"Credibility ablation changed {len(changed)} claim(s).")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
