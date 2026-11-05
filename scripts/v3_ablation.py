"""Run the v3 ablation over the 30 development claims.

Cache-only: no live requests. Produces notes/v3_ablation.json and a markdown
table. Configurations: v1, A, B, C, A+B, B+C, A+B+C.
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

import config
from app.pipeline import VerificationPipeline
from app.schemas import Verdict
from app.v3 import V3Options

CLAIMS_PATH = PROJECT_ROOT / "tests" / "data" / "claims.csv"
OUT_JSON = PROJECT_ROOT / "notes" / "v3_ablation.json"
OUT_MD = PROJECT_ROOT / "notes" / "v3_ablation.md"

CONFIGS = [
    ("v1", V3Options()),
    ("A", V3Options(retrieval=True)),
    ("B", V3Options(passages=True)),
    ("C", V3Options(guards=True)),
    ("A+B", V3Options(retrieval=True, passages=True)),
    ("B+C", V3Options(passages=True, guards=True)),
    ("A+B+C", V3Options(retrieval=True, passages=True, guards=True)),
]


def load_claims():
    rows = list(csv.DictReader(CLAIMS_PATH.open(encoding="utf-8", newline="")))
    for i, row in enumerate(rows, start=2):
        row["_row"] = i
    return rows


def summarize(values):
    if not values:
        return {"mean": 0.0, "max": 0}
    return {"mean": round(statistics.mean(values), 1), "max": max(values)}


def metrics_for(predictions, claims):
    total = len(claims)
    exact = 0
    correct_sr = 0
    abstained_sr = 0
    flips = 0
    false_positive = 0
    for claim, pred in zip(claims, predictions):
        expected = claim["label"]
        if pred == expected:
            exact += 1
        if expected in ("SUPPORTED", "REFUTED"):
            if pred == expected:
                correct_sr += 1
            elif pred == "UNVERIFIED":
                abstained_sr += 1
            elif pred in ("SUPPORTED", "REFUTED"):
                flips += 1
        else:
            if pred in ("SUPPORTED", "REFUTED"):
                false_positive += 1
    sr_total = sum(1 for c in claims if c["label"] in ("SUPPORTED", "REFUTED"))
    return {
        "n": total,
        "exact": f"{exact}/{total}",
        "correct_sr": f"{correct_sr}/{sr_total}",
        "abstained_sr": f"{abstained_sr}/{sr_total}",
        "flips": flips,
        "false_positive_on_unverified": false_positive,
        "exact_count": exact,
        "correct_sr_count": correct_sr,
        "abstained_sr_count": abstained_sr,
        "flip_count": flips,
    }


def run_config(name, options, claims):
    pipe = VerificationPipeline(cache_only=True, v3=options)
    # cold pass
    cold_latencies = []
    cold_preds = []
    cold_verdicts = []
    guard_agg = None
    clause_info = {}
    for claim in claims:
        v = pipe.verify(claim["claim"])
        cold_latencies.append(v.latency_ms)
        cold_preds.append(v.verdict)
        cold_verdicts.append(v)
        if pipe.last_v3 is not None:
            if guard_agg is None:
                guard_agg = {
                    "guard1_changed": 0,
                    "guard1_refute_to_neutral": 0,
                    "guard1_kept": 0,
                    "guard2_changed": 0,
                    "guard2_neutral_to_support": 0,
                    "guard2_weak_refute_to_support": 0,
                    "guard2_kept": 0,
                    "passages_seen": 0,
                }
            for k in guard_agg:
                guard_agg[k] += pipe.last_v3.guard_stats.as_dict().get(k, 0)
            if pipe.last_v3.clause_result is not None:
                clause_info[claim["_row"]] = pipe.last_v3.clause_result.as_dict()
    # warm pass
    warm_latencies = []
    warm_preds = []
    warm_verdicts = []
    for claim in claims:
        v = pipe.verify(claim["claim"])
        warm_latencies.append(v.latency_ms)
        warm_preds.append(v.verdict)
        warm_verdicts.append(v)

    guard_stats = guard_agg
    # clause info only for passage configs; already collected cold pass
    if not (options.passages or options.guards):
        clause_info = {}

    return {
        "name": name,
        "options": {
            "retrieval": options.retrieval,
            "passages": options.passages,
            "guards": options.guards,
        },
        "cold_latencies": cold_latencies,
        "warm_latencies": warm_latencies,
        "cold_predictions": cold_preds,
        "warm_predictions": warm_preds,
        "cold_verdicts": [v.as_dict() if hasattr(v, "as_dict") else None for v in []],
        "metrics_cold": metrics_for(cold_preds, claims),
        "metrics_warm": metrics_for(warm_preds, claims),
        "guard_stats": guard_stats,
        "clause_info": clause_info,
        "predictions": warm_preds,
    }


def main():
    claims = load_claims()
    results = {}
    for name, options in CONFIGS:
        print(f"Running {name}...", flush=True)
        started = time.perf_counter()
        res = run_config(name, options, claims)
        print(f"  done in {time.perf_counter()-started:.1f}s", flush=True)
        results[name] = res

    # v1 baseline predictions (warm pass)
    v1_preds = results["v1"]["warm_predictions"]

    # changed claims analysis
    changed = []
    for idx, claim in enumerate(claims):
        row = claim["_row"]
        changes = []
        for name, _ in CONFIGS:
            if name == "v1":
                continue
            pred = results[name]["warm_predictions"][idx]
            if pred != v1_preds[idx]:
                changes.append((name, pred))
        if changes:
            changed.append(
                {
                    "row": row,
                    "claim": claim["claim"],
                    "expected": claim["label"],
                    "v1": v1_preds[idx],
                    "changes": changes,
                }
            )

    # regressions: correct abstention or correct verdict became wrong
    regressions = []
    for idx, claim in enumerate(claims):
        row = claim["_row"]
        v1_pred = v1_preds[idx]
        expected = claim["label"]
        v1_correct = v1_pred == expected
        v1_abstention = v1_pred == "UNVERIFIED" and expected in ("SUPPORTED", "REFUTED")
        for name, _ in CONFIGS:
            if name == "v1":
                continue
            new_pred = results[name]["warm_predictions"][idx]
            if (v1_correct or v1_abstention) and new_pred != expected and new_pred in ("SUPPORTED", "REFUTED", "UNVERIFIED"):
                # became wrong: was correct/abstention, now wrong directional or wrong abstention?
                if v1_correct and new_pred != expected:
                    regressions.append(
                        {
                            "row": row,
                            "config": name,
                            "expected": expected,
                            "v1": v1_pred,
                            "new": new_pred,
                            "type": "correct_verdict_lost",
                        }
                    )
                elif v1_abstention and new_pred in ("SUPPORTED", "REFUTED") and new_pred != expected:
                    regressions.append(
                        {
                            "row": row,
                            "config": name,
                            "expected": expected,
                            "v1": v1_pred,
                            "new": new_pred,
                            "type": "abstention_became_wrong_verdict",
                        }
                    )

    # dedupe regressions
    seen = set()
    uniq = []
    for r in regressions:
        key = (r["row"], r["config"], r["type"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    regressions = uniq

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": "development evidence, not held-out",
        "claims": len(claims),
        "configs": {
            name: {
                "options": results[name]["options"],
                "metrics_cold": results[name]["metrics_cold"],
                "metrics_warm": results[name]["metrics_warm"],
                "cold_latency": summarize(results[name]["cold_latencies"]),
                "warm_latency": summarize(results[name]["warm_latencies"]),
                "guard_stats": results[name]["guard_stats"],
                "predictions": results[name]["predictions"],
            }
            for name, _ in CONFIGS
        },
        "changed_claims": changed,
        "regressions": regressions,
        "clause_info": results["A+B+C"]["clause_info"],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown
    lines = []
    lines.append("# v3 claim-aware ablation")
    lines.append("")
    lines.append('**Label: development evidence, not held-out.**')
    lines.append("")
    lines.append("Development set = 30 CSV claims only. Live claims excluded from headline numbers.")
    lines.append("")
    lines.append("## Ablation table (warm pass)")
    lines.append("")
    lines.append("| Config | Exact | Correct on S/R | Abstained on S/R | Flips | Cold latency mean/max (ms) | Warm latency mean/max (ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, _ in CONFIGS:
        r = results[name]
        mc = r["metrics_warm"]
        cold = summarize(r["cold_latencies"])
        warm = summarize(r["warm_latencies"])
        lines.append(
            f"| {name} | {mc['exact']} | {mc['correct_sr']} | {mc['abstained_sr']} | {mc['flips']} | "
            f"{cold['mean']}/{cold['max']} | {warm['mean']}/{warm['max']} |"
        )
    lines.append("")
    lines.append("## Guard change counts (summed over 30 claims)")
    lines.append("")
    for name in ("C", "B+C", "A+B+C"):
        gs = results[name]["guard_stats"]
        if gs:
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"- passages seen: {gs['passages_seen']}")
            lines.append(f"- guard (i) contradictions removed: {gs['guard1_changed']} (refute->neutral); kept: {gs['guard1_kept']}")
            lines.append(f"- guard (ii) neutral promoted to support: {gs['guard2_changed']} (all neutral->support; weak-refute->support: {gs['guard2_weak_refute_to_support']})")
            lines.append("")
    lines.append("## Claims whose verdict changed vs v1")
    lines.append("")
    if not changed:
        lines.append("None.")
    else:
        for c in changed:
            lines.append(f"- Row {c['row']}: {c['claim'][:90]}")
            lines.append(f"  - expected: {c['expected']}; v1: {c['v1']}")
            for name, pred in c["changes"]:
                lines.append(f"  - {name}: {pred}")
    lines.append("")
    lines.append("## Regressions (correct verdict or correct abstention became wrong)")
    lines.append("")
    if not regressions:
        lines.append("None.")
    else:
        for r in regressions:
            lines.append(f"- Row {r['row']} ({r['config']}): {r['type']}; expected {r['expected']}, v1 {r['v1']}, new {r['new']}")
    lines.append("")
    lines.append("## Clause info (row 21)")
    lines.append("")
    ci = results["A+B+C"]["clause_info"]
    if ci:
        for row, info in ci.items():
            lines.append(f"- Row {row}: {json.dumps(info)}")
    else:
        lines.append("No coordinated claims detected.")
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
