"""Tune guard (i) term coverage X on the dev set using B+C configuration."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from app.pipeline import VerificationPipeline
from app.v3 import V3Options

CLAIMS_PATH = PROJECT_ROOT / "tests" / "data" / "claims.csv"


def metrics(preds, claims):
    exact = sum(1 for p, c in zip(preds, claims) if p == c["label"])
    sr = [c for c in claims if c["label"] in ("SUPPORTED", "REFUTED")]
    correct = sum(1 for p, c in zip(preds, claims) if c["label"] in ("SUPPORTED", "REFUTED") and p == c["label"])
    abst = sum(1 for p, c in zip(preds, claims) if c["label"] in ("SUPPORTED", "REFUTED") and p == "UNVERIFIED")
    flips = sum(1 for p, c in zip(preds, claims) if c["label"] in ("SUPPORTED", "REFUTED") and p in ("SUPPORTED", "REFUTED") and p != c["label"])
    return {"exact": f"{exact}/{len(claims)}", "correct_sr": f"{correct}/{len(sr)}", "abstained_sr": f"{abst}/{len(sr)}", "flips": flips}


def main():
    claims = list(csv.DictReader(CLAIMS_PATH.open(encoding="utf-8", newline="")))
    for i, row in enumerate(claims, start=2):
        row["_row"] = i
    for x in (0.4, 0.5, 0.6, 0.7, 0.8):
        opts = V3Options(passages=True, guards=True, guard_term_coverage=x)
        pipe = VerificationPipeline(cache_only=True, v3=opts)
        preds = []
        for claim in claims:
            v = pipe.verify(claim["claim"])
            preds.append(v.verdict)
        m = metrics(preds, claims)
        print(f"X={x:.2f}: exact={m['exact']} correct_sr={m['correct_sr']} abstained_sr={m['abstained_sr']} flips={m['flips']}", flush=True)


if __name__ == "__main__":
    main()
