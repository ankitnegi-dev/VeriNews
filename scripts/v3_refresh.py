"""Refresh the number-free first query for each development claim (component A).

Only the new number-free query is fetched. Results are merged into the normal
retrieval cache, so the later ablation can run cache-only. Claims are processed
oldest-first because older events leave the NewsAPI window first; progress is
saved after every claim so the run can resume.

Usage:
    python scripts/v3_refresh.py
    python scripts/v3_refresh.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from app.claim_analysis import analyze
from app.claim_aware import number_free_query
from app.retrieval import DailyLimitError, retrieve

CLAIMS_PATH = PROJECT_ROOT / "tests" / "data" / "claims.csv"
STATE_PATH = config.V3_REFRESH_STATE_PATH


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"completed_rows": [], "requests_before": None, "entries": []}
    except (OSError, ValueError, TypeError):
        return {"completed_rows": [], "requests_before": None, "entries": []}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _request_count() -> int:
    try:
        payload = json.loads(config.REQUEST_COUNT_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    return int(payload.get("count", 0))


def _sorted_claims() -> list[dict]:
    rows = list(csv.DictReader(CLAIMS_PATH.open(encoding="utf-8", newline="")))
    def key(row):
        try:
            return datetime.fromisoformat(row["event_date"]).date()
        except (ValueError, TypeError):
            return datetime.max.date()
    rows.sort(key=key)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state = _load_state()
    completed = set(state.get("completed_rows", []))
    if state.get("requests_before") is None:
        state["requests_before"] = _request_count()
        _save_state(state)

    claims = _sorted_claims()
    print(f"claims: {len(claims)}; already refreshed: {len(completed)}")
    print(f"requests before: {state['requests_before']}; now: {_request_count()}")

    for row in claims:
        row_number = row["_row"] if "_row" in row else None
    # csv has no row number column; use index in original file order (2-based)
    all_rows = list(csv.DictReader(CLAIMS_PATH.open(encoding="utf-8", newline="")))
    # map claim text -> original row number
    row_number_by_claim = {r["claim"]: i + 2 for i, r in enumerate(all_rows)}

    for claim_row in claims:
        claim = claim_row["claim"]
        row_number = row_number_by_claim[claim]
        if row_number in completed:
            print(f"row {row_number}: skip (already refreshed)")
            continue

        analysis = analyze(claim)
        query = number_free_query(claim)
        if not query:
            query = analysis.queries[0] if analysis.queries else claim
        print(f"row {row_number}: event_date={claim_row['event_date']} query={query!r}")

        if args.dry_run:
            completed.add(row_number)
            state["completed_rows"] = sorted(completed)
            _save_state(state)
            continue

        try:
            articles = retrieve([query], cache_only=False, force=True)
        except DailyLimitError as exc:
            print(f"Daily limit reached: {exc}")
            _save_state(state)
            print(f"requests used so far: {_request_count() - state['requests_before']}")
            return 0
        except Exception as exc:
            print(f"row {row_number}: retrieval failed: {exc}")
            _save_state(state)
            continue

        entry = {
            "row": row_number,
            "event_date": claim_row["event_date"],
            "query": query,
            "articles": len(articles),
            "domains": sorted({a.source_domain for a in articles}),
        }
        state.setdefault("entries", []).append(entry)
        completed.add(row_number)
        state["completed_rows"] = sorted(completed)
        _save_state(state)
        print(f"   -> {len(articles)} articles")
        time.sleep(0.5)

    used = _request_count() - state["requests_before"]
    print(f"Refresh complete. Requests used this run: {used}; total today: {_request_count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
