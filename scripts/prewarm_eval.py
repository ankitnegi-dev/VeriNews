"""Prewarm the retrieval cache for the evaluation claim set.

This script reads ``tests/data/claims.csv`` exactly as supplied. It never edits
the CSV or changes labels. At most 20 claims are attempted per UTC day, and the
existing 80-request daily cap is enforced before and during retrieval.

Progress is stored in ``.cache/prewarm_state.json``, so the script resumes on
the next run. Do not run this script until the user confirms. ``--dry-run``
shows the next batch without making network calls or changing state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from app.claim_analysis import analyze
from app.retrieval import RequestBudget, retrieve

logger = logging.getLogger(__name__)

DEFAULT_CLAIMS_PATH = PROJECT_ROOT / "tests" / "data" / "claims.csv"
DEFAULT_STATE_PATH = config.CACHE_DIR / "prewarm_state.json"
MAX_CLAIMS_PER_DAY = 20
REQUIRED_COLUMNS = {"claim", "label", "split"}
VALID_SPLITS = {"dev", "test"}


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _claim_id(claim: str) -> str:
    return hashlib.sha256(claim.encode("utf-8")).hexdigest()[:16]


def load_claims(path: Path, split: str) -> list[dict[str, str]]:
    """Read claims without modifying them, optionally filtering by split."""
    if not path.exists():
        raise SystemExit(f"Claims file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Claims file is missing columns: {', '.join(sorted(missing))}"
            )

        claims: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            claim = " ".join((row.get("claim") or "").split())
            label = (row.get("label") or "").strip()
            row_split = (row.get("split") or "").strip().lower()
            if not claim:
                logger.warning("Skipping empty claim on row %d", row_number)
                continue
            if row_split not in VALID_SPLITS:
                raise SystemExit(
                    f"Row {row_number} has invalid split {row_split!r}; "
                    "expected dev or test"
                )
            if split != "all" and row_split != split:
                continue

            claim_id = _claim_id(claim)
            if claim_id in seen_ids:
                logger.warning("Skipping duplicate claim on row %d", row_number)
                continue
            seen_ids.add(claim_id)
            claims.append(
                {
                    "id": claim_id,
                    "claim": claim,
                    "label": label,
                    "split": row_split,
                    "row": str(row_number),
                }
            )
    return claims


def load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"version": 1, "completed": {}, "daily_claims": {}}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(
            f"Could not read prewarm state at {path}. Fix it manually; "
            "the script will not reset it because that could exceed the daily cap."
        ) from exc

    if not isinstance(payload, dict):
        raise SystemExit(f"Prewarm state at {path} must be a JSON object.")

    completed = payload.get("completed", {})
    daily_claims = payload.get("daily_claims", {})
    if not isinstance(completed, dict) or not isinstance(daily_claims, dict):
        raise SystemExit(
            f"Prewarm state at {path} has invalid completed/daily_claims fields."
        )
    return {"version": 1, "completed": completed, "daily_claims": daily_claims}


def save_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _remaining_request_budget(budget: RequestBudget) -> tuple[int, int, str]:
    count, limit, day = budget.usage()
    return max(0, limit - count), limit, day


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and cache evidence for the evaluation claims."
    )
    parser.add_argument(
        "--claims-path",
        type=Path,
        default=DEFAULT_CLAIMS_PATH,
        help="CSV file with claim,label,split columns.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Resumable prewarm state file.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "dev", "test"),
        default="all",
        help="Claim split to process. Test evidence is for final evaluation only.",
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=MAX_CLAIMS_PER_DAY,
        help="Claims to attempt in this run (never more than 20 per day).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the next batch without fetching or changing state.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=config.LOG_LEVEL, format="%(levelname)s %(message)s")

    if args.max_claims <= 0:
        raise SystemExit("--max-claims must be positive.")
    run_limit = min(args.max_claims, MAX_CLAIMS_PER_DAY)

    claims = load_claims(args.claims_path, args.split)
    state = load_state(args.state_path)
    completed = state["completed"]
    daily_claims = state["daily_claims"]
    assert isinstance(completed, dict)
    assert isinstance(daily_claims, dict)

    today = _today()
    try:
        already_today = int(daily_claims.get(today, 0))
    except (TypeError, ValueError) as exc:
        raise SystemExit("State file has an invalid daily claim count.") from exc
    remaining_today = max(0, run_limit - already_today)
    pending = [claim for claim in claims if claim["id"] not in completed]
    batch = pending[:remaining_today]

    budget = RequestBudget()
    remaining_requests, request_limit, budget_day = _remaining_request_budget(budget)
    print(
        f"Claims file: {args.claims_path}\n"
        f"Split: {args.split} | total={len(claims)} | pending={len(pending)}\n"
        f"Daily claims: {already_today}/{run_limit} attempted on {today}\n"
        f"Request budget: {request_limit - remaining_requests}/{request_limit} "
        f"used on {budget_day}"
    )

    if not batch:
        if remaining_today <= 0:
            print("Daily claim limit reached. Run again on the next UTC day.")
        else:
            print("No pending claims for this split.")
        return 0

    if args.dry_run:
        print(f"Dry run: would attempt {len(batch)} claim(s):")
        for claim in batch:
            print(f"  row {claim['row']} [{claim['split']}] {claim['claim']}")
        return 0

    if not config.NEWS_API_KEY and not config.GNEWS_API_KEY:
        raise SystemExit(
            "Neither NEWS_API_KEY nor GNEWS_API_KEY is set. "
            "Add a key to .env before running the prewarm script."
        )

    if remaining_requests <= 0:
        print("Daily request cap reached. Run again on the next UTC day.")
        return 0

    attempted = 0
    for claim in batch:
        try:
            queries = analyze(claim["claim"]).queries
        except ValueError as exc:
            logger.warning("Skipping row %s: %s", claim["row"], exc)
            completed[claim["id"]] = {
                "claim": claim["claim"],
                "split": claim["split"],
                "date": today,
                "articles": 0,
                "error": str(exc),
            }
            save_state(args.state_path, state)
            continue

        # Default retrieval can try NewsAPI and then GNews for each query.
        worst_case_requests = len(queries) * 2
        remaining_requests, request_limit, budget_day = _remaining_request_budget(
            budget
        )
        if remaining_requests < worst_case_requests:
            print(
                "Stopping before claim row "
                f"{claim['row']}: need up to {worst_case_requests} requests, "
                f"but only {remaining_requests}/{request_limit} remain on "
                f"{budget_day}. Run again on the next UTC day."
            )
            break

        attempted += 1
        daily_claims[today] = already_today + attempted
        # Save the attempt before fetching, so a crash cannot silently exceed
        # the 20-claims-per-day limit on the next run.
        save_state(args.state_path, state)

        logger.info(
            "Prewarming row %s [%s] with %d queries: %s",
            claim["row"],
            claim["split"],
            len(queries),
            claim["claim"],
        )
        try:
            articles = retrieve(queries, cache_only=False)
        except Exception as exc:
            logger.exception("Retrieval failed for row %s", claim["row"])
            completed[claim["id"]] = {
                "claim": claim["claim"],
                "split": claim["split"],
                "date": today,
                "articles": 0,
                "error": str(exc),
            }
            save_state(args.state_path, state)
            continue

        completed[claim["id"]] = {
            "claim": claim["claim"],
            "split": claim["split"],
            "date": today,
            "articles": len(articles),
        }
        save_state(args.state_path, state)
        print(
            f"[{attempted}/{len(batch)}] row {claim['row']} [{claim['split']}] "
            f"cached {len(articles)} articles"
        )
        if not articles:
            logger.warning(
                "No cached articles for row %s; marked complete with 0 articles",
                claim["row"],
            )

    pending_after = sum(1 for claim in claims if claim["id"] not in completed)
    print(
        f"Attempted {attempted} claim(s) this run. "
        f"Daily claims now {daily_claims.get(today, 0)}/{run_limit}. "
        f"Pending after this run: {pending_after}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
