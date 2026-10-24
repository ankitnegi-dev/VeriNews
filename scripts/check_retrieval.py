"""Run one live claim through retrieval twice and verify the cache.

The second run must not use any live requests. The script never prints API
keys. It fails clearly if .env has no key or if the claim returns no articles.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow both "python scripts/check_retrieval.py" and
# "python -m scripts.check_retrieval".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from app.claim_analysis import analyze
from app.retrieval import RequestBudget, retrieve


DEFAULT_CLAIM = "Apple announced a new iPhone at an event in Cupertino."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claim",
        default=DEFAULT_CLAIM,
        help="Claim to verify through retrieval.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if config.CACHE_ONLY:
        print(
            "CACHE_ONLY=true is set. Set CACHE_ONLY=false in .env before "
            "running a live check.",
            file=sys.stderr,
        )
        return 2

    if not (config.NEWS_API_KEY.strip() or config.GNEWS_API_KEY.strip()):
        print(
            "No API key found in .env. Add NEWS_API_KEY=... or "
            "GNEWS_API_KEY=... to .env before running a live check.",
            file=sys.stderr,
        )
        return 2

    claim = analyze(args.claim)
    budget = RequestBudget()
    before = budget.usage()

    print(f"Claim: {claim.text}")
    for index, query in enumerate(claim.queries, start=1):
        print(f"Query {index}: {query}")

    first = retrieve(claim.queries)
    after_first = budget.usage()
    second = retrieve(claim.queries)
    after_second = budget.usage()

    first_requests = after_first[0] - before[0]
    second_requests = after_second[0] - after_first[0]

    print(
        f"First run: {len(first)} articles, {first_requests} live requests "
        f"(counter {after_first[0]}/{after_first[1]} on {after_first[2]})"
    )
    print(
        f"Second run: {len(second)} articles, {second_requests} live requests "
        f"(counter {after_second[0]}/{after_second[1]} on {after_second[2]})"
    )

    for index, article in enumerate(first, start=1):
        print(f"  {index}. [{article.source_domain}] {article.title}")
        print(f"     {article.url}")

    if not first:
        print("No articles returned. Check the provider warnings above.", file=sys.stderr)
        return 1
    if second_requests != 0:
        print("Cache check failed: the second run made a live request.", file=sys.stderr)
        return 1
    if [article.url for article in first] != [article.url for article in second]:
        print("Cache check failed: the two runs returned different articles.", file=sys.stderr)
        return 1

    print("Cache check passed: the second run was served from .cache/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
