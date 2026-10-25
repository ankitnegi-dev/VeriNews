"""Benchmark six realistic NLI pairs and exercise the pair cache."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.nli as nli
import config

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=config.LOG_LEVEL, format="%(levelname)s %(message)s")

CLAIM = "Actor Tom Skerritt, known for Top Gun and Alien, died at age 93."
SNIPPETS = [
    "Tom Skerritt Dies: Alien and Top Gun Actor Was 93",
    "Tom Skerritt, Alien And Top Gun Actor, Dead At 93",
    'Tom Skerritt of "Top Gun" has died at the age of 93.',
    "Tom Skerritt dead at 93: Actor known for Top Gun and Alien passes away as heartbroken family pay tribute",
    "Tom Skerritt, the 'Top Gun,' 'Alien,' and 'Picket Fences' star and Emmy winner, has died at the age of 93.",
    "Veteran actor Tom Skerritt dies at 93",
]
THRESHOLD_SECONDS = 8.0


def main() -> int:
    cache = nli.NLICache()
    removed = 0
    for snippet in SNIPPETS:
        path = cache.path_for(CLAIM, snippet)
        if path.exists():
            path.unlink()
            removed += 1

    print(f"Model: {config.NLI_MODEL_NAME}")
    print(f"Device: {'cuda' if config.DEVICE >= 0 else 'cpu'}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Cache: {cache.directory} ({removed} benchmark entries removed)")

    started = time.perf_counter()
    nli._get_model()
    load_seconds = time.perf_counter() - started
    print(f"Model load: {load_seconds:.2f}s")

    started = time.perf_counter()
    first = nli.classify(CLAIM, SNIPPETS, cache=cache, batch_size=6)
    first_seconds = time.perf_counter() - started

    started = time.perf_counter()
    second = nli.classify(CLAIM, SNIPPETS, cache=cache, batch_size=6)
    cached_seconds = time.perf_counter() - started

    print()
    print(f"{'#':<2} {'P(entail)':>10} {'P(neutral)':>11} {'P(contra)':>10}  stance")
    for index, (snippet, result) in enumerate(zip(SNIPPETS, first), start=1):
        print(
            f"{index:<2} {result.p_entail:>10.3f} {result.p_neutral:>11.3f} "
            f"{result.p_contra:>10.3f}  {result.stance}"
        )
        print(f"   hit={result.cached} | {snippet[:150]}")

    print()
    print(f"First batch (6 misses + cache writes): {first_seconds:.2f}s")
    print(f"Second batch (6 cache hits): {cached_seconds:.2f}s")
    print(f"Second batch cached flags: {[result.cached for result in second]}")

    if first_seconds > THRESHOLD_SECONDS:
        print(
            f"WARNING: the first batch took {first_seconds:.2f}s, over "
            f"{THRESHOLD_SECONDS:.1f}s. Do not switch models without user approval. "
            f"Fallback model: {config.NLI_FALLBACK_MODEL_NAME}"
        )
    else:
        print(
            f"First batch is under {THRESHOLD_SECONDS:.1f}s; current model stays."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
