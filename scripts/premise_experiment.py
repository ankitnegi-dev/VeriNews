"""Compare three NLI premises on the dev claims only.

Variants:
    A: the current best sentence from ranking
    B: article title + the best sentence
    C: article title + description + content, capped at 350 NLI tokens

Retrieval is cache-only. NLI cache keys include the exact premise text, so the
three variants cannot share a cache entry. Cache entries for the experiment
pairs are cleared first so the batch times are cold-cache times.

Usage:
    python scripts/premise_experiment.py
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from app.claim_analysis import analyze
from app.nli import NLICache, _get_model, classify
from app.ranking import rank_articles
from app.retrieval import retrieve

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

CLAIMS_PATH = PROJECT_ROOT / "tests" / "data" / "claims.csv"
SPLIT = "dev"
VARIANTS = ("A", "B", "C")
PREMISE_TOKEN_CAP = 350
SHORT_CLAIMS = {
    "South African human rights lawyer Navi Pillay won the 2026 Nobel Peace Prize.": "Navi Pillay",
    "Canadian poet Anne Carson won the 2026 Nobel Prize in Literature.": "Anne Carson",
    "Netflix reported third-quarter 2026 revenue of $12.87 billion and earnings of 87 cents per share.": "Netflix Q3",
    "Manchester United beat Como 4-1 in a Champions League match on October 21, 2026.": "Man Utd/Como",
    "SpaceX launched a Dragon cargo spacecraft to the International Space Station from Vandenberg Space Force Base in California on October 13, 2026.": "SpaceX Vandenberg",
    "A magnitude 7.7 earthquake struck southern Panama on October 9, 2026, and the Panama Canal suspended all ship passage.": "Panama quake",
    "Netflix will cut its subscription prices for every customer next week, according to an unnamed insider.": "Netflix price cut",
    "Bitcoin will trade above $250,000 before the end of 2027.": "Bitcoin $250k",
    "Several unnamed engineers at a major chip company are secretly planning to quit next month.": "Chip engineers",
    "An anonymous source says a famous Bollywood actor will announce retirement next week.": "Bollywood retirement",
}


def load_dev_claims() -> list[dict[str, str]]:
    with CLAIMS_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if (row.get("split") or "").strip() == SPLIT]


def truncate_tokens(text: str, tokenizer, cap: int) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= cap:
        return text
    return tokenizer.decode(token_ids[:cap], skip_special_tokens=True)


def build_best_sentence(title: str, snippet: str) -> str:
    if snippet.strip().lower() == title.strip().lower():
        return title
    return f"{title}. {snippet}"


def directional_margin(expected: str, entail: float, contra: float) -> float:
    if expected == "SUPPORTED":
        return entail - contra
    if expected == "REFUTED":
        return contra - entail
    return -(entail + contra)


def directional_win(expected: str, entail: float, contra: float) -> bool:
    if expected == "SUPPORTED":
        return entail > contra
    if expected == "REFUTED":
        return contra > entail
    return entail + contra < 1.0


def main() -> int:
    claims = load_dev_claims()
    if not claims:
        print(f"No {SPLIT} claims found in {CLAIMS_PATH}")
        return 1

    print("Premise experiment: dev only, cached retrieval")
    print(f"NLI model: {config.NLI_MODEL_NAME}")
    print(f"Variant C cap: {PREMISE_TOKEN_CAP} tokens")
    print()

    started = time.perf_counter()
    _model, tokenizer, _labels, _device = _get_model()
    print(f"Model load: {time.perf_counter() - started:.2f}s")

    cache = NLICache()
    records: list[dict[str, object]] = []
    for row in claims:
        claim = " ".join((row.get("claim") or "").split())
        expected = (row.get("label") or "").strip()
        analysis = analyze(claim)
        articles = retrieve(analysis.queries, cache_only=True)
        ranked = rank_articles(claim, articles)
        by_url = {article.url: article for article in articles}

        premises: dict[str, list[str]] = {variant: [] for variant in VARIANTS}
        for item in ranked:
            article = by_url.get(item.url)
            if article is None:
                continue
            premises["A"].append(item.snippet)
            premises["B"].append(build_best_sentence(item.title, item.snippet))
            premises["C"].append(
                truncate_tokens(
                    f"{item.title}. {article.content}",
                    tokenizer,
                    PREMISE_TOKEN_CAP,
                )
            )

        records.append(
            {
                "claim": claim,
                "expected": expected,
                "short": SHORT_CLAIMS.get(claim, claim[:45]),
                "kept": len(ranked),
                "premises": premises,
            }
        )

    # Make every variant batch cold so the timings are comparable.
    for record in records:
        for variant in VARIANTS:
            for premise in record["premises"][variant]:
                cache.path_for(record["claim"], premise).unlink(missing_ok=True)
    print("Cleared the experiment pairs from the NLI cache for cold timing.")
    print()

    results: dict[str, list[dict[str, object]]] = {variant: [] for variant in VARIANTS}
    for variant in VARIANTS:
        for record in records:
            premises = record["premises"][variant]
            started = time.perf_counter()
            nli_results = classify(
                record["claim"],
                premises,
                cache=cache,
                batch_size=config.NLI_BATCH_SIZE,
            )
            elapsed = time.perf_counter() - started
            entail = sum(result.p_entail for result in nli_results)
            contra = sum(result.p_contra for result in nli_results)
            results[variant].append(
                {
                    "entail": entail,
                    "contra": contra,
                    "seconds": elapsed,
                    "win": directional_win(
                        record["expected"], entail, contra
                    ),
                    "margin": directional_margin(
                        record["expected"], entail, contra
                    ),
                }
            )

    header = (
        "claim|expected|kept|"
        "A ent|A con|A s|"
        "B ent|B con|B s|"
        "C ent|C con|C s"
    )
    print(header)
    print("-" * len(header))
    for index, record in enumerate(records):
        cells = [
            str(record["short"]),
            str(record["expected"]),
            str(record["kept"]),
        ]
        for variant in VARIANTS:
            row = results[variant][index]
            cells.extend(
                [
                    f"{row['entail']:.3f}",
                    f"{row['contra']:.3f}",
                    f"{row['seconds']:.2f}",
                ]
            )
        print("|".join(cells))

    print()
    for variant in VARIANTS:
        total_time = sum(row["seconds"] for row in results[variant])
        mean_margin = sum(row["margin"] for row in results[variant]) / len(records)
        wins = sum(1 for row in results[variant] if row["win"])
        print(
            f"{variant}: total={total_time:.2f}s | "
            f"mean directional margin={mean_margin:.3f} | "
            f"directional wins={wins}/{len(records)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
