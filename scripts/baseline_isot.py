"""Phase 1 baseline: TF-IDF + logistic regression on the ISOT dataset.

The ISOT "True" articles often begin with a wire-service dateline such as
"WASHINGTON (Reuters) - ". A classifier can learn that dateline instead of
learning news style, so we report the same experiment twice: with the raw text
and with only that leading dateline removed.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Allow both "python scripts/baseline_isot.py" and "python -m scripts.baseline_isot".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logging.basicConfig(level=config.LOG_LEVEL, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_NAMES = ["Fake", "True"]

# Matches "WASHINGTON (Reuters) - ", "(Reuters) - ", "NEW YORK (AP) - ", etc.
# The optional leading group is the dateline city. Kept intentionally simple.
SOURCE_TAG_RE = re.compile(
    r"^\s*(?:[A-Za-z][A-Za-z .'\/-]{0,40}\s*)?"
    r"\((?:Reuters|AP|AFP)\)\s*[-\u2013\u2014]\s*",
    re.IGNORECASE,
)


def validate_isot_files() -> None:
    """Fail early with a clear message if the supplied files are missing/empty."""
    for path in (config.ISOT_FAKE_PATH, config.ISOT_TRUE_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"ISOT file not found: {path}. Put Fake.csv and True.csv in {config.ISOT_DIR}."
            )
        if path.stat().st_size == 0:
            raise ValueError(f"ISOT file is empty: {path}. Replace the 0-byte placeholder.")


def load_isot() -> tuple[pd.DataFrame, dict[str, int]]:
    """Load both ISOT classes, label Fake=0 / True=1, and drop empty text rows."""
    validate_isot_files()

    frames: list[pd.DataFrame] = []
    raw_counts: dict[str, int] = {}
    empty_counts: dict[str, int] = {}
    for path, label, name in (
        (config.ISOT_FAKE_PATH, 0, "fake"),
        (config.ISOT_TRUE_PATH, 1, "true"),
    ):
        frame = pd.read_csv(path, usecols=["title", "text"])
        raw_counts[name] = len(frame)
        frame = frame.dropna(subset=["text"]).copy()
        frame["text"] = frame["text"].astype(str).str.strip()
        before = len(frame)
        frame = frame[frame["text"].str.len() > 0]
        empty_counts[name] = before - len(frame)
        frame["label"] = label
        frames.append(frame[["title", "text", "label"]])
        logger.info(
            "loaded %s rows from %s (%s empty bodies dropped)",
            len(frame),
            path.name,
            empty_counts[name],
        )

    dataset = pd.concat(frames, ignore_index=True)
    logger.info(
        "ISOT total=%s fake=%s true=%s",
        len(dataset),
        int((dataset["label"] == 0).sum()),
        int((dataset["label"] == 1).sum()),
    )
    stats = {
        "fake_rows_raw": raw_counts["fake"],
        "true_rows_raw": raw_counts["true"],
        "rows_dropped_empty_text": empty_counts["fake"] + empty_counts["true"],
    }
    return dataset, stats


def strip_source_tag(text: str) -> str:
    """Remove one leading wire-service dateline, if present."""
    return SOURCE_TAG_RE.sub("", text, count=1)


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays into JSON-safe Python values."""
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def evaluate_variant(
    dataset: pd.DataFrame,
    text_column: str,
    train_index: pd.Index,
    test_index: pd.Index,
    variant_name: str,
) -> dict[str, Any]:
    """Train and score one text variant on the shared train/test split."""
    vectorizer = TfidfVectorizer(
        max_features=config.BASELINE_TFIDF_MAX_FEATURES,
        stop_words=config.BASELINE_TFIDF_STOP_WORDS,
    )
    x_train = vectorizer.fit_transform(dataset.loc[train_index, text_column])
    x_test = vectorizer.transform(dataset.loc[test_index, text_column])

    classifier = LogisticRegression(
        max_iter=config.BASELINE_LOGREG_MAX_ITER,
        random_state=config.BASELINE_RANDOM_STATE,
    )
    classifier.fit(x_train, dataset.loc[train_index, "label"])
    predictions = classifier.predict(x_test)

    y_test = dataset.loc[test_index, "label"]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )
    per_class = precision_score(
        y_test, predictions, labels=[0, 1], average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_test, predictions, labels=[0, 1], average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        y_test, predictions, labels=[0, 1], average=None, zero_division=0
    )

    result = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "per_class": {
            name: {
                "precision": float(per_class[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int((y_test == index).sum()),
            }
            for index, name in enumerate(TARGET_NAMES)
        },
        "confusion_matrix": {
            "labels": TARGET_NAMES,
            "rows_actual_columns_predicted": matrix.tolist(),
        },
        "classification_report": report,
        "test_rows": int(len(test_index)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
    }
    if variant_name == "stripped":
        result["source_tags_removed"] = int(
            dataset.loc[test_index, "text"]
            .apply(lambda text: bool(SOURCE_TAG_RE.match(text)))
            .sum()
        )
    return _jsonable(result)


def main() -> None:
    dataset, load_stats = load_isot()
    rows_before_dedupe = len(dataset)
    duplicate_text_rows = int(dataset.duplicated(subset=["text"]).sum())
    dataset = dataset.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    logger.info(
        "deduplicated %s rows by article text: %s -> %s",
        duplicate_text_rows,
        rows_before_dedupe,
        len(dataset),
    )

    dataset["text_as_is"] = dataset["text"]
    dataset["text_stripped"] = dataset["text"].apply(strip_source_tag)

    tagged_rows = int(dataset["text"].apply(lambda text: bool(SOURCE_TAG_RE.match(text))).sum())
    logger.info("rows with a leading wire-service dateline: %s", tagged_rows)

    train_index, test_index = train_test_split(
        dataset.index,
        test_size=config.BASELINE_TEST_SIZE,
        stratify=dataset["label"],
        random_state=config.BASELINE_RANDOM_STATE,
    )

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "fake_rows_raw": load_stats["fake_rows_raw"],
            "true_rows_raw": load_stats["true_rows_raw"],
            "rows_dropped_empty_text": load_stats["rows_dropped_empty_text"],
            "rows_before_dedupe": rows_before_dedupe,
            "duplicate_text_rows_removed": duplicate_text_rows,
            "fake_rows": int((dataset["label"] == 0).sum()),
            "true_rows": int((dataset["label"] == 1).sum()),
            "total_rows": int(len(dataset)),
            "rows_with_leading_source_tag": tagged_rows,
        },
        "protocol": {
            "test_size": config.BASELINE_TEST_SIZE,
            "random_state": config.BASELINE_RANDOM_STATE,
            "vectorizer": {
                "type": "TfidfVectorizer",
                "max_features": config.BASELINE_TFIDF_MAX_FEATURES,
                "stop_words": config.BASELINE_TFIDF_STOP_WORDS,
            },
            "classifier": {
                "type": "LogisticRegression",
                "max_iter": config.BASELINE_LOGREG_MAX_ITER,
            },
            "text_field": "text",
            "labels": {"Fake": 0, "True": 1},
            "dedupe": {"subset": "text", "keep": "first", "when": "before split"},
        },
        "as_is": evaluate_variant(
            dataset, "text_as_is", train_index, test_index, "as_is"
        ),
        "stripped": evaluate_variant(
            dataset, "text_stripped", train_index, test_index, "stripped"
        ),
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.BASELINE_METRICS_PATH.write_text(
        json.dumps(_jsonable(results), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nBaseline summary (test split)")
    print(f"{'variant':<10} {'accuracy':>9} {'macro F1':>9} {'confusion matrix':>24}")
    for name in ("as_is", "stripped"):
        metrics = results[name]
        print(
            f"{name:<10} {metrics['accuracy']:>9.4f} {metrics['f1_macro']:>9.4f}"
            f" {str(metrics['confusion_matrix']['rows_actual_columns_predicted']):>24}"
        )
    print(f"\nSaved metrics to {config.BASELINE_METRICS_PATH}")


if __name__ == "__main__":
    main()
