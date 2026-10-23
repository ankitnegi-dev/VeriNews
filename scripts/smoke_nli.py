"""Phase 1 smoke test: load the NLI model and check three clear cases.

Premise = evidence, hypothesis = claim. A working model should label these
ENTAILMENT, NEUTRAL and CONTRADICTION respectively.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Allow both "python scripts/smoke_nli.py" and "python -m scripts.smoke_nli".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logging.basicConfig(level=config.LOG_LEVEL, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAIRS = [
    (
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower is in Paris.",
        "ENTAILMENT",
    ),
    (
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower was completed in 1889.",
        "NEUTRAL",
    ),
    (
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower is located in Berlin, Germany.",
        "CONTRADICTION",
    ),
]


def find_label_index(id2label: dict[int, str], fragment: str) -> int:
    """Find a label by name because model label order is not guaranteed."""
    for index, label in id2label.items():
        if fragment in label.lower():
            return index
    raise ValueError(f"No label containing {fragment!r} in {id2label}")


def choose_stance(p_entail: float, p_neutral: float, p_contra: float) -> str:
    if p_entail >= config.NLI_SUPPORT_THRESHOLD:
        return "SUPPORTS"
    if p_contra >= config.NLI_REFUTE_THRESHOLD:
        return "REFUTES"
    return "NEUTRAL"


def main() -> None:
    device = "cuda" if config.DEVICE >= 0 else "cpu"
    print(f"Loading {config.NLI_MODEL_NAME} on {device} ...")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL_NAME)
    model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - started
    print(f"Model loaded in {load_seconds:.1f}s")
    print(f"Raw id2label: {model.config.id2label}")

    id2label = {int(index): str(label) for index, label in model.config.id2label.items()}
    entail_index = find_label_index(id2label, "entail")
    neutral_index = find_label_index(id2label, "neutral")
    contra_index = find_label_index(id2label, "contra")

    total_seconds = 0.0
    correct = 0
    print(f"\n{'#':<2} {'expected':<15} {'predicted':<15} {'P(entail)':>10} "
          f"{'P(neutral)':>11} {'P(contra)':>10} {'latency':>9}")
    for number, (premise, hypothesis, expected) in enumerate(PAIRS, start=1):
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=config.NLI_MAX_TOKENS,
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        pair_started = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        elapsed = time.perf_counter() - pair_started
        total_seconds += elapsed

        p_entail = float(probabilities[entail_index])
        p_neutral = float(probabilities[neutral_index])
        p_contra = float(probabilities[contra_index])
        predicted_index = int(torch.argmax(probabilities))
        predicted = id2label[predicted_index].upper()
        correct += int(predicted.startswith(expected[:4]))

        stance = choose_stance(p_entail, p_neutral, p_contra)
        print(
            f"{number:<2} {expected:<15} {predicted:<15} {p_entail:>10.3f} "
            f"{p_neutral:>11.3f} {p_contra:>10.3f} {elapsed:>8.2f}s  -> {stance}"
        )

    average = total_seconds / len(PAIRS)
    print(f"\nCorrect: {correct}/{len(PAIRS)}")
    print(f"Average latency: {average:.2f}s per pair (CPU threads: {torch.get_num_threads()})")
    if average > 3.0:
        print(
            "WARNING: average latency is over 3s per pair. Do not switch models "
            "without user approval; the fallback is "
            f"{config.NLI_FALLBACK_MODEL_NAME}."
        )


if __name__ == "__main__":
    main()
