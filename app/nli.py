"""NLI verification with a persistent per-pair cache.

Premise = evidence snippet, hypothesis = claim. The model label order is read
from ``model.config.id2label`` and is never assumed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

import torch

import config
from app.schemas import NLIResult

logger = logging.getLogger(__name__)

_MODEL = None
_TOKENIZER = None
_LABEL_INDICES: tuple[int, int, int] | None = None
_DEVICE = "cpu"


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip()


def cache_key(claim: str, snippet: str) -> str:
    payload = f"{_normalize(claim)}\n{_normalize(snippet)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find_label_index(id2label: dict[int, str], fragment: str) -> int:
    """Find a label by name because model label order is not guaranteed."""
    for index, label in id2label.items():
        if fragment in str(label).lower():
            return int(index)
    raise ValueError(f"No label containing {fragment!r} in {id2label}")


def choose_stance(p_entail: float, p_neutral: float, p_contra: float) -> str:
    if p_entail >= config.NLI_SUPPORT_THRESHOLD:
        return "SUPPORTS"
    if p_contra >= config.NLI_REFUTE_THRESHOLD:
        return "REFUTES"
    return "NEUTRAL"


class NLICache:
    """One JSON file per (claim, snippet) pair, keyed by a SHA-256 hash."""

    def __init__(
        self,
        directory: Path | str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.directory = (
            Path(directory) if directory is not None else config.NLI_CACHE_DIR
        )
        self.model_name = model_name or config.NLI_MODEL_NAME

    def path_for(self, claim: str, snippet: str) -> Path:
        return self.directory / f"nli_{cache_key(claim, snippet)}.json"

    def get(self, claim: str, snippet: str) -> NLIResult | None:
        path = self.path_for(claim, snippet)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError):
            logger.warning("Could not read NLI cache file %s; ignoring it", path)
            return None

        if not isinstance(payload, dict) or payload.get("model_name") != self.model_name:
            return None
        try:
            return NLIResult(
                p_entail=float(payload["p_entail"]),
                p_neutral=float(payload["p_neutral"]),
                p_contra=float(payload["p_contra"]),
                stance=str(payload["stance"]),
                cached=True,
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("NLI cache file %s has an invalid shape; ignoring it", path)
            return None

    def set(self, claim: str, snippet: str, result: NLIResult) -> Path:
        path = self.path_for(claim, snippet)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "claim": _normalize(claim),
            "snippet": _normalize(snippet),
            "p_entail": result.p_entail,
            "p_neutral": result.p_neutral,
            "p_contra": result.p_contra,
            "stance": result.stance,
        }
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path


def _get_model():
    """Load the NLI model and tokenizer once, on first use."""
    global _MODEL, _TOKENIZER, _LABEL_INDICES, _DEVICE
    if _MODEL is None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Run: pip install -r requirements.txt"
            ) from exc

        logger.info("Loading NLI model %s", config.NLI_MODEL_NAME)
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(
                config.NLI_MODEL_NAME
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load NLI model {config.NLI_MODEL_NAME!r}. "
                "Check the model cache or download it with transformers."
            ) from exc

        device = "cuda" if config.DEVICE >= 0 else "cpu"
        model.to(device)
        model.eval()
        id2label = {
            int(index): str(label)
            for index, label in model.config.id2label.items()
        }
        _LABEL_INDICES = (
            find_label_index(id2label, "entail"),
            find_label_index(id2label, "neutral"),
            find_label_index(id2label, "contra"),
        )
        _MODEL = model
        _TOKENIZER = tokenizer
        _DEVICE = device
        logger.info("NLI labels: %s", id2label)

    return _MODEL, _TOKENIZER, _LABEL_INDICES, _DEVICE


def _infer(
    claim: str,
    snippets: Sequence[str],
    batch_size: int,
) -> list[tuple[float, float, float]]:
    model, tokenizer, label_indices, device = _get_model()
    entail_index, neutral_index, contra_index = label_indices
    probabilities: list[tuple[float, float, float]] = []

    for start in range(0, len(snippets), batch_size):
        batch = list(snippets[start : start + batch_size])
        inputs = tokenizer(
            batch,
            [claim] * len(batch),
            padding=True,
            truncation=True,
            max_length=config.NLI_MAX_TOKENS,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        rows = torch.softmax(logits, dim=-1).cpu().tolist()
        for row in rows:
            probabilities.append(
                (
                    float(row[entail_index]),
                    float(row[neutral_index]),
                    float(row[contra_index]),
                )
            )
    return probabilities


def classify(
    claim: str,
    snippets: Sequence[str],
    *,
    use_cache: bool = True,
    cache: NLICache | None = None,
    batch_size: int | None = None,
) -> list[NLIResult]:
    """Return one NLI result per snippet, using the disk cache when enabled."""
    cleaned_claim = _normalize(claim)
    if not cleaned_claim:
        raise ValueError("Claim is empty.")
    cleaned_snippets = [_normalize(snippet) for snippet in snippets]
    if not cleaned_snippets:
        return []

    size = config.NLI_BATCH_SIZE if batch_size is None else batch_size
    if size <= 0:
        raise ValueError("batch_size must be positive.")

    active_cache = (cache or NLICache()) if use_cache else None
    results: list[NLIResult | None] = [None] * len(cleaned_snippets)
    misses: list[tuple[int, str]] = []
    hits = 0

    for index, snippet in enumerate(cleaned_snippets):
        if not snippet:
            results[index] = NLIResult(0.0, 1.0, 0.0, "NEUTRAL", cached=False)
            continue
        cached = active_cache.get(cleaned_claim, snippet) if active_cache else None
        if cached is not None:
            results[index] = cached
            hits += 1
        else:
            misses.append((index, snippet))

    logger.info("NLI cache: %d hits, %d misses", hits, len(misses))
    if misses:
        probabilities = _infer(
            cleaned_claim,
            [snippet for _, snippet in misses],
            size,
        )
        for (index, snippet), (p_entail, p_neutral, p_contra) in zip(
            misses, probabilities
        ):
            result = NLIResult(
                p_entail=p_entail,
                p_neutral=p_neutral,
                p_contra=p_contra,
                stance=choose_stance(p_entail, p_neutral, p_contra),
                cached=False,
            )
            results[index] = result
            if active_cache:
                active_cache.set(cleaned_claim, snippet, result)

    return [
        result
        if result is not None
        else NLIResult(0.0, 1.0, 0.0, "NEUTRAL", cached=False)
        for result in results
    ]
