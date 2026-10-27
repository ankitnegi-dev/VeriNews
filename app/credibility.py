"""Load and query source credibility tiers.

The table is deliberately data-driven: ``credibility/domains.yaml`` groups
domains by tier, and unknown domains fall back to the low tier. Matching is
case-insensitive and accepts either a bare domain or a full URL. A listed domain
also covers its subdomains, so ``edition.cnn.com`` inherits ``cnn.com``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml

import config


def normalize_domain(value: str) -> str:
    """Return a lowercase hostname without ``www.``, a port, or a path."""
    cleaned = (value or "").strip().lower()
    if not cleaned:
        return ""

    if "://" in cleaned:
        host = urlsplit(cleaned).hostname or ""
    elif "/" in cleaned or ":" in cleaned:
        host = urlsplit("//" + cleaned).hostname or ""
    else:
        host = cleaned

    host = host.strip().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _normalize_suffix(value: str) -> str:
    suffix = str(value).strip().lower()
    if not suffix:
        return ""
    if not suffix.startswith("."):
        suffix = "." + suffix
    return suffix.rstrip(".")


@dataclass(frozen=True)
class CredibilityTable:
    """Tier scores plus the domain and suffix rules that select them."""

    scores: dict[str, float]
    default_tier: str
    domain_tiers: dict[str, str]
    suffix_tiers: tuple[tuple[str, str], ...]

    @property
    def default_score(self) -> float:
        return self.scores[self.default_tier]

    def tier_for(self, source: str) -> str:
        host = normalize_domain(source)
        if not host:
            return self.default_tier

        tier = self.domain_tiers.get(host)
        if tier is not None:
            return tier

        # Walk parent domains so subdomains inherit a listed domain.
        parts = host.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[index:])
            tier = self.domain_tiers.get(parent)
            if tier is not None:
                return tier

        for suffix, suffix_tier in self.suffix_tiers:
            if host == suffix[1:] or host.endswith(suffix):
                return suffix_tier

        return self.default_tier

    def score_for(self, source: str) -> float:
        return self.scores[self.tier_for(source)]


def _load_scores(payload: dict) -> dict[str, float]:
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict) or not raw_scores:
        raise ValueError("credibility file must define a non-empty scores mapping")

    scores: dict[str, float] = {}
    for raw_tier, raw_score in raw_scores.items():
        tier = str(raw_tier).strip().lower()
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score for tier {tier!r} is not a number") from exc
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score for tier {tier!r} must be between 0 and 1")
        scores[tier] = score
    return scores


def _load_domain_tiers(
    payload: dict,
    scores: dict[str, float],
) -> dict[str, str]:
    raw_domains = payload.get("domains") or {}
    if not isinstance(raw_domains, dict):
        raise ValueError("credibility file field 'domains' must be a mapping")

    domain_tiers: dict[str, str] = {}
    for raw_tier, raw_values in raw_domains.items():
        tier = str(raw_tier).strip().lower()
        if tier not in scores:
            raise ValueError(f"domain list uses unknown tier {tier!r}")
        if not isinstance(raw_values, list):
            raise ValueError(f"domains for tier {tier!r} must be a list")

        for raw_domain in raw_values:
            domain = normalize_domain(str(raw_domain))
            if not domain:
                continue
            if domain in domain_tiers:
                raise ValueError(f"domain {domain!r} is listed more than once")
            domain_tiers[domain] = tier
    return domain_tiers


def _load_suffix_tiers(
    payload: dict,
    scores: dict[str, float],
) -> tuple[tuple[str, str], ...]:
    raw_suffixes = payload.get("suffixes") or {}
    if not isinstance(raw_suffixes, dict):
        raise ValueError("credibility file field 'suffixes' must be a mapping")

    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_tier, raw_values in raw_suffixes.items():
        tier = str(raw_tier).strip().lower()
        if tier not in scores:
            raise ValueError(f"suffix list uses unknown tier {tier!r}")
        if not isinstance(raw_values, list):
            raise ValueError(f"suffixes for tier {tier!r} must be a list")

        for raw_suffix in raw_values:
            suffix = _normalize_suffix(str(raw_suffix))
            if not suffix:
                continue
            if suffix in seen:
                raise ValueError(f"suffix {suffix!r} is listed more than once")
            seen.add(suffix)
            records.append((suffix, tier))

    # Longest suffix first so .gov.in wins over a hypothetical .in rule.
    records.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple(records)


def load_credibility(path: Path | str | None = None) -> CredibilityTable:
    """Read and validate a credibility YAML file."""
    file_path = Path(path) if path is not None else config.CREDIBILITY_PATH
    try:
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Credibility file not found: {file_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse credibility file {file_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("credibility file must contain a mapping at the top level")

    scores = _load_scores(payload)
    default_tier = str(payload.get("default", "")).strip().lower()
    if default_tier not in scores:
        raise ValueError(f"default tier {default_tier!r} is not in scores")

    return CredibilityTable(
        scores=scores,
        default_tier=default_tier,
        domain_tiers=_load_domain_tiers(payload, scores),
        suffix_tiers=_load_suffix_tiers(payload, scores),
    )


def get_credibility(
    source: str,
    table: CredibilityTable | None = None,
) -> float:
    """Return the credibility score for a domain or article URL."""
    active_table = table or load_credibility()
    return active_table.score_for(source)
