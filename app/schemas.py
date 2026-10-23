"""Shared data contracts for the VeriNews pipeline.

These dataclasses are the boundary between stages: each stage returns the next
stage's input. Keeping them in one file prevents each module from inventing its
own slightly different shape.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Claim:
    text: str
    entities: list[dict[str, str]]
    queries: list[str]


@dataclass
class Article:
    title: str
    source_domain: str
    url: str
    published_at: str | None
    content: str


@dataclass
class Evidence:
    title: str
    source_domain: str
    url: str
    published_at: str | None
    snippet: str
    similarity: float
    credibility: float
    p_entail: float
    p_neutral: float
    p_contra: float
    stance: str


@dataclass
class Verdict:
    claim: str
    verdict: str
    confidence: float
    support_score: float
    refute_score: float
    total_weight: float
    queries: list[str]
    evidence: list[Evidence]
    explanation: str
    latency_ms: int
