"""Offline API tests with a fake pipeline."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import Evidence, Verdict


def _verdict(claim: str, verdict: str = "SUPPORTED") -> Verdict:
    return Verdict(
        claim=claim,
        verdict=verdict,
        confidence=0.91,
        support_score=2.4,
        refute_score=0.1,
        total_weight=2.5,
        queries=["example query"],
        evidence=[
            Evidence(
                title="Example article",
                source_domain="reuters.com",
                url="https://www.reuters.com/example",
                published_at="2026-10-24T12:00:00Z",
                snippet="The article supports the claim.",
                similarity=0.93,
                credibility=1.0,
                p_entail=0.95,
                p_neutral=0.04,
                p_contra=0.01,
                stance="SUPPORTS",
            )
        ],
        explanation="SUPPORTED: one example source supports the claim.",
        latency_ms=7,
    )


class FakePipeline:
    def __init__(self, *, verdict: Verdict | None = None, error: Exception | None = None):
        self.verdict = verdict
        self.error = error
        self.calls: list[str] = []
        self.warmed_up = False

    def warmup(self) -> None:
        self.warmed_up = True

    def verify(self, claim_text: str) -> Verdict:
        self.calls.append(claim_text)
        if self.error is not None:
            raise self.error
        return self.verdict or _verdict(claim_text)


def _client(pipeline: FakePipeline, *, warmup: bool = False) -> TestClient:
    return TestClient(create_app(pipeline=pipeline, warmup=warmup))


def test_valid_claim_returns_verdict_json():
    pipeline = FakePipeline()

    with _client(pipeline) as client:
        response = client.post("/verify", json={"claim": "  A valid claim.  "})

    assert response.status_code == 200
    body = response.json()
    assert body["claim"] == "A valid claim."
    assert body["verdict"] == "SUPPORTED"
    assert body["confidence"] == 0.91
    assert body["evidence"][0]["title"] == "Example article"
    assert body["evidence"][0]["stance"] == "SUPPORTS"
    assert body["latency_ms"] == 7
    assert pipeline.calls == ["A valid claim."]


def test_empty_claim_is_rejected():
    pipeline = FakePipeline()

    with _client(pipeline) as client:
        empty = client.post("/verify", json={"claim": ""})
        whitespace = client.post("/verify", json={"claim": "   "})
        missing = client.post("/verify", json={})

    assert empty.status_code == 422
    assert whitespace.status_code == 422
    assert missing.status_code == 422
    assert pipeline.calls == []


def test_too_long_claim_is_rejected():
    pipeline = FakePipeline()

    with _client(pipeline) as client:
        response = client.post("/verify", json={"claim": "x" * 501})

    assert response.status_code == 422
    assert pipeline.calls == []


def test_health_endpoint():
    pipeline = FakePipeline()

    with _client(pipeline) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "cache_only" in body
    assert body["nli_model"]


def test_unexpected_pipeline_error_becomes_unverified_not_500():
    pipeline = FakePipeline(error=RuntimeError("boom"))

    with _client(pipeline) as client:
        response = client.post("/verify", json={"claim": "A valid claim."})

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "UNVERIFIED"
    assert "Unexpected pipeline failure" in body["explanation"]
    assert body["confidence"] == 0.0


def test_static_page_is_served_at_root():
    pipeline = FakePipeline()

    with _client(pipeline) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "VeriNews" in response.text


def test_startup_warmup_runs_when_enabled():
    pipeline = FakePipeline()

    with _client(pipeline, warmup=True):
        pass

    assert pipeline.warmed_up is True
