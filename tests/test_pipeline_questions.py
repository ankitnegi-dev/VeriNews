"""Offline test for the question-input guard."""

from __future__ import annotations

from app.pipeline import VerificationPipeline


def test_question_guard_returns_exact_message_without_running_stages():
    def fail(*args, **kwargs):
        raise AssertionError("stages must not run for a question")

    pipeline = VerificationPipeline(
        analyzer=fail,
        retriever=fail,
        ranker=fail,
        classifier=fail,
    )

    verdict = pipeline.verify("Is this true?")

    assert verdict.verdict == "UNVERIFIED"
    assert verdict.explanation == (
        "This looks like a question. Enter one statement of fact."
    )
    assert verdict.queries == []
    assert verdict.evidence == []
    assert verdict.latency_ms >= 0
