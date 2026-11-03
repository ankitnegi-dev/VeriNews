"""Offline tests for full-text extraction and the local full-text cache."""

from __future__ import annotations

from pathlib import Path

import app.fulltext as fulltext

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ARTICLE_HTML = (FIXTURES / "article_river_plan.html").read_text(encoding="utf-8")
URL = "https://example.com/news/river-plan"


def test_extract_text_keeps_article_body_and_drops_chrome():
    text = fulltext.extract_text(ARTICLE_HTML, url=URL)

    assert "city council approved the river restoration plan" in text.lower()
    assert "12 million dollars" in text
    assert "Home | World" not in text
    assert "console.log should not appear" not in text
    assert "Copyright 2026 Example News" not in text
    assert "  " not in text


def test_extract_text_handles_empty_and_broken_html():
    assert fulltext.extract_text("") == ""
    assert fulltext.extract_text("   ") == ""
    assert isinstance(fulltext.extract_text("<html><body><p>", url=URL), str)


def test_cache_round_trip_uses_a_url_hash(tmp_path):
    path = fulltext.write_cached_text(
        URL,
        "Cached article text.",
        directory=tmp_path,
        status_code=200,
    )

    assert path.parent == tmp_path
    assert path.name == f"{fulltext.cache_key(URL)}.json"
    assert fulltext.read_cached_text(URL, directory=tmp_path) == "Cached article text."
    assert fulltext.read_cached_text("https://example.com/other", directory=tmp_path) is None


def test_fetch_full_text_returns_cached_text_without_network(monkeypatch, tmp_path):
    fulltext.write_cached_text(URL, "Already cached.", directory=tmp_path)

    def fail_request(*args, **kwargs):
        raise AssertionError("network must not be used for a cached URL")

    monkeypatch.setattr(fulltext.requests, "get", fail_request)

    assert fulltext.fetch_full_text(URL, directory=tmp_path) == "Already cached."


def test_fetch_full_text_honours_robots(monkeypatch, tmp_path):
    monkeypatch.setattr(fulltext, "_robots_allowed", lambda *args, **kwargs: False)

    def fail_request(*args, **kwargs):
        raise AssertionError("page must not be fetched when robots.txt disallows it")

    monkeypatch.setattr(fulltext.requests, "get", fail_request)

    assert fulltext.fetch_full_text(URL, directory=tmp_path) == ""
    payload = (tmp_path / f"{fulltext.cache_key(URL)}.json").read_text(encoding="utf-8")
    assert "robots_disallowed" in payload
