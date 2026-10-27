"""Offline tests for source credibility lookup and YAML validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.credibility import get_credibility, load_credibility, normalize_domain


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("reuters.com", "reuters.com"),
        ("Reuters.COM", "reuters.com"),
        ("www.reuters.com", "reuters.com"),
        ("https://edition.cnn.com/world?q=1", "edition.cnn.com"),
        ("edition.cnn.com:443", "edition.cnn.com"),
        ("  example.com.  ", "example.com"),
        ("https://user:pass@example.com/path", "example.com"),
        ("", ""),
    ],
)
def test_normalize_domain(raw, expected):
    assert normalize_domain(raw) == expected


def test_reviewed_table_scores_known_and_unknown_domains():
    table = load_credibility()

    assert table.score_for("reuters.com") == 1.0
    assert table.score_for("www.reuters.com") == 1.0
    assert table.score_for("news.bbc.co.uk") == 1.0
    assert table.score_for("economictimes.indiatimes.com") == 0.7
    assert table.score_for("theonion.com") == 0.2
    assert table.score_for("unknown.example") == 0.4
    assert table.score_for("whitehouse.gov") == 1.0
    assert table.score_for("college.nic.in") == 1.0
    assert table.score_for("notreuters.com") == 0.4


TEMP_TABLE = """
version: 1
default: low
scores:
  high: 1.0
  medium: 0.7
  low: 0.4
  blocked: 0.2
domains:
  high:
    - example.com
  medium:
    - news.example.org
  blocked:
    - satire.example
    - example.gov
suffixes:
  high:
    - .gov
    - .gov.in
"""


def _write_table(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "domains.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_exact_subdomain_suffix_and_default_matching(tmp_path):
    table = load_credibility(_write_table(tmp_path, TEMP_TABLE))

    assert table.score_for("m.example.com") == 1.0
    assert table.score_for("news.example.org") == 0.7
    assert table.score_for("sub.news.example.org") == 0.7
    assert table.score_for("satire.example") == 0.2
    assert table.score_for("agency.gov") == 1.0
    assert table.score_for("agency.gov.in") == 1.0
    assert table.score_for("unknown.net") == 0.4


def test_subdomain_matching_requires_a_dot_boundary(tmp_path):
    table = load_credibility(_write_table(tmp_path, TEMP_TABLE))

    assert table.score_for("example.com.evil.net") == 0.4
    assert table.score_for("fakeexample.com") == 0.4
    assert table.score_for("gov.unknown.net") == 0.4


def test_explicit_domain_tier_beats_a_broad_suffix(tmp_path):
    table = load_credibility(_write_table(tmp_path, TEMP_TABLE))

    assert table.score_for("example.gov") == 0.2
    assert table.score_for("sub.example.gov") == 0.2
    assert table.score_for("other.gov") == 1.0


def test_get_credibility_accepts_a_url():
    assert get_credibility("https://www.bbc.com/news/world") == 1.0


def test_invalid_score_is_rejected(tmp_path):
    text = TEMP_TABLE.replace("high: 1.0", "high: 1.5")

    with pytest.raises(ValueError, match="between 0 and 1"):
        load_credibility(_write_table(tmp_path, text))


def test_unknown_default_tier_is_rejected(tmp_path):
    text = TEMP_TABLE.replace("default: low", "default: missing")

    with pytest.raises(ValueError, match="default tier"):
        load_credibility(_write_table(tmp_path, text))


def test_unknown_domain_tier_is_rejected(tmp_path):
    text = TEMP_TABLE.replace("  medium:\n    - news.example.org", "  mystery:\n    - news.example.org")

    with pytest.raises(ValueError, match="unknown tier"):
        load_credibility(_write_table(tmp_path, text))


def test_duplicate_domain_is_rejected(tmp_path):
    text = TEMP_TABLE.replace(
        "  medium:\n    - news.example.org",
        "  medium:\n    - news.example.org\n    - www.example.com",
    )

    with pytest.raises(ValueError, match="more than once"):
        load_credibility(_write_table(tmp_path, text))


def test_missing_file_has_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Credibility file not found"):
        load_credibility(tmp_path / "absent.yaml")
