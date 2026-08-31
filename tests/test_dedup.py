"""Tests for deduplication."""

from datetime import datetime, timezone

from finintel.models.article import Article
from finintel.storage.dedup import Deduplicator, normalize_title, title_similarity


def _article(title: str, url: str) -> Article:
    return Article(
        title=title,
        url=url,
        source="Test",
        published_at=datetime.now(timezone.utc),
    )


def test_normalize_title():
    assert normalize_title("Apple, Inc. Reports!") == "apple inc reports"


def test_title_similarity():
    assert title_similarity(
        "Apple beats Q4 earnings estimates",
        "Apple beats Q4 earnings estimate",
    ) > 0.9


def test_url_dedup():
    dedup = Deduplicator()
    a1 = _article("Apple earnings beat", "https://a.com/1")
    a2 = _article("Different title", "https://a.com/1")
    is_dup, _ = dedup.is_duplicate(a1)
    assert not is_dup
    is_dup2, _ = dedup.is_duplicate(a2)
    assert is_dup2


def test_near_duplicate_title():
    dedup = Deduplicator(similarity_threshold=0.92)
    a1 = _article("Tesla stock surges after earnings beat", "https://a.com/1")
    a2 = _article("Tesla stock surges after earnings beats", "https://a.com/2")
    dedup.is_duplicate(a1)
    is_dup, _ = dedup.is_duplicate(a2)
    assert is_dup
