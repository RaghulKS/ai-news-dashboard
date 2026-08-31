"""Tests for domain models."""

from datetime import datetime, timezone

from finintel.models.article import Article
from finintel.models.entity import EntityType, LinkedEntity
from finintel.models.event import EventType, FinancialEvent


def test_article_url_hash():
    article = Article(
        title="Apple beats earnings estimates",
        url="https://example.com/a",
        source="Reuters",
        published_at=datetime.now(timezone.utc),
    )
    assert len(article.url_hash) == 64


def test_article_full_text():
    article = Article(
        title="Title",
        description="Desc",
        content="Content",
        url="https://example.com/b",
        source="Bloomberg",
        published_at=datetime.now(timezone.utc),
    )
    assert "Title" in article.full_text
    assert "Content" in article.full_text


def test_linked_entity():
    entity = LinkedEntity(
        surface_form="AAPL",
        canonical_name="Apple Inc.",
        entity_type=EntityType.TICKER,
        ticker="AAPL",
        sector="Technology",
    )
    assert entity.ticker == "AAPL"


def test_financial_event():
    event = FinancialEvent(
        event_type=EventType.EARNINGS,
        confidence=0.9,
        headline="Apple reports Q4 earnings beat",
    )
    assert event.event_type == EventType.EARNINGS
