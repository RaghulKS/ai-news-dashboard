"""Tests for event extraction."""

from finintel.models.event import EventType
from finintel.nlp.event_extractor import EventExtractor


def test_earnings_extraction():
    extractor = EventExtractor()
    event = extractor.extract(
        "Apple reported quarterly earnings that beat analyst estimates",
        headline="Apple Q4 earnings beat",
    )
    assert event.event_type == EventType.EARNINGS
    assert event.confidence > 0.5


def test_ma_extraction():
    extractor = EventExtractor()
    event = extractor.extract(
        "Microsoft announces acquisition of gaming studio for $69 billion",
        headline="Microsoft acquisition deal",
    )
    assert event.event_type == EventType.MA


def test_layoffs_extraction():
    extractor = EventExtractor()
    event = extractor.extract(
        "Company announces layoffs affecting 10,000 workers",
        headline="Tech layoffs continue",
    )
    assert event.event_type == EventType.LAYOFFS


def test_macro_extraction():
    extractor = EventExtractor()
    event = extractor.extract(
        "Federal Reserve raises interest rates by 25 basis points",
        headline="Fed rate hike",
    )
    assert event.event_type == EventType.MACRO


def test_analyst_extraction():
    extractor = EventExtractor()
    event = extractor.extract(
        "Goldman Sachs upgrades Apple to buy with price target of $200",
        headline="Analyst upgrade",
    )
    assert event.event_type == EventType.ANALYST
