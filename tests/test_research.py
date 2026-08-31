"""Tests for market research module."""

from datetime import datetime

from finintel.market.research import EventReturnResearch


def test_insufficient_data_disclaimer():
    research = EventReturnResearch(horizons=[1])
    events = [{"event_time": datetime(2024, 1, 1)}]
    results = research.analyze_events(events, "AAPL")
    assert len(results) == 1
    assert not results[0].is_significant
    assert "Insufficient" in results[0].disclaimer


def test_walk_forward_empty():
    research = EventReturnResearch()
    folds = research.walk_forward_validation([], "AAPL")
    assert folds == []
