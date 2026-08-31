"""Tests for evaluation metrics."""

from finintel.evaluation.metrics import (
    calibration_error,
    entity_link_metrics,
    event_classification_metrics,
    retrieval_at_k,
    sentiment_metrics,
)


def test_sentiment_metrics():
    y_true = ["positive", "negative", "neutral", "positive"]
    y_pred = ["positive", "negative", "positive", "positive"]
    m = sentiment_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.75
    assert "f1_macro" in m


def test_entity_link_metrics():
    pred = [{"AAPL", "MSFT"}, {"TSLA"}]
    gold = [{"AAPL"}, {"TSLA", "NVDA"}]
    m = entity_link_metrics(pred, gold)
    assert m["precision"] > 0
    assert m["recall"] > 0


def test_event_classification_metrics():
    y_true = ["earnings", "earnings", "m_and_a"]
    y_pred = ["earnings", "guidance", "m_and_a"]
    m = event_classification_metrics(y_true, y_pred)
    assert m["macro_f1"] > 0
    assert "earnings" in m["per_class"]


def test_retrieval_at_k():
    relevant = {"doc1", "doc2"}
    retrieved = ["doc1", "doc3", "doc2"]
    assert retrieval_at_k(relevant, retrieved, k=2) == 0.5


def test_calibration_error():
    y_true = [1, 0, 1, 0, 1]
    y_prob = [0.9, 0.1, 0.8, 0.3, 0.7]
    ece = calibration_error(y_true, y_prob, n_bins=5)
    assert 0 <= ece <= 1
