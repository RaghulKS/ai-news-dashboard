"""Tests for sentiment analysis."""

from finintel.nlp.sentiment import LexiconSentiment


def test_positive_sentiment():
    analyzer = LexiconSentiment()
    result = analyzer.analyze("Stock surges to record high after strong earnings beat")
    assert result["label"].value == "positive"
    assert result["confidence"] > 0.5


def test_negative_sentiment():
    analyzer = LexiconSentiment()
    result = analyzer.analyze("Shares plunge after company misses estimates and announces layoffs")
    assert result["label"].value == "negative"


def test_neutral_sentiment():
    analyzer = LexiconSentiment()
    result = analyzer.analyze("The company held its annual meeting today")
    assert result["label"].value == "neutral"


def test_empty_text():
    analyzer = LexiconSentiment()
    from finintel.nlp.sentiment import FinBERTSentiment
    result = FinBERTSentiment._neutral()
    assert result["label"].value == "neutral"
