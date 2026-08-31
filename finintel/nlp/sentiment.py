"""Financial sentiment analysis with FinBERT primary, Azure optional."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

from finintel.config import AppConfig, load_config
from finintel.models.event import SentimentLabel

logger = logging.getLogger(__name__)


class SentimentAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str) -> dict:
        ...


class FinBERTSentiment(SentimentAnalyzer):
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512,
            )

    def analyze(self, text: str) -> dict:
        if not text.strip():
            return self._neutral()
        try:
            self._load()
            result = self._pipeline(text[:1500])[0]
            label = result["label"].lower()
            score = float(result["score"])
            if label == "positive":
                return {
                    "label": SentimentLabel.POSITIVE,
                    "score": score,
                    "confidence": score,
                    "backend": "finbert",
                    "raw_scores": {"positive": score, "negative": 1 - score, "neutral": 0.0},
                }
            if label == "negative":
                return {
                    "label": SentimentLabel.NEGATIVE,
                    "score": -score,
                    "confidence": score,
                    "backend": "finbert",
                    "raw_scores": {"positive": 1 - score, "negative": score, "neutral": 0.0},
                }
            return {
                "label": SentimentLabel.NEUTRAL,
                "score": 0.0,
                "confidence": score,
                "backend": "finbert",
                "raw_scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            }
        except Exception as exc:
            logger.warning("FinBERT failed, using lexicon fallback: %s", exc)
            return LexiconSentiment().analyze(text)

    @staticmethod
    def _neutral() -> dict:
        return {
            "label": SentimentLabel.NEUTRAL,
            "score": 0.0,
            "confidence": 0.5,
            "backend": "neutral_default",
            "raw_scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
        }


class LexiconSentiment(SentimentAnalyzer):
    POSITIVE = {
        "surge", "rally", "beat", "exceed", "growth", "profit", "upgrade", "bullish",
        "record", "strong", "gain", "rise", "outperform", "boost", "soar",
    }
    NEGATIVE = {
        "plunge", "crash", "miss", "decline", "loss", "downgrade", "bearish", "weak",
        "fall", "drop", "lawsuit", "layoff", "fraud", "probe", "warning", "cut",
    }

    def analyze(self, text: str) -> dict:
        words = set(text.lower().split())
        pos = len(words & self.POSITIVE)
        neg = len(words & self.NEGATIVE)
        if pos > neg:
            conf = min(0.5 + 0.1 * (pos - neg), 0.95)
            return {
                "label": SentimentLabel.POSITIVE,
                "score": conf,
                "confidence": conf,
                "backend": "lexicon",
                "raw_scores": {"positive": conf, "negative": 1 - conf, "neutral": 0.0},
            }
        if neg > pos:
            conf = min(0.5 + 0.1 * (neg - pos), 0.95)
            return {
                "label": SentimentLabel.NEGATIVE,
                "score": -conf,
                "confidence": conf,
                "backend": "lexicon",
                "raw_scores": {"positive": 1 - conf, "negative": conf, "neutral": 0.0},
            }
        return FinBERTSentiment._neutral()


class AzureSentimentAdapter(SentimentAnalyzer):
    def __init__(self):
        from azure.azure_sentiment import AzureSentimentAnalyzer
        self._analyzer = AzureSentimentAnalyzer()

    def analyze(self, text: str) -> dict:
        result = self._analyzer.analyze_sentiment(text)
        label_map = {
            "positive": SentimentLabel.POSITIVE,
            "negative": SentimentLabel.NEGATIVE,
            "neutral": SentimentLabel.NEUTRAL,
        }
        label = label_map.get(result["sentiment"], SentimentLabel.NEUTRAL)
        scores = result["confidence_scores"]
        score = scores.get("positive", 0) - scores.get("negative", 0)
        return {
            "label": label,
            "score": score,
            "confidence": max(scores.values()),
            "backend": "azure",
            "raw_scores": scores,
        }


def create_sentiment_analyzer(config: Optional[AppConfig] = None) -> SentimentAnalyzer:
    cfg = config or load_config()
    if cfg.nlp.sentiment_backend == "azure" and cfg.azure_endpoint and cfg.azure_key:
        try:
            return AzureSentimentAdapter()
        except Exception as exc:
            logger.warning("Azure sentiment unavailable: %s", exc)
    if cfg.nlp.sentiment_backend == "finbert":
        return FinBERTSentiment(cfg.nlp.finbert_model)
    if cfg.nlp.use_azure_fallback and cfg.azure_endpoint and cfg.azure_key:
        try:
            return AzureSentimentAdapter()
        except Exception:
            pass
    return LexiconSentiment()
