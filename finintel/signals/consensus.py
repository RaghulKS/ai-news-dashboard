"""Source diversity, consensus and contradiction signals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from finintel.models.event import EventCluster, SentimentLabel


@dataclass
class ClusterSignals:
    source_diversity: float
    sentiment_consensus: float
    contradiction_level: float
    event_agreement: float
    overall_confidence: float


class SignalAggregator:
    def compute(self, cluster: EventCluster) -> ClusterSignals:
        source_div = min(cluster.source_count / max(cluster.article_count, 1), 1.0)
        sentiment_consensus = 1.0 - cluster.contradiction_score
        event_agreement = min(cluster.article_count / 5, 1.0) * 0.5 + 0.5
        overall = (
            0.25 * source_div
            + 0.25 * sentiment_consensus
            + 0.25 * (1.0 - cluster.contradiction_score)
            + 0.25 * cluster.market_relevance
        )
        return ClusterSignals(
            source_diversity=source_div,
            sentiment_consensus=sentiment_consensus,
            contradiction_level=cluster.contradiction_score,
            event_agreement=event_agreement,
            overall_confidence=overall,
        )

    def detect_contradictions(self, sentiment_labels: list[SentimentLabel]) -> dict:
        counts = Counter(sentiment_labels)
        total = len(sentiment_labels)
        if total < 2:
            return {"has_contradiction": False, "ratio": 0.0}
        pos = counts.get(SentimentLabel.POSITIVE, 0)
        neg = counts.get(SentimentLabel.NEGATIVE, 0)
        if pos > 0 and neg > 0:
            ratio = min(pos, neg) / total
            return {"has_contradiction": True, "ratio": ratio, "positive": pos, "negative": neg}
        return {"has_contradiction": False, "ratio": 0.0}
