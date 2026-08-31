"""Tests for semantic clustering."""

from datetime import datetime, timezone

import numpy as np

from finintel.clustering.semantic import SemanticClusterer
from finintel.models.article import Article
from finintel.models.event import EventType, FinancialEvent, SentimentLabel
from finintel.nlp.embeddings import EmbeddingService


def _make_article(title: str, url: str) -> Article:
    return Article(
        title=title,
        description=title,
        url=url,
        source="TestSource",
        published_at=datetime.now(timezone.utc),
    )


def _make_analysis(title: str):
    return {
        "sentiment": {
            "label": SentimentLabel.POSITIVE,
            "score": 0.5,
            "confidence": 0.8,
        },
        "entities": [],
        "event": FinancialEvent(
            event_type=EventType.EARNINGS,
            confidence=0.8,
            headline=title,
        ),
        "novelty_score": 0.9,
        "embedding": [0.1] * 8,
    }


def test_clustering_groups_similar(monkeypatch):
    clusterer = SemanticClusterer(similarity_threshold=0.5)

    def mock_embed(texts):
        vecs = []
        for t in texts:
            if "Apple" in t:
                vecs.append(np.array([1.0, 0.0, 0.0, 0.0]))
            elif "Tesla" in t:
                vecs.append(np.array([0.0, 1.0, 0.0, 0.0]))
            else:
                vecs.append(np.array([0.0, 0.0, 1.0, 0.0]))
        return np.vstack(vecs)

    monkeypatch.setattr(clusterer.embedder, "embed", mock_embed)

    articles = [
        _make_article("Apple beats Q4 earnings", "https://a.com/1"),
        _make_article("Apple reports strong Q4 earnings beat", "https://a.com/2"),
        _make_article("Tesla misses delivery targets", "https://a.com/3"),
    ]
    analyses = [_make_analysis(a.title) for a in articles]
    clusters = clusterer.cluster_articles(articles, analyses)

    assert len(clusters) == 2
    apple_cluster = next(c for c in clusters if c.article_count == 2)
    assert apple_cluster.article_count == 2


def test_cosine_similarity():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert EmbeddingService.cosine_similarity(a, b) == pytest.approx(1.0, abs=0.01)


import pytest  # noqa: E402
