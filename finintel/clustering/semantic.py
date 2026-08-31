"""Semantic story clustering and novelty detection."""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

import numpy as np

from finintel.config import AppConfig, load_config
from finintel.models.article import Article
from finintel.models.entity import LinkedEntity
from finintel.models.event import EventCluster, EventType, SentimentLabel
from finintel.nlp.embeddings import EmbeddingService
from finintel.nlp.summarizer import GroundedSummarizer


class SemanticClusterer:
    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        config: Optional[AppConfig] = None,
    ):
        cfg = config or load_config()
        self.threshold = similarity_threshold or cfg.nlp.cluster_similarity_threshold
        self.embedder = EmbeddingService(config=cfg)
        self.summarizer = GroundedSummarizer()

    def cluster_articles(
        self,
        articles: list[Article],
        analyses: list[dict],
    ) -> list[EventCluster]:
        if not articles:
            return []

        texts = [a.full_text for a in articles]
        embeddings = self.embedder.embed(texts)
        n = len(articles)
        assigned = [-1] * n
        cluster_id = 0

        for i in range(n):
            if assigned[i] >= 0:
                continue
            assigned[i] = cluster_id
            for j in range(i + 1, n):
                if assigned[j] >= 0:
                    continue
                sim = EmbeddingService.cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self.threshold:
                    assigned[j] = cluster_id
            cluster_id += 1

        groups: dict[int, list[int]] = defaultdict(list)
        for idx, cid in enumerate(assigned):
            groups[cid].append(idx)

        clusters: list[EventCluster] = []
        for cid, indices in groups.items():
            cluster_articles = [articles[i] for i in indices]
            cluster_analyses = [analyses[i] for i in indices]
            clusters.append(self._build_cluster(cluster_articles, cluster_analyses, embeddings[indices]))
        clusters.sort(key=lambda c: c.last_seen, reverse=True)
        return clusters

    def _build_cluster(
        self,
        articles: list[Article],
        analyses: list[dict],
        embeddings: np.ndarray,
    ) -> EventCluster:
        event_types = [a["event"].event_type for a in analyses]
        dominant_event = Counter(event_types).most_common(1)[0][0]

        sentiments = [a["sentiment"]["label"] for a in analyses]
        scores = [a["sentiment"]["score"] for a in analyses]
        sentiment_consensus = Counter(sentiments).most_common(1)[0][0]
        avg_score = float(np.mean(scores)) if scores else 0.0

        all_entities: list[LinkedEntity] = []
        for a in analyses:
            all_entities.extend(a["entities"])
        unique_entities = self._dedupe_entities(all_entities)
        tickers = list({e.ticker for e in unique_entities if e.ticker})

        sources = list({a.source for a in articles})
        pub_times = [a.published_at for a in articles]
        centroid = embeddings.mean(axis=0).tolist()

        contradiction = self._contradiction_score(analyses)
        novelty = self._novelty_score(analyses)
        relevance = self._market_relevance(dominant_event, len(articles), len(sources), tickers)

        summary = None
        if len(articles) > 1:
            summary = self.summarizer.summarize([a.full_text for a in articles])

        return EventCluster(
            cluster_id=str(uuid.uuid4())[:12],
            representative_title=articles[0].title,
            event_type=dominant_event,
            article_count=len(articles),
            source_count=len(sources),
            sources=sources,
            first_seen=min(pub_times),
            last_seen=max(pub_times),
            avg_sentiment_score=avg_score,
            sentiment_consensus=sentiment_consensus,
            novelty_score=novelty,
            market_relevance=relevance,
            contradiction_score=contradiction,
            entities=unique_entities,
            tickers=tickers,
            article_urls=[a.url for a in articles],
            summary=summary,
            embedding_centroid=centroid,
        )

    @staticmethod
    def _dedupe_entities(entities: list[LinkedEntity]) -> list[LinkedEntity]:
        seen: set[str] = set()
        result: list[LinkedEntity] = []
        for e in entities:
            key = f"{e.entity_type}:{e.canonical_name}"
            if key not in seen:
                seen.add(key)
                result.append(e)
        return result

    @staticmethod
    def _contradiction_score(analyses: list[dict]) -> float:
        labels = [a["sentiment"]["label"] for a in analyses]
        if len(labels) < 2:
            return 0.0
        counts = Counter(labels)
        dominant = counts.most_common(1)[0][1]
        return 1.0 - (dominant / len(labels))

    @staticmethod
    def _novelty_score(analyses: list[dict]) -> float:
        scores = [a.get("novelty_score", 1.0) for a in analyses]
        return float(np.mean(scores)) if scores else 1.0

    @staticmethod
    def _market_relevance(
        event_type: EventType,
        article_count: int,
        source_count: int,
        tickers: list[str],
    ) -> float:
        type_weights = {
            EventType.EARNINGS: 0.9,
            EventType.GUIDANCE: 0.85,
            EventType.MA: 0.95,
            EventType.REGULATORY: 0.8,
            EventType.MACRO: 0.85,
            EventType.ANALYST: 0.7,
            EventType.LAYOFFS: 0.75,
            EventType.LITIGATION: 0.7,
            EventType.PRODUCT: 0.6,
            EventType.DIVIDEND: 0.65,
            EventType.IPO: 0.8,
            EventType.OTHER: 0.3,
        }
        base = type_weights.get(event_type, 0.3)
        coverage = min(article_count / 10, 1.0) * 0.2
        diversity = min(source_count / 5, 1.0) * 0.15
        ticker_bonus = 0.1 if tickers else 0.0
        return min(base + coverage + diversity + ticker_bonus, 1.0)


class NoveltyDetector:
    """Compare new articles against historical embedding centroids."""

    def __init__(self, embedder: Optional[EmbeddingService] = None):
        self.embedder = embedder or EmbeddingService()
        self._historical_centroids: list[np.ndarray] = []

    def load_historical(self, centroids: list[list[float]]) -> None:
        self._historical_centroids = [np.array(c) for c in centroids]

    def score(self, text: str) -> float:
        if not self._historical_centroids:
            return 1.0
        vec = self.embedder.embed_single(text)
        max_sim = max(
            EmbeddingService.cosine_similarity(vec, h) for h in self._historical_centroids
        )
        return max(0.0, 1.0 - max_sim)
