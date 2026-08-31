"""End-to-end financial event intelligence pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from finintel.clustering.semantic import NoveltyDetector, SemanticClusterer
from finintel.config import AppConfig, load_config
from finintel.ingestion.sources import IngestionPipeline
from finintel.models.event import EventCluster
from finintel.nlp.embeddings import EmbeddingService
from finintel.nlp.entity_linker import EntityLinker
from finintel.nlp.event_extractor import EventExtractor
from finintel.nlp.sentiment import create_sentiment_analyzer
from finintel.nlp.summarizer import GroundedSummarizer
from finintel.signals.consensus import SignalAggregator
from finintel.storage.database import ArticleStore
from finintel.storage.dedup import Deduplicator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    query: str
    articles_fetched: int
    articles_stored: int
    duplicates_skipped: int
    clusters: list[EventCluster] = field(default_factory=list)
    analyses: list[dict] = field(default_factory=list)


class FinancialEventPipeline:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.ingestion = IngestionPipeline(self.config)
        self.store = ArticleStore()
        self.dedup = Deduplicator()
        self.sentiment = create_sentiment_analyzer(self.config)
        self.entity_linker = EntityLinker()
        self.event_extractor = EventExtractor()
        self.embedder = EmbeddingService(config=self.config)
        self.clusterer = SemanticClusterer(config=self.config)
        self.novelty = NoveltyDetector(self.embedder)
        self.signals = SignalAggregator()
        self.summarizer = GroundedSummarizer()

        existing_urls = {a.url for a in self.store.get_recent_articles(2000)}
        existing_titles = self.store.get_title_hashes()
        self.dedup.load_existing(existing_urls, existing_titles)

        historical = self.store.get_clusters(100)
        centroids = []
        for c in historical:
            meta = c.get("metadata_json")
            if meta:
                import json
                m = json.loads(meta) if isinstance(meta, str) else meta
                if m.get("centroid"):
                    centroids.append(m["centroid"])
        self.novelty.load_historical(centroids)

    def run(self, query: str, count: Optional[int] = None) -> PipelineResult:
        articles = self.ingestion.ingest(query, count)
        stored = 0
        skipped = 0
        analyses: list[dict] = []
        kept_articles = []

        for article in articles:
            is_dup, dup_ref = self.dedup.is_duplicate(article)
            if is_dup:
                skipped += 1
                continue

            text = article.full_text
            sentiment = self.sentiment.analyze(text)
            entities = self.entity_linker.extract_and_link(text, query=query)
            event = self.event_extractor.extract(text, entities=entities, headline=article.title)
            novelty_score = self.novelty.score(text)
            embedding = self.embedder.embed_single(text).tolist()

            analysis = {
                "article": article,
                "sentiment": sentiment,
                "entities": entities,
                "event": event,
                "novelty_score": novelty_score,
                "embedding": embedding,
            }
            analyses.append(analysis)
            kept_articles.append(article)

            row_id = self.store.insert_article(
                article,
                sentiment_label=sentiment["label"].value,
                sentiment_score=sentiment["score"],
                sentiment_confidence=sentiment["confidence"],
                event_type=event.event_type.value,
                event_confidence=event.confidence,
                novelty_score=novelty_score,
                entities=[e.model_dump() for e in entities],
                embedding=embedding,
            )
            if row_id:
                stored += 1

        clusters = self.clusterer.cluster_articles(kept_articles, analyses)

        for cluster in clusters:
            signals = self.signals.compute(cluster)
            self.store.upsert_cluster({
                "cluster_id": cluster.cluster_id,
                "representative_title": cluster.representative_title,
                "event_type": cluster.event_type.value,
                "article_count": cluster.article_count,
                "source_count": cluster.source_count,
                "first_seen": cluster.first_seen.isoformat(),
                "last_seen": cluster.last_seen.isoformat(),
                "avg_sentiment_score": cluster.avg_sentiment_score,
                "sentiment_consensus": cluster.sentiment_consensus.value,
                "novelty_score": cluster.novelty_score,
                "market_relevance": cluster.market_relevance,
                "contradiction_score": cluster.contradiction_score,
                "entities": [e.model_dump() for e in cluster.entities],
                "tickers": cluster.tickers,
                "summary": cluster.summary,
                "metadata": {
                    "signals": signals.__dict__,
                    "centroid": cluster.embedding_centroid,
                    "sources": cluster.sources,
                },
            })

            for url in cluster.article_urls:
                idx = next(
                    (i for i, a in enumerate(kept_articles) if a.url == url),
                    None,
                )
                if idx is not None:
                    a = analyses[idx]
                    self.store.update_article_analysis(
                        url,
                        sentiment_label=a["sentiment"]["label"].value,
                        sentiment_score=a["sentiment"]["score"],
                        sentiment_confidence=a["sentiment"]["confidence"],
                        event_type=a["event"].event_type.value,
                        event_confidence=a["event"].confidence,
                        cluster_id=cluster.cluster_id,
                        novelty_score=a["novelty_score"],
                        entities=[e.model_dump() for e in a["entities"]],
                        embedding=a["embedding"],
                    )

        return PipelineResult(
            query=query,
            articles_fetched=len(articles),
            articles_stored=stored,
            duplicates_skipped=skipped,
            clusters=clusters,
            analyses=analyses,
        )
