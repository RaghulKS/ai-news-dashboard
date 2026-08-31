"""SQLite persistence layer with deduplication support."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from finintel.config import DB_PATH
from finintel.models.article import Article, ArticleRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    url_hash TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    content TEXT,
    source TEXT NOT NULL,
    source_type TEXT DEFAULT 'newsapi',
    published_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    query TEXT,
    is_duplicate INTEGER DEFAULT 0,
    duplicate_of TEXT,
    sentiment_label TEXT,
    sentiment_score REAL,
    sentiment_confidence REAL,
    event_type TEXT,
    event_confidence REAL,
    cluster_id TEXT,
    novelty_score REAL,
    entities_json TEXT,
    embedding_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_cluster ON articles(cluster_id);
CREATE INDEX IF NOT EXISTS idx_articles_url_hash ON articles(url_hash);

CREATE TABLE IF NOT EXISTS event_clusters (
    cluster_id TEXT PRIMARY KEY,
    representative_title TEXT NOT NULL,
    event_type TEXT NOT NULL,
    article_count INTEGER DEFAULT 0,
    source_count INTEGER DEFAULT 0,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    avg_sentiment_score REAL DEFAULT 0,
    sentiment_consensus TEXT,
    novelty_score REAL DEFAULT 1.0,
    market_relevance REAL DEFAULT 0,
    contradiction_score REAL DEFAULT 0,
    entities_json TEXT,
    tickers_json TEXT,
    summary TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS market_returns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    event_cluster_id TEXT,
    event_time TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    return_pct REAL,
    UNIQUE(ticker, event_cluster_id, horizon_days)
);
"""


class ArticleStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def url_exists(self, url: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM articles WHERE url = ?", (url,)).fetchone()
            return row is not None

    def insert_article(self, article: Article, **extras) -> Optional[int]:
        if self.url_exists(article.url):
            return None
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO articles (
                    url, url_hash, title, description, content, source, source_type,
                    published_at, ingested_at, query, is_duplicate, duplicate_of,
                    sentiment_label, sentiment_score, sentiment_confidence,
                    event_type, event_confidence, cluster_id, novelty_score,
                    entities_json, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.url,
                    article.url_hash,
                    article.title,
                    article.description,
                    article.content,
                    article.source,
                    article.source_type,
                    article.published_at.isoformat(),
                    now,
                    article.query,
                    int(extras.get("is_duplicate", False)),
                    extras.get("duplicate_of"),
                    extras.get("sentiment_label"),
                    extras.get("sentiment_score"),
                    extras.get("sentiment_confidence"),
                    extras.get("event_type"),
                    extras.get("event_confidence"),
                    extras.get("cluster_id"),
                    extras.get("novelty_score"),
                    json.dumps(extras.get("entities")) if extras.get("entities") else None,
                    json.dumps(extras.get("embedding")) if extras.get("embedding") else None,
                ),
            )
            return cur.lastrowid

    def get_recent_articles(self, limit: int = 500) -> list[ArticleRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM articles ORDER BY published_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_articles_by_cluster(self, cluster_id: str) -> list[ArticleRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM articles WHERE cluster_id = ? ORDER BY published_at",
                (cluster_id,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def update_article_analysis(
        self,
        url: str,
        *,
        sentiment_label: str,
        sentiment_score: float,
        sentiment_confidence: float,
        event_type: str,
        event_confidence: float,
        cluster_id: str,
        novelty_score: float,
        entities: list,
        embedding: list[float],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE articles SET
                    sentiment_label = ?, sentiment_score = ?, sentiment_confidence = ?,
                    event_type = ?, event_confidence = ?, cluster_id = ?,
                    novelty_score = ?, entities_json = ?, embedding_json = ?
                WHERE url = ?
                """,
                (
                    sentiment_label,
                    sentiment_score,
                    sentiment_confidence,
                    event_type,
                    event_confidence,
                    cluster_id,
                    novelty_score,
                    json.dumps(entities),
                    json.dumps(embedding),
                    url,
                ),
            )

    def upsert_cluster(self, cluster: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO event_clusters (
                    cluster_id, representative_title, event_type, article_count,
                    source_count, first_seen, last_seen, avg_sentiment_score,
                    sentiment_consensus, novelty_score, market_relevance,
                    contradiction_score, entities_json, tickers_json, summary, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cluster_id) DO UPDATE SET
                    representative_title = excluded.representative_title,
                    article_count = excluded.article_count,
                    source_count = excluded.source_count,
                    last_seen = excluded.last_seen,
                    avg_sentiment_score = excluded.avg_sentiment_score,
                    sentiment_consensus = excluded.sentiment_consensus,
                    novelty_score = excluded.novelty_score,
                    market_relevance = excluded.market_relevance,
                    contradiction_score = excluded.contradiction_score,
                    entities_json = excluded.entities_json,
                    tickers_json = excluded.tickers_json,
                    summary = excluded.summary,
                    metadata_json = excluded.metadata_json
                """,
                (
                    cluster["cluster_id"],
                    cluster["representative_title"],
                    cluster["event_type"],
                    cluster["article_count"],
                    cluster["source_count"],
                    cluster["first_seen"],
                    cluster["last_seen"],
                    cluster["avg_sentiment_score"],
                    cluster["sentiment_consensus"],
                    cluster["novelty_score"],
                    cluster["market_relevance"],
                    cluster["contradiction_score"],
                    json.dumps(cluster.get("entities", [])),
                    json.dumps(cluster.get("tickers", [])),
                    cluster.get("summary"),
                    json.dumps(cluster.get("metadata", {})),
                ),
            )

    def get_clusters(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM event_clusters ORDER BY last_seen DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_title_hashes(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT title FROM articles").fetchall()
        return {r["title"].lower().strip() for r in rows}

    def _row_to_record(self, row: sqlite3.Row) -> ArticleRecord:
        return ArticleRecord(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            content=row["content"],
            url=row["url"],
            source=row["source"],
            published_at=datetime.fromisoformat(row["published_at"]),
            query=row["query"],
            source_type=row["source_type"],
            ingested_at=datetime.fromisoformat(row["ingested_at"]) if row["ingested_at"] else None,
            is_duplicate=bool(row["is_duplicate"]),
            duplicate_of=row["duplicate_of"],
            sentiment_label=row["sentiment_label"],
            sentiment_score=row["sentiment_score"],
            sentiment_confidence=row["sentiment_confidence"],
            event_type=row["event_type"],
            event_confidence=row["event_confidence"],
            cluster_id=row["cluster_id"],
            novelty_score=row["novelty_score"],
            entities_json=row["entities_json"],
        )
