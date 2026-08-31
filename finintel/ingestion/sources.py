"""Multi-source news ingestion."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from xml.etree import ElementTree

import feedparser
import requests

from finintel.config import AppConfig, load_config
from finintel.models.article import Article

logger = logging.getLogger(__name__)


class NewsSource(ABC):
    @abstractmethod
    def fetch(self, query: str, count: int) -> list[Article]:
        ...


class NewsAPISource(NewsSource):
    FINANCIAL_KEYWORDS = [
        "earnings", "revenue", "profit", "guidance", "merger", "acquisition",
        "IPO", "dividend", "analyst", "regulatory", "layoff", "lawsuit",
        "forecast", "stock", "shares", "market",
    ]

    def __init__(self, api_key: str, days_back: int = 7, timeout: int = 30):
        self.api_key = api_key
        self.days_back = days_back
        self.timeout = timeout
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch(self, query: str, count: int) -> list[Article]:
        if not self.api_key:
            logger.warning("NewsAPI key not configured; skipping NewsAPI source")
            return []

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.days_back)
        kw = " OR ".join(f'"{k}"' for k in self.FINANCIAL_KEYWORDS[:12])
        search_q = f'"{query}" AND ({kw})'

        params = {
            "q": search_q,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(count * 2, 100),
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "apiKey": self.api_key,
        }

        try:
            resp = requests.get(self.base_url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.error("NewsAPI request failed: %s", exc)
            return []

        if data.get("status") != "ok":
            logger.error("NewsAPI error: %s", data.get("message"))
            return []

        articles: list[Article] = []
        for raw in data.get("articles", []):
            title = raw.get("title", "")
            if len(title) < 15 or not raw.get("url"):
                continue
            pub = raw.get("publishedAt", "")
            try:
                published = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            except ValueError:
                published = datetime.now(timezone.utc)

            articles.append(
                Article(
                    title=title,
                    description=raw.get("description"),
                    content=raw.get("content"),
                    url=raw["url"],
                    source=raw.get("source", {}).get("name", "unknown"),
                    published_at=published,
                    query=query,
                    source_type="newsapi",
                    raw=raw,
                )
            )
            if len(articles) >= count:
                break
        return articles


class RSSSource(NewsSource):
    def __init__(self, feeds: list[str], timeout: int = 30):
        self.feeds = feeds
        self.timeout = timeout

    def fetch(self, query: str, count: int) -> list[Article]:
        query_lower = query.lower()
        articles: list[Article] = []

        for feed_url in self.feeds:
            try:
                parsed = feedparser.parse(feed_url)
            except Exception as exc:
                logger.warning("RSS parse failed for %s: %s", feed_url, exc)
                continue

            for entry in parsed.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                link = entry.get("link", "")
                if not title or not link:
                    continue

                text_blob = f"{title} {summary}".lower()
                if query_lower not in text_blob and not self._matches_ticker(query, text_blob):
                    continue

                pub = self._parse_date(entry)
                source_name = parsed.feed.get("title", feed_url)
                articles.append(
                    Article(
                        title=title,
                        description=summary[:2000] if summary else None,
                        url=link,
                        source=source_name,
                        published_at=pub,
                        query=query,
                        source_type="rss",
                    )
                )
                if len(articles) >= count:
                    return articles
        return articles

    @staticmethod
    def _parse_date(entry: dict) -> datetime:
        for key in ("published_parsed", "updated_parsed"):
            tp = entry.get(key)
            if tp:
                return datetime(*tp[:6], tzinfo=timezone.utc)
        for key in ("published", "updated"):
            val = entry.get(key)
            if val:
                try:
                    return parsedate_to_datetime(val)
                except (TypeError, ValueError):
                    pass
        return datetime.now(timezone.utc)

    @staticmethod
    def _matches_ticker(query: str, text: str) -> bool:
        q = query.upper().strip()
        if len(q) <= 5 and q.isalpha():
            return f"({q})" in text.upper() or f" {q} " in f" {text.upper()} "
        return False


class IngestionPipeline:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.sources: list[NewsSource] = []

        if self.config.news_api_key:
            self.sources.append(
                NewsAPISource(
                    self.config.news_api_key,
                    days_back=self.config.ingestion.days_back,
                    timeout=self.config.ingestion.request_timeout,
                )
            )
        self.sources.append(
            RSSSource(
                self.config.ingestion.rss_feeds,
                timeout=self.config.ingestion.request_timeout,
            )
        )

    def ingest(self, query: str, count: Optional[int] = None) -> list[Article]:
        target = count or self.config.ingestion.max_articles_per_query
        per_source = max(target // max(len(self.sources), 1), 5)
        seen_urls: set[str] = set()
        merged: list[Article] = []

        for source in self.sources:
            try:
                batch = source.fetch(query, per_source)
            except Exception as exc:
                logger.error("Source %s failed: %s", type(source).__name__, exc)
                continue
            for article in batch:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    merged.append(article)
                if len(merged) >= target:
                    return merged[:target]

        merged.sort(key=lambda a: a.published_at, reverse=True)
        return merged[:target]
