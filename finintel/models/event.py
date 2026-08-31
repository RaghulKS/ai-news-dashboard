"""Financial event and cluster models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from finintel.models.entity import LinkedEntity


class EventType(str, Enum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    MA = "m_and_a"
    REGULATORY = "regulatory"
    PRODUCT = "product"
    LAYOFFS = "layoffs"
    LITIGATION = "litigation"
    MACRO = "macro"
    ANALYST = "analyst"
    DIVIDEND = "dividend"
    IPO = "ipo"
    OTHER = "other"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class FinancialEvent(BaseModel):
    event_type: EventType
    confidence: float
    headline: str
    evidence: str = ""
    entities: list[LinkedEntity] = Field(default_factory=list)
    sentiment: Optional[SentimentLabel] = None
    sentiment_score: Optional[float] = None
    article_url: Optional[str] = None
    published_at: Optional[datetime] = None


class EventCluster(BaseModel):
    cluster_id: str
    representative_title: str
    event_type: EventType
    article_count: int
    source_count: int
    sources: list[str] = Field(default_factory=list)
    first_seen: datetime
    last_seen: datetime
    avg_sentiment_score: float = 0.0
    sentiment_consensus: SentimentLabel = SentimentLabel.NEUTRAL
    novelty_score: float = 1.0
    market_relevance: float = 0.0
    contradiction_score: float = 0.0
    entities: list[LinkedEntity] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    article_urls: list[str] = Field(default_factory=list)
    summary: Optional[str] = None
    embedding_centroid: Optional[list[float]] = None
