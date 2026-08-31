"""Article domain models."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field


class Article(BaseModel):
    title: str
    description: Optional[str] = None
    content: Optional[str] = None
    url: str
    source: str
    published_at: datetime
    query: Optional[str] = None
    source_type: str = "newsapi"
    raw: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url_hash(self) -> str:
        return sha256(self.url.encode()).hexdigest()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def full_text(self) -> str:
        parts = [self.title]
        if self.description:
            parts.append(self.description)
        if self.content:
            parts.append(self.content)
        return " ".join(parts)


class ArticleRecord(Article):
    id: Optional[int] = None
    ingested_at: Optional[datetime] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    embedding_id: Optional[str] = None
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_confidence: Optional[float] = None
    event_type: Optional[str] = None
    event_confidence: Optional[float] = None
    cluster_id: Optional[str] = None
    novelty_score: Optional[float] = None
    entities_json: Optional[str] = None
