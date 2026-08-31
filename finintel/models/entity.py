"""Financial entity models."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    COMPANY = "company"
    TICKER = "ticker"
    SECTOR = "sector"
    EXECUTIVE = "executive"
    MACRO = "macro"
    INDEX = "index"
    CURRENCY = "currency"
    REGULATOR = "regulator"
    UNKNOWN = "unknown"


class Entity(BaseModel):
    text: str
    entity_type: EntityType
    start: int = 0
    end: int = 0
    confidence: float = 1.0
    metadata: dict = Field(default_factory=dict)


class LinkedEntity(BaseModel):
    surface_form: str
    canonical_name: str
    entity_type: EntityType
    ticker: Optional[str] = None
    sector: Optional[str] = None
    confidence: float = 1.0
    link_source: str = "rule"
