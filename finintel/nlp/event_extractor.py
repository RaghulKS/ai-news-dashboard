"""Rule-based financial event extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from finintel.models.entity import LinkedEntity
from finintel.models.event import EventType, FinancialEvent


@dataclass
class EventPattern:
    event_type: EventType
    patterns: list[str]
    weight: float = 1.0


EVENT_PATTERNS: list[EventPattern] = [
    EventPattern(EventType.EARNINGS, [
        r"\b(earnings|eps|quarterly results|q[1-4]\s+results|beats?\s+estimates?|misses?\s+estimates?)\b",
        r"\b(reported|announced)\s+(revenue|profit|loss)\b",
    ], 1.0),
    EventPattern(EventType.GUIDANCE, [
        r"\b(guidance|outlook|forecast|raises?\s+guidance|cuts?\s+guidance|full[- ]year\s+outlook)\b",
    ], 0.95),
    EventPattern(EventType.MA, [
        r"\b(merger|acquisition|acquires?|takeover|deal worth|buyout)\b",
        r"\b(m&a|merges?\s+with)\b",
        r"\bto buy\b.{0,30}\b(for|at)\s+\$",
    ], 1.0),
    EventPattern(EventType.REGULATORY, [
        r"\b(sec|fda|ftc|doj|regulator|regulatory|investigation|probe|antitrust|fine|penalty)\b",
        r"\b(approval|cleared by|subpoena)\b",
    ], 0.9),
    EventPattern(EventType.PRODUCT, [
        r"\b(launches?|unveils?|introduces?|new product|product line|rollout)\b",
    ], 0.85),
    EventPattern(EventType.LAYOFFS, [
        r"\b(layoffs?|job cuts?|workforce reduction|restructuring|headcount)\b",
    ], 0.95),
    EventPattern(EventType.LITIGATION, [
        r"\b(lawsuit|sued|litigation|settlement|class action|legal action)\b",
    ], 0.9),
    EventPattern(EventType.MACRO, [
        r"\b(fed|federal reserve|interest rate|cpi|inflation|gdp|jobs report|unemployment)\b",
        r"\b(central bank|monetary policy|rate (hike|cut))\b",
    ], 0.9),
    EventPattern(EventType.ANALYST, [
        r"\b(upgrades?|downgrades?|price target|analyst|overweight|underweight|buy rating|sell rating)\b",
    ], 0.85),
    EventPattern(EventType.DIVIDEND, [
        r"\b(dividend|share buyback|stock repurchase)\b",
    ], 0.8),
    EventPattern(EventType.IPO, [
        r"\b(ipo|initial public offering|goes public|listing)\b",
    ], 0.9),
]


class EventExtractor:
    def extract(
        self,
        text: str,
        entities: Optional[list[LinkedEntity]] = None,
        headline: str = "",
    ) -> FinancialEvent:
        combined = f"{headline} {text}".strip()
        best_type = EventType.OTHER
        best_score = 0.0
        evidence = ""

        for ep in EVENT_PATTERNS:
            for pattern in ep.patterns:
                m = re.search(pattern, combined, re.IGNORECASE)
                if m:
                    score = ep.weight * 0.7 + 0.3
                    if score > best_score:
                        best_score = score
                        best_type = ep.event_type
                        evidence = m.group(0)

        if best_score < 0.5:
            best_type = EventType.OTHER
            best_score = 0.3

        return FinancialEvent(
            event_type=best_type,
            confidence=min(best_score, 1.0),
            headline=headline or text[:120],
            evidence=evidence,
            entities=entities or [],
        )
