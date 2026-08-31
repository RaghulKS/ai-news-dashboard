"""Financial entity recognition and linking to tickers/sectors."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from finintel.config import DATA_DIR
from finintel.models.entity import EntityType, LinkedEntity

TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5})\b")
EXECUTIVE_PATTERN = re.compile(
    r"\b(CEO|CFO|COO|CTO|chairman|president|chief executive|chief financial)\b",
    re.IGNORECASE,
)
MACRO_ENTITIES = {
    "federal reserve": ("Federal Reserve", "macro"),
    "fed": ("Federal Reserve", "macro"),
    "ecb": ("European Central Bank", "macro"),
    "inflation": ("Inflation", "macro"),
    "cpi": ("Consumer Price Index", "macro"),
    "gdp": ("Gross Domestic Product", "macro"),
    "unemployment": ("Unemployment", "macro"),
    "interest rate": ("Interest Rates", "macro"),
    "treasury": ("US Treasury", "macro"),
}
SECTOR_KEYWORDS = {
    "technology": "Technology",
    "healthcare": "Healthcare",
    "energy": "Energy",
    "financials": "Financials",
    "financial services": "Financials",
    "consumer": "Consumer",
    "industrials": "Industrials",
    "real estate": "Real Estate",
    "utilities": "Utilities",
    "materials": "Materials",
    "communication": "Communication Services",
}
REGULATORS = {
    "sec": "SEC",
    "fda": "FDA",
    "ftc": "FTC",
    "doj": "DOJ",
    "cftc": "CFTC",
    "finra": "FINRA",
}


class EntityLinker:
    def __init__(self, ticker_map_path: Optional[Path] = None):
        path = ticker_map_path or DATA_DIR / "entities" / "ticker_map.json"
        with open(path, encoding="utf-8") as f:
            self.ticker_map: dict = json.load(f)
        self._alias_index = self._build_alias_index()

    def _build_alias_index(self) -> dict[str, tuple[str, dict]]:
        index: dict[str, tuple[str, dict]] = {}
        for ticker, info in self.ticker_map.items():
            index[ticker.lower()] = (ticker, info)
            index[info["name"].lower()] = (ticker, info)
            for alias in info.get("aliases", []):
                index[alias.lower()] = (ticker, info)
        return index

    def extract_and_link(self, text: str, query: Optional[str] = None) -> list[LinkedEntity]:
        entities: list[LinkedEntity] = []
        seen: set[str] = set()
        lower = text.lower()

        if query:
            q = query.strip()
            if q.upper() in self.ticker_map:
                self._add_ticker(entities, seen, q.upper())
            elif q.lower() in self._alias_index:
                ticker, info = self._alias_index[q.lower()]
                self._add_ticker(entities, seen, ticker, info["name"])

        for match in TICKER_PATTERN.finditer(text):
            ticker = match.group(1)
            if ticker in self.ticker_map:
                self._add_ticker(entities, seen, ticker)

        for alias, (ticker, info) in self._alias_index.items():
            if len(alias) < 3:
                continue
            if alias in lower:
                self._add_ticker(entities, seen, ticker, info["name"])

        for kw, (name, etype) in MACRO_ENTITIES.items():
            if kw in lower:
                key = f"macro:{name}"
                if key not in seen:
                    seen.add(key)
                    entities.append(
                        LinkedEntity(
                            surface_form=kw,
                            canonical_name=name,
                            entity_type=EntityType.MACRO,
                            confidence=0.85,
                            link_source="macro_dict",
                        )
                    )

        for kw, sector in SECTOR_KEYWORDS.items():
            if kw in lower:
                key = f"sector:{sector}"
                if key not in seen:
                    seen.add(key)
                    entities.append(
                        LinkedEntity(
                            surface_form=kw,
                            canonical_name=sector,
                            entity_type=EntityType.SECTOR,
                            sector=sector,
                            confidence=0.8,
                            link_source="sector_dict",
                        )
                    )

        for abbr, name in REGULATORS.items():
            if re.search(rf"\b{re.escape(abbr)}\b", lower):
                key = f"reg:{name}"
                if key not in seen:
                    seen.add(key)
                    entities.append(
                        LinkedEntity(
                            surface_form=abbr.upper(),
                            canonical_name=name,
                            entity_type=EntityType.REGULATOR,
                            confidence=0.9,
                            link_source="regulator_dict",
                        )
                    )

        if EXECUTIVE_PATTERN.search(text):
            for m in re.finditer(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*,?\s*(CEO|CFO|COO|CTO)", text):
                name = m.group(1)
                key = f"exec:{name}"
                if key not in seen:
                    seen.add(key)
                    entities.append(
                        LinkedEntity(
                            surface_form=name,
                            canonical_name=name,
                            entity_type=EntityType.EXECUTIVE,
                            confidence=0.75,
                            link_source="pattern",
                        )
                    )

        return entities

    def _add_ticker(
        self,
        entities: list[LinkedEntity],
        seen: set[str],
        ticker: str,
        name: Optional[str] = None,
    ) -> None:
        if ticker in seen:
            return
        seen.add(ticker)
        info = self.ticker_map.get(ticker, {})
        entities.append(
            LinkedEntity(
                surface_form=ticker,
                canonical_name=name or info.get("name", ticker),
                entity_type=EntityType.TICKER,
                ticker=ticker,
                sector=info.get("sector"),
                confidence=0.95,
                link_source="ticker_map",
            )
        )

    def get_tickers(self, entities: list[LinkedEntity]) -> list[str]:
        return list({e.ticker for e in entities if e.ticker})
