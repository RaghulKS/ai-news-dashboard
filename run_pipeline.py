#!/usr/bin/env python3
"""CLI entry point for the financial event intelligence pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from finintel.config import load_config
from finintel.pipeline.orchestrator import FinancialEventPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Financial Event Intelligence Pipeline")
    parser.add_argument("query", help="Ticker, company name, or topic")
    parser.add_argument("--count", type=int, default=20, help="Max articles to fetch")
    parser.add_argument("--sentiment", choices=["lexicon", "finbert", "azure"], default="lexicon")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    cfg = load_config()
    cfg.nlp.sentiment_backend = args.sentiment

    pipeline = FinancialEventPipeline(cfg)
    result = pipeline.run(args.query, args.count)

    if args.json:
        output = {
            "query": result.query,
            "articles_fetched": result.articles_fetched,
            "articles_stored": result.articles_stored,
            "duplicates_skipped": result.duplicates_skipped,
            "clusters": [c.model_dump(mode="json") for c in result.clusters],
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print(f"\nQuery: {result.query}")
        print(f"Fetched: {result.articles_fetched} | Stored: {result.articles_stored} | "
              f"Duplicates skipped: {result.duplicates_skipped}")
        print(f"Event clusters: {len(result.clusters)}\n")
        for i, c in enumerate(result.clusters, 1):
            print(f"{i}. [{c.event_type.value}] {c.representative_title}")
            print(f"   Articles: {c.article_count} | Sources: {c.source_count} | "
                  f"Relevance: {c.market_relevance:.2f} | Novelty: {c.novelty_score:.2f}")
            if c.tickers:
                print(f"   Tickers: {', '.join(c.tickers)}")
            if c.summary:
                print(f"   Summary: {c.summary[:200]}")
            print()


if __name__ == "__main__":
    main()
