"""Application configuration with env overrides and YAML support."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "finintel.db"


def _load_yaml_config() -> dict[str, Any]:
    config_path = ROOT_DIR / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_yaml = _load_yaml_config()


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key) or _yaml.get(key.lower(), default)


@dataclass
class IngestionConfig:
    max_articles_per_query: int = int(_env("MAX_ARTICLES_PER_QUERY", "30") or 30)
    days_back: int = int(_env("DAYS_BACK_FOR_NEWS", "7") or 7)
    request_timeout: int = int(_env("REQUEST_TIMEOUT", "30") or 30)
    rss_feeds: list[str] = field(default_factory=lambda: _yaml.get("rss_feeds", [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
    ]))


@dataclass
class NLPConfig:
    sentiment_backend: str = _env("SENTIMENT_BACKEND", "finbert") or "finbert"
    embedding_model: str = _env(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    ) or "sentence-transformers/all-MiniLM-L6-v2"
    finbert_model: str = _env("FINBERT_MODEL", "ProsusAI/finbert") or "ProsusAI/finbert"
    cluster_similarity_threshold: float = float(
        _env("CLUSTER_SIMILARITY_THRESHOLD", "0.75") or 0.75
    )
    use_azure_fallback: bool = (_env("USE_AZURE_FALLBACK", "false") or "false").lower() == "true"
    enable_summarization: bool = (_env("ENABLE_SUMMARIZATION", "true") or "true").lower() == "true"


@dataclass
class MarketConfig:
    default_horizons: list[int] = field(default_factory=lambda: [1, 5, 21])
    walk_forward_train_days: int = int(_env("WALK_FORWARD_TRAIN_DAYS", "252") or 252)
    walk_forward_test_days: int = int(_env("WALK_FORWARD_TEST_DAYS", "63") or 63)


@dataclass
class AppConfig:
    log_level: str = _env("LOG_LEVEL", "INFO") or "INFO"
    news_api_key: Optional[str] = _env("NEWS_API_KEY")
    azure_endpoint: Optional[str] = _env("AZURE_TEXT_ANALYTICS_ENDPOINT")
    azure_key: Optional[str] = _env("AZURE_TEXT_ANALYTICS_KEY")
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    market: MarketConfig = field(default_factory=MarketConfig)

    def ensure_dirs(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "entities").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "evaluation").mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    cfg = AppConfig()
    try:
        from config_secrets import (  # type: ignore
            AZURE_TEXT_ANALYTICS_ENDPOINT,
            AZURE_TEXT_ANALYTICS_KEY,
            NEWS_API_KEY,
        )
        cfg.news_api_key = cfg.news_api_key or NEWS_API_KEY
        cfg.azure_endpoint = cfg.azure_endpoint or AZURE_TEXT_ANALYTICS_ENDPOINT
        cfg.azure_key = cfg.azure_key or AZURE_TEXT_ANALYTICS_KEY
    except ImportError:
        pass
    cfg.ensure_dirs()
    return cfg
