"""Article deduplication via URL and near-duplicate title detection."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from hashlib import sha256
from typing import Optional

from finintel.models.article import Article


def normalize_title(title: str) -> str:
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


class Deduplicator:
    def __init__(self, similarity_threshold: float = 0.92):
        self.similarity_threshold = similarity_threshold
        self._seen_urls: set[str] = set()
        self._seen_titles: list[str] = []

    def load_existing(self, urls: set[str], titles: set[str]) -> None:
        self._seen_urls = urls
        self._seen_titles = list(titles)

    def is_duplicate(self, article: Article) -> tuple[bool, Optional[str]]:
        if article.url in self._seen_urls:
            return True, article.url

        norm = normalize_title(article.title)
        for existing in self._seen_titles:
            if title_similarity(norm, existing) >= self.similarity_threshold:
                return True, existing

        self._seen_urls.add(article.url)
        self._seen_titles.append(norm)
        return False, None

    @staticmethod
    def content_hash(text: str) -> str:
        return sha256(normalize_title(text).encode()).hexdigest()
