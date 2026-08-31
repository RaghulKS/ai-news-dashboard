"""Dense text embeddings for semantic clustering."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from finintel.config import CACHE_DIR, AppConfig, load_config

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: Optional[str] = None, config: Optional[AppConfig] = None):
        cfg = config or load_config()
        self.model_name = model_name or cfg.nlp.embedding_model
        self.cache_dir = CACHE_DIR / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        cached, uncached_idx, uncached_texts = self._load_cached(texts)
        if uncached_texts:
            try:
                self._load_model()
                new_vecs = self._model.encode(uncached_texts, normalize_embeddings=True)
            except Exception as exc:
                logger.warning("Transformer embeddings failed, using TF-IDF fallback: %s", exc)
                new_vecs = self._tfidf_fallback(uncached_texts)
            for i, vec in zip(uncached_idx, new_vecs):
                cached[i] = vec
                self._save_cache(texts[i], vec)
        return np.vstack(cached)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _cache_path(self, text: str) -> Path:
        return self.cache_dir / f"{self._cache_key(text)}.json"

    def _load_cached(self, texts: list[str]) -> tuple[list, list[int], list[str]]:
        cached: list = [None] * len(texts)
        uncached_idx: list[int] = []
        uncached_texts: list[str] = []
        for i, t in enumerate(texts):
            p = self._cache_path(t)
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    cached[i] = np.array(json.load(f))
            else:
                uncached_idx.append(i)
                uncached_texts.append(t)
        return cached, uncached_idx, uncached_texts

    def _save_cache(self, text: str, vec: np.ndarray) -> None:
        with open(self._cache_path(text), "w", encoding="utf-8") as f:
            json.dump(vec.tolist(), f)

    @staticmethod
    def _tfidf_fallback(texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=128)
        matrix = vec.fit_transform(texts).toarray()
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
