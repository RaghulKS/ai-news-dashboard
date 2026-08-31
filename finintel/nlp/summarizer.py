"""Extractive summarization grounded in source articles."""

from __future__ import annotations

import re
from typing import Optional


class GroundedSummarizer:
    """Selects representative sentences from cluster articles — no hallucination."""

    def summarize(self, texts: list[str], max_sentences: int = 3) -> str:
        if not texts:
            return ""
        sentences: list[tuple[str, float]] = []
        for text in texts:
            for sent in self._split_sentences(text):
                sent = sent.strip()
                if len(sent) < 30:
                    continue
                score = self._sentence_score(sent)
                sentences.append((sent, score))

        if not sentences:
            return texts[0][:300]

        seen: set[str] = set()
        ranked = sorted(sentences, key=lambda x: x[1], reverse=True)
        selected: list[str] = []
        for sent, _ in ranked:
            key = sent.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            selected.append(sent)
            if len(selected) >= max_sentences:
                break
        return " ".join(selected)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return re.split(r"(?<=[.!?])\s+", text)

    @staticmethod
    def _sentence_score(sent: str) -> float:
        score = len(sent) / 200.0
        financial_terms = [
            "earnings", "revenue", "guidance", "merger", "acquisition",
            "stock", "shares", "analyst", "forecast", "profit", "loss",
        ]
        lower = sent.lower()
        score += 0.2 * sum(1 for t in financial_terms if t in lower)
        if re.search(r"\$[\d,]+", sent):
            score += 0.3
        return score
