#!/usr/bin/env python3
"""Run evaluation suite against labeled sample data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from finintel.evaluation.metrics import (
    entity_link_metrics,
    event_classification_metrics,
    sentiment_metrics,
)
from finintel.nlp.entity_linker import EntityLinker
from finintel.nlp.event_extractor import EventExtractor
from finintel.nlp.sentiment import LexiconSentiment

LABELS_PATH = ROOT / "data" / "evaluation" / "sample_labels.json"


def main():
    with open(LABELS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    sentiment = LexiconSentiment()
    y_true_s, y_pred_s = [], []
    for item in data["sentiment"]:
        y_true_s.append(item["label"])
        y_pred_s.append(sentiment.analyze(item["text"])["label"].value)
    s_metrics = sentiment_metrics(y_true_s, y_pred_s)
    print("=== Sentiment Evaluation ===")
    for k, v in s_metrics.items():
        print(f"  {k}: {v:.4f}")

    extractor = EventExtractor()
    y_true_e, y_pred_e = [], []
    for item in data["events"]:
        y_true_e.append(item["label"])
        y_pred_e.append(extractor.extract(item["text"]).event_type.value)
    e_metrics = event_classification_metrics(y_true_e, y_pred_e)
    print("\n=== Event Classification Evaluation ===")
    print(f"  macro_f1: {e_metrics['macro_f1']:.4f}")

    linker = EntityLinker()
    pred_sets, gold_sets = [], []
    for item in data["entities"]:
        entities = linker.extract_and_link(item["text"])
        pred_sets.append(set(linker.get_tickers(entities)))
        gold_sets.append(set(item["tickers"]))
    n_metrics = entity_link_metrics(pred_sets, gold_sets)
    print("\n=== Entity Linking Evaluation ===")
    for k, v in n_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
