"""Evaluation metrics for NLP pipeline components."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)


def sentiment_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
    }


def entity_link_metrics(
    predicted: list[set[str]],
    gold: list[set[str]],
) -> dict:
    tp = fp = fn = 0
    for pred, truth in zip(predicted, gold):
        tp += len(pred & truth)
        fp += len(pred - truth)
        fn += len(truth - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def event_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict] = {}
    for label in labels:
        yt = [1 if y == label else 0 for y in y_true]
        yp = [1 if y == label else 0 for y in y_pred]
        per_class[label] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        }
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class": per_class,
    }


def clustering_quality(embeddings: np.ndarray, labels: list[int]) -> dict:
    if len(set(labels)) < 2 or len(labels) < 3:
        return {"silhouette": None, "note": "Need >=2 clusters and >=3 points"}
    try:
        sil = float(silhouette_score(embeddings, labels))
    except ValueError:
        sil = None
    return {"silhouette": sil}


def retrieval_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / min(k, len(relevant)) if relevant else 0.0


def calibration_error(y_true: list[int], y_prob: list[float], n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    for i in range(n_bins):
        mask = (y_prob_arr >= bins[i]) & (y_prob_arr < bins[i + 1])
        if not mask.any():
            continue
        acc = y_true_arr[mask].mean()
        conf = y_prob_arr[mask].mean()
        ece += mask.sum() / len(y_true_arr) * abs(acc - conf)
    return float(ece)
