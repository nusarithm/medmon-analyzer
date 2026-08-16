"""Emotion prediction for Indonesian text."""
import os
from typing import List

from service.loader import run

MODEL = os.getenv("EMOTION_MODEL", "w11wo/indonesian-roberta-base-prdect-id")


def predict_emotion(text: str, top_k: int = 1) -> List[dict]:
    """Predict emotion, returning the top_k labels by score."""
    results = run("text-classification", MODEL, text, top_k=None)

    # top_k=None yields every label; for a single input that is one list
    scores = results[0] if results and isinstance(results[0], list) else results
    ranked = sorted(scores, key=lambda x: x["score"], reverse=True)
    return [
        {"label": item["label"], "score": float(item["score"])}
        for item in ranked[:top_k]
    ]


def predict_emotion_batch(texts: List[str]) -> List[dict]:
    """Top-1 emotion for each text in one batched pass."""
    results = run("text-classification", MODEL, texts, top_k=None)
    out = []
    for scores in results:
        best = max(scores, key=lambda x: x["score"])
        out.append({"label": best["label"], "score": float(best["score"])})
    return out
