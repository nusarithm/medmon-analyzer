"""Sentiment analysis for Indonesian text."""
import os
from typing import List, Union

from service.loader import run

MODEL = os.getenv("SENTIMENT_MODEL", "masnasri-a/indobert-sentiment-analysis")


def analyze_sentiment(text: Union[str, List[str]]):
    """Analyze sentiment. Accepts one string or a list (batched)."""
    results = run("sentiment-analysis", MODEL, text)

    if isinstance(text, str):
        item = results[0] if isinstance(results, list) else results
        return {"label": item["label"], "score": float(item["score"])}

    return [
        {"label": r["label"], "score": float(r["score"])}
        for r in (results if isinstance(results[0], dict) else [x[0] for x in results])
    ]
