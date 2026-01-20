"""Emotion prediction using w11wo/indonesian-roberta-base-prdect-id model."""
from typing import List, Union
from transformers import pipeline

_PIPELINE = None


def predict_emotion(text: str, top_k: int = 1) -> List[dict]:
    """Predict emotion for text.
    
    Args:
        text: Input text to classify.
        top_k: Number of top predictions to return.
    
    Returns:
        List of predictions with label and score.
    """
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = pipeline(
            "text-classification",
            model="w11wo/indonesian-roberta-base-prdect-id",
            tokenizer="w11wo/indonesian-roberta-base-prdect-id",
            return_all_scores=True
        )
    
    result = _PIPELINE(text)
    sorted_result = sorted(result[0], key=lambda x: x['score'], reverse=True)
    # Convert numpy types to Python native types for JSON serialization
    return [
        {'label': item['label'], 'score': float(item['score'])}
        for item in sorted_result[:top_k]
    ]
