"""Sentiment analysis using IndoBERT model."""
from transformers import pipeline

_PIPELINE = None


def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of Indonesian text.
    
    Args:
        text: Input text to analyze.
    
    Returns:
        Dictionary with label and score.
    """
    global _PIPELINE
    if _PIPELINE is None:
        model_name = "masnasri-a/indobert-sentiment-analysis"
        _PIPELINE = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    
    result = _PIPELINE(text)
    # Convert numpy types to Python native types for JSON serialization
    item = result[0]
    return {'label': item['label'], 'score': float(item['score'])}