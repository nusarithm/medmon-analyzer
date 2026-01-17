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
        model_name = "mdhugol/indonesia-bert-sentiment-classification"
        _PIPELINE = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    
    result = _PIPELINE(text)
    return result[0]