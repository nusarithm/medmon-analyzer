"""Named Entity Recognition using cahya/bert-base-indonesian-NER model."""
from typing import List
from transformers import pipeline

_PIPELINE = None


def extract_entities(text: str, min_score: float = 0.8) -> List[dict]:
    """Extract named entities from Indonesian text.
    
    Args:
        text: Input text to analyze.
        min_score: Minimum confidence score threshold.
    
    Returns:
        List of entities with entity_group, word, and score.
    """
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = pipeline(
            "ner",
            model="cahya/bert-base-indonesian-NER",
            tokenizer="cahya/bert-base-indonesian-NER",
            aggregation_strategy="simple"
        )
    
    entities = _PIPELINE(text)
    return [e for e in entities if e['score'] >= min_score]
