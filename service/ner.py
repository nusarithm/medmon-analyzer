"""Named Entity Recognition for Indonesian text."""
import os
from typing import List

from service.loader import run

MODEL = os.getenv("NER_MODEL", "cahya/bert-base-indonesian-NER")

# NER is the heaviest of the three and the least used; set NER_ENABLED=0 to
# keep it out of memory entirely on a small box.
ENABLED = os.getenv("NER_ENABLED", "1") == "1"


def _clean(entities, min_score: float) -> List[dict]:
    return [
        {
            "entity_group": e["entity_group"],
            "word": e["word"],
            "score": float(e["score"]),
            "start": int(e["start"]),
            "end": int(e["end"]),
        }
        for e in entities
        if e["score"] >= min_score
    ]


def extract_entities(text: str, min_score: float = 0.8) -> List[dict]:
    """Extract named entities above a confidence threshold."""
    if not ENABLED:
        return []
    entities = run("ner", MODEL, text, aggregation_strategy="simple")
    return _clean(entities, min_score)


def extract_entities_batch(texts: List[str], min_score: float = 0.8) -> List[List[dict]]:
    """Entities for each text in one batched pass."""
    if not ENABLED:
        return [[] for _ in texts]
    batched = run("ner", MODEL, texts, aggregation_strategy="simple")
    return [_clean(e, min_score) for e in batched]
