"""
Aspect extractor using spaCy noun-chunk heuristics.
Falls back to a simple keyword list when spaCy model is unavailable.
"""

from typing import List

COMMON_ASPECTS = [
    "battery", "screen", "camera", "performance", "price", "design",
    "build quality", "display", "speaker", "charging", "software",
    "storage", "processor", "keyboard", "touchpad", "service",
]

def extract_aspects(text: str) -> List[str]:
    """
    Extract aspect terms from a review.
    Tries spaCy first; falls back to keyword matching.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        aspects = []
        for chunk in doc.noun_chunks:
            candidate = chunk.root.text.lower()
            if candidate not in {"i", "we", "it", "this", "that", "they"}:
                aspects.append(candidate)
        # deduplicate while preserving order
        seen, unique = set(), []
        for a in aspects:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique if unique else _keyword_fallback(text)
    except Exception:
        return _keyword_fallback(text)

def _keyword_fallback(text: str) -> List[str]:
    text_lower = text.lower()
    return [a for a in COMMON_ASPECTS if a in text_lower] or ["product"]
