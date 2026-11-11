"""Transparent rule-based sentiment and priority baselines."""
import re

POSITIVE_TERMS = {
    "awesome", "great", "good", "love", "thanks", "thank you", "helpful",
    "excellent", "resolved", "perfect", "happy", "appreciate",
}
NEGATIVE_TERMS = {
    "bad", "broken", "angry", "terrible", "worst", "hate", "disappointed",
    "refund", "charged twice", "fraud", "not working", "failed", "unacceptable",
}
URGENT_TERMS = {
    "urgent", "immediately", "asap", "right now", "locked out", "security",
    "fraud", "cannot access", "can't access", "charged twice", "deadline",
}


def _contains_phrase(text: str, phrases: set[str]) -> bool:
    return any(
        re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, flags=re.IGNORECASE)
        for phrase in phrases
    )


def analyze_sentiment(text: str) -> str:
    """Return a keyword-based sentiment estimate, not contextual NLP."""
    positive = _contains_phrase(text, POSITIVE_TERMS)
    negative = _contains_phrase(text, NEGATIVE_TERMS)
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    return "neutral"


def predict_priority(text: str, sentiment: str | None = None) -> str:
    """Estimate priority from explicit urgency phrases and sentiment."""
    if _contains_phrase(text, URGENT_TERMS):
        return "high"
    if sentiment == "negative" or analyze_sentiment(text) == "negative":
        return "medium"
    return "low"
