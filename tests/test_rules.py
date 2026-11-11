from supportiq.rules import analyze_sentiment, predict_priority


def test_negative_sentiment_label():
    assert analyze_sentiment("This is broken and terrible") == "negative"


def test_positive_sentiment_label():
    assert analyze_sentiment("Great work, thank you") == "positive"


def test_urgent_ticket_is_high_priority():
    assert predict_priority("I am locked out and need help immediately") == "high"


def test_negative_ticket_is_medium_priority():
    assert predict_priority("This product is broken") == "medium"
