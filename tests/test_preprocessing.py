import pandas as pd
import pytest

from supportiq.preprocessing import normalize_text, validate_ticket_frame


def test_normalize_text_collapses_whitespace_and_masks_urls():
    assert normalize_text("  Hello\nhttps://example.com   there ") == "Hello URL there"


def test_validation_requires_text_column():
    with pytest.raises(ValueError, match="text"):
        validate_ticket_frame(pd.DataFrame({"message": ["hello"]}))


def test_validation_drops_empty_text():
    result = validate_ticket_frame(pd.DataFrame({"text": ["hello", "  ", None]}))
    assert result["text"].tolist() == ["hello"]
