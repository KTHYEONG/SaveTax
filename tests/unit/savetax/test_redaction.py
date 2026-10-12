"""Invariant guards for token redaction."""

from src.savetax.redaction import redact


def test_hyphenated_token_is_fully_covered() -> None:
    assert redact("Bearer savetax__aB-cD_12-xy tail") == "Bearer <redacted> tail"


def test_text_without_token_is_unchanged() -> None:
    assert redact("savetax hackathon 2026") == "savetax hackathon 2026"


def test_multiple_tokens_are_all_replaced() -> None:
    result = redact("a savetax_abc-123 tail savetax_X_y-z end")
    assert "savetax_" not in result
    assert result.count("<redacted>") == 2
