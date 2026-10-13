"""Invariant guards for the approval gate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from src.savetax.approval import build_review_set, challenge_for, verify_approval
from src.savetax.competition import EventInfo
from src.savetax.errors import DomainValidationError
from src.savetax.ledger import Ledger, LedgerEntry

import pytest


def _now() -> datetime:
    return datetime(2026, 10, 13, 12, tzinfo=UTC)


def _event() -> EventInfo:
    return EventInfo(
        dataset_release_id="release-1",
        ruleset_version="rules-v1",
        phase="open",
        opens_at=datetime(2026, 10, 1, tzinfo=UTC),
        closes_at=datetime(2026, 11, 1, tzinfo=UTC),
        evidence_format="v2",
        item_catalog="catalog",
    )


def _row(item: str = "11111111", body: str = "aa", total: int = 100) -> object:
    from src.savetax.approval import ReviewRow

    return ReviewRow(
        item_id=item,
        name="name",
        body_sha256=body,
        expected_total_krw=total,
        lines=1,
        max_ratio=Decimal("1.5"),
        source_urls=("https://example.com/a",),
        unit_notes=(),
        caveats=(),
        next_action="submit",
    )


def _approve(ledger: Ledger, item: str, body: str, at: datetime, kind: str = "submit") -> None:
    ledger.append(
        LedgerEntry(
            at=at, kind="approved", item_id=item, submission_id=None,
            body_sha256=body, idempotency_key=None, data={"kind": kind, "note": "ok", "total": 1},
        )
    )


def test_challenge_is_deterministic() -> None:
    from src.savetax.approval import ReviewRow

    rows_a: list[ReviewRow] = [_row("11111111", "aa", 100), _row("22222222", "bb", 50)]  # type: ignore[arg-type]
    rows_b: list[ReviewRow] = [_row("22222222", "bb", 50), _row("11111111", "aa", 100)]  # type: ignore[arg-type]
    assert challenge_for(rows_a, "release-1") == challenge_for(rows_b, "release-1")


def test_challenge_changes_with_body() -> None:
    assert challenge_for([_row("11111111", "aa", 100)], "release-1") != challenge_for(  # type: ignore[list-item]
        [_row("11111111", "bb", 100)], "release-1"
    )


def test_approval_is_bound_to_body(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl", now=_now)
    _approve(ledger, "11111111", "H1", _now())
    assert verify_approval(ledger, "11111111", "H2", _now(), timedelta(hours=6)) is False
    assert verify_approval(ledger, "11111111", "H1", _now(), timedelta(hours=6)) is True


def test_approval_expires(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl", now=_now)
    _approve(ledger, "11111111", "H1", _now() - timedelta(hours=7))
    assert verify_approval(ledger, "11111111", "H1", _now(), timedelta(hours=6)) is False


def test_kinds_are_isolated(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl", now=_now)
    _approve(ledger, "11111111", "H1", _now(), kind="submit")
    assert verify_approval(ledger, "11111111", "H1", _now(), timedelta(hours=6), kind="withdraw") is False
    assert verify_approval(ledger, "11111111", "H1", _now(), timedelta(hours=6), kind="submit") is True


def test_withdrawal_revokes(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl", now=_now)
    _approve(ledger, "11111111", "H1", _now() - timedelta(hours=1))
    ledger.append(
        LedgerEntry(
            at=_now(), kind="withdrawn", item_id="11111111", submission_id=None,
            body_sha256=None, idempotency_key=None, data={},
        )
    )
    assert verify_approval(ledger, "11111111", "H1", _now(), timedelta(hours=6)) is False


def test_empty_review_set_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        build_review_set([], _event(), _now())
