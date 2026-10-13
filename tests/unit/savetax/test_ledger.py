"""Invariant guards for the append-only ledger."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

from src.savetax.errors import DataIntegrityError
from src.savetax.ledger import Ledger, LedgerEntry

import pytest


def _now() -> datetime:
    return datetime(2026, 10, 13, tzinfo=UTC)


def _entry(**over: object) -> LedgerEntry:
    base: dict[str, object] = {
        "at": _now(),
        "kind": "previewed",
        "item_id": "11111111",
        "submission_id": None,
        "body_sha256": "ab" * 32,
        "idempotency_key": None,
        "data": {"total": 1},
    }
    base.update(over)
    return LedgerEntry(**base)  # type: ignore[arg-type]


def test_append_is_durable_and_ordered(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ledger = Ledger(tmp_path / "ledger.jsonl", now=_now)
    first = _entry(kind="drafted")
    second = _entry(kind="previewed")
    third = _entry(kind="approved")
    ledger.append(first)
    ledger.append(second)
    ledger.append(third)
    assert ledger.entries() == [first, second, third]
    assert os.stat(ledger.path).st_mode & 0o777 == 0o600


def test_malformed_line_fails_closed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"at": "broken"}\n', encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        Ledger(path, now=_now).entries()


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[1, 2]",
        '{"kind": "x", "data": {}}',
        '{"at": "2026-10-13T12:00:00+00:00", "kind": 5, "data": {}}',
        '{"at": "yesterday", "kind": "k", "data": {}}',
        '{"at": "2026-10-13T12:00:00+00:00", "kind": "k", "data": {}, "item_id": 5}',
    ],
)
def test_malformed_variants_fail_closed(tmp_path, line: str) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "ledger.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        Ledger(path, now=_now).entries()


def test_key_survives_restart(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "ledger.jsonl"
    first = Ledger(path, now=_now)
    key = first.key_for("submit", "11111111", "aa", lambda: "key-1")
    second = Ledger(path, now=_now)
    assert second.key_for("submit", "11111111", "aa", lambda: "key-2") == key


def test_changed_body_gets_new_key(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ledger = Ledger(tmp_path / "ledger.jsonl", now=_now)
    first = ledger.key_for("submit", "11111111", "aa", lambda: "key-1")
    second = ledger.key_for("submit", "11111111", "bb", lambda: "key-2")
    assert first != second


def test_key_is_persisted_before_use(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path, now=_now)
    seen: list[str] = []

    def factory() -> str:
        seen.append("called")
        return "key-9"

    assert ledger.key_for("submit", "11111111", "aa", factory) == "key-9"
    assert seen == ["called"]
    assert "key-9" in path.read_text(encoding="utf-8")


def test_no_secret_fields_stored(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ledger = Ledger(tmp_path / "ledger.jsonl", now=_now)
    ledger.append(_entry(kind="previewed", data={"total": 5, "item": "11111111"}))
    raw = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert "quote" not in json.loads(raw.splitlines()[0])["data"]
    assert "quote" not in raw


def test_append_unwritable_fails_closed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    occupied = tmp_path / "ledger.jsonl"
    occupied.mkdir()
    with pytest.raises(DataIntegrityError):
        Ledger(occupied, now=_now).append(_entry())


def test_append_chmod_failure_fails_closed(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import os as _os

    def _boom(_path: object, _mode: int) -> None:
        raise OSError("chmod denied")

    monkeypatch.setattr(_os, "chmod", _boom)
    with pytest.raises(DataIntegrityError):
        Ledger(tmp_path / "ledger.jsonl", now=_now).append(_entry())


def test_entries_unreadable_fails_closed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    occupied = tmp_path / "ledger.jsonl"
    occupied.mkdir()
    with pytest.raises(DataIntegrityError):
        Ledger(occupied, now=_now).entries()


def test_entries_skips_blank_lines(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ledger = Ledger(tmp_path / "ledger.jsonl", now=_now)
    first = _entry(kind="drafted")
    ledger.append(first)
    with open(ledger.path, "a", encoding="utf-8") as handle:
        handle.write("\n   \n")
    assert ledger.entries() == [first]


def test_key_factory_invalid_fails_closed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(DataIntegrityError):
        Ledger(tmp_path / "ledger.jsonl", now=_now).key_for("submit", "11111111", "aa", lambda: "")
