"""Append-only audit trail with crash-safe idempotency keys."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.savetax.errors import DataIntegrityError
from src.savetax.redaction import redact

__all__ = ["Ledger", "LedgerEntry"]

_KINDS = frozenset({"key", "drafted", "previewed", "approved", "submitted", "lint", "revised", "withdrawn", "refused"})
_DATA_FIELDS = frozenset({"for_kind", "key", "kind", "note", "total", "lines", "item", "amount",
                          "submission_id", "revision", "status", "held", "fact_ids", "verdicts", "hints", "reasons", "body_hash"})


@dataclass(frozen=True)
class LedgerEntry:
    """One append-only record."""

    at: datetime
    kind: str
    item_id: str | None
    submission_id: str | None
    body_sha256: str | None
    idempotency_key: str | None
    data: Mapping[str, object]


def _entry_to_json(entry: LedgerEntry) -> str:
    if set(entry.data) - _DATA_FIELDS:
        raise DataIntegrityError("[DATA] forbidden ledger metadata field")
    return json.dumps(
        {
            "at": entry.at.isoformat(),
            "kind": entry.kind,
            "item_id": entry.item_id,
            "submission_id": entry.submission_id,
            "body_sha256": entry.body_sha256,
            "idempotency_key": entry.idempotency_key,
            "data": dict(entry.data),
        },
        ensure_ascii=False,
    )


def _entry_from_json(line: str) -> LedgerEntry:
    try:
        decoded = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        raise DataIntegrityError("[DATA] malformed ledger line: body is not JSON") from None
    if not isinstance(decoded, dict):
        raise DataIntegrityError("[DATA] malformed ledger line: expected an object")
    try:
        if set(decoded) != {"at", "kind", "item_id", "submission_id", "body_sha256", "idempotency_key", "data"}:
            raise KeyError("schema")
        at_raw = decoded["at"]
        kind = decoded["kind"]
        data = decoded["data"]
    except KeyError:
        raise DataIntegrityError("[DATA] malformed ledger line: missing field") from None
    if not isinstance(at_raw, str) or not isinstance(kind, str) or not isinstance(data, dict):
        raise DataIntegrityError("[DATA] malformed ledger line: bad field types")
    try:
        at = datetime.fromisoformat(at_raw)
    except ValueError:
        raise DataIntegrityError("[DATA] malformed ledger line: bad timestamp") from None
    if at.tzinfo is None or kind not in _KINDS or set(data) - _DATA_FIELDS:
        raise DataIntegrityError("[DATA] malformed ledger schema")
    if any(not (value is None or isinstance(value, (str, int, bool))
                or (isinstance(value, list) and all(isinstance(item, str) for item in value)))
           for value in data.values()):
        raise DataIntegrityError("[DATA] malformed ledger metadata")
    if redact(line) != line:
        raise DataIntegrityError("[DATA] credential-like ledger content is forbidden")
    for field in ("item_id", "submission_id", "body_sha256", "idempotency_key"):
        value = decoded.get(field)
        if value is not None and not isinstance(value, str):
            raise DataIntegrityError(f"[DATA] malformed ledger line: bad {field}")
    return LedgerEntry(
        at=at,
        kind=kind,
        item_id=decoded.get("item_id"),
        submission_id=decoded.get("submission_id"),
        body_sha256=decoded.get("body_sha256"),
        idempotency_key=decoded.get("idempotency_key"),
        data=data,
    )


class Ledger:
    """Append-only JSONL audit trail with crash-safe idempotency keys."""

    def __init__(self, path: Path, *, now: Callable[[], datetime]) -> None:
        self._path = path
        self._now = now

    @property
    def path(self) -> Path:
        """Return the ledger file path."""
        return self._path

    def append(self, entry: LedgerEntry) -> None:
        """Append one line and fsync; raises DataIntegrityError when unwritable."""
        try:
            created = not self._path.exists()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = _entry_to_json(entry) + "\n"
            _entry_from_json(line)
            fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        except (OSError, TypeError, ValueError, UnicodeError):
            raise DataIntegrityError("[DATA] ledger file cannot be written") from None
        try:
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                os.fchmod(handle.fileno(), 0o600)
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(self._path, 0o600)
            if created:
                for parent in self._path.parents:
                    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
        except OSError:
            raise DataIntegrityError("[DATA] ledger file cannot be written") from None

    def entries(self) -> list[LedgerEntry]:
        """Return every entry in order; raises DataIntegrityError on a malformed line."""
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except (OSError, UnicodeError):
            raise DataIntegrityError("[DATA] ledger file cannot be read") from None
        out: list[LedgerEntry] = []
        for raw in text.splitlines():
            if not raw.strip():
                continue
            out.append(_entry_from_json(raw))
        return out

    def key_for(self, kind: str, item_id: str, body_sha256: str, new_key: Callable[[], str]) -> str:
        """Return the existing key of this logical request, or persist and return a new one."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path.with_suffix(self._path.suffix + ".lock"), "a", encoding="utf-8") as lock:
                os.fchmod(lock.fileno(), 0o600)
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                return self._key_for(kind, item_id, body_sha256, new_key)
        except OSError:
            raise DataIntegrityError("[DATA] ledger key cannot be persisted") from None

    def _key_for(self, kind: str, item_id: str, body_sha256: str, new_key: Callable[[], str]) -> str:
        for entry in self.entries():
            if entry.kind != "key":
                continue
            data: Any = dict(entry.data)
            if (
                data.get("for_kind") == kind
                and entry.item_id == item_id
                and entry.body_sha256 == body_sha256
                and isinstance(data.get("key"), str)
                and data["key"]
            ):
                return str(data["key"])
        key = new_key()
        if not isinstance(key, str) or not key:
            raise DataIntegrityError("[DATA] invalid idempotency key factory value")
        self.append(
            LedgerEntry(
                at=self._now(),
                kind="key",
                item_id=item_id,
                submission_id=None,
                body_sha256=body_sha256,
                idempotency_key=key,
                data={"for_kind": kind, "key": key},
            )
        )
        return key
