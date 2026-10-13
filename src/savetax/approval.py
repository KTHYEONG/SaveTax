"""Approval gate binding a participant challenge to exact body hashes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from src.savetax.competition import EventInfo
from src.savetax.errors import DomainValidationError
from src.savetax.ledger import Ledger

__all__ = ["ReviewRow", "ReviewSet", "build_review_set", "challenge_for", "verify_approval"]


@dataclass(frozen=True)
class ReviewRow:
    """One approval-table row for an item (score submission)."""

    item_id: str
    name: str
    body_sha256: str
    expected_total_krw: int
    lines: int
    max_ratio: Decimal
    source_urls: tuple[str, ...]
    unit_notes: tuple[str, ...]
    caveats: tuple[str, ...]
    next_action: str


@dataclass(frozen=True)
class ReviewSet:
    """A set of rows with the confirmation challenge the participant must repeat to approve."""

    rows: tuple[ReviewRow, ...]
    total_krw: int
    challenge: str
    created_at: datetime
    event_release_id: str


def challenge_for(rows: Sequence[ReviewRow], event_release_id: str) -> str:
    """Return `APPROVE <n> <total> <8-hex>` digested from sorted row tuples."""
    ordered = sorted(rows, key=lambda row: (row.item_id, row.body_sha256, row.expected_total_krw))
    total = sum(row.expected_total_krw for row in ordered)
    payload = json.dumps([event_release_id, [(row.item_id, row.body_sha256, row.expected_total_krw)
                                           for row in ordered]], separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"APPROVE {len(ordered)} {total} {digest}"


def build_review_set(rows: Sequence[ReviewRow], event: EventInfo, now: datetime) -> ReviewSet:
    """Return the review set; raises DomainValidationError when empty."""
    ordered = tuple(sorted(rows, key=lambda row: (-row.expected_total_krw, row.item_id)))
    if not ordered:
        raise DomainValidationError("[DATA] invalid review set: no rows")
    total = sum(row.expected_total_krw for row in ordered)
    return ReviewSet(
        rows=ordered,
        total_krw=total,
        challenge=challenge_for(ordered, event.dataset_release_id),
        created_at=now,
        event_release_id=event.dataset_release_id,
    )


def verify_approval(
    ledger: Ledger,
    item_id: str,
    body_sha256: str,
    now: datetime,
    ttl: timedelta,
    *,
    kind: str = "submit",
) -> bool:
    """Return True only when an unexpired approval for kind matches item and body hash."""
    entries = ledger.entries()
    for position in range(len(entries) - 1, -1, -1):
        entry = entries[position]
        if entry.kind != "approved":
            continue
        if entry.item_id != item_id or entry.body_sha256 != body_sha256:
            continue
        if entry.data.get("kind") != kind:
            continue
        age = now - entry.at
        if age < timedelta(0) or age >= ttl:
            continue
        revoked = any(
            later.kind in ("withdrawn", "refused") and later.item_id == item_id
            for later in entries[position + 1:]
        )
        if revoked:
            continue
        return True
    return False
