"""Pre-flight checks and the approval-gated submission workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from src.savetax.approval import ReviewRow, ReviewSet, build_review_set, verify_approval
from src.savetax.audit import expected_amount_krw
from src.savetax.competition import EventInfo, ItemState, ParticipantStatus
from src.savetax.errors import (
    ApiError,
    ConfigError,
    DataIntegrityError,
    DomainValidationError,
    NetworkError,
    SubmissionPending,
)
from src.savetax.evidence import EvidenceBundle, Lane, _line_to_draft
from src.savetax.ledger import Ledger, LedgerEntry
from src.savetax.redaction import redact
from src.savetax.submissions import PreviewResult, Receipt, SubmissionDetail

if TYPE_CHECKING:
    from src.savetax.competition import CompetitionClient
    from src.savetax.submissions import SubmissionClient

logger = logging.getLogger(__name__)

__all__ = [
    "SubmitPlan",
    "body_hash",
    "load_bundles",
    "load_drafts",
    "preflight",
    "run_approve",
    "run_review",
    "run_revise",
    "run_status",
    "run_submit",
    "run_withdraw",
    "select_for_budget",
    "submission_for_item",
    "track_lint",
]

_STOP_CODES = frozenset({"PARTICIPANT_LIMIT", "DATASET_VERSION_MISMATCH", "SCHEMA_INVALID"})


def _default_key() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class SubmitPlan:
    """Pre-flight verdict for one approved draft."""

    item_id: str
    allowed: bool
    reasons: tuple[str, ...]


def body_hash(body: Mapping[str, object]) -> str:
    """Return the SHA-256 of the canonical JSON bytes for a draft body."""
    try:
        text = json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)
        payload = text.encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise DataIntegrityError("[DATA] submission body is not JSON serializable") from None
    return hashlib.sha256(payload).hexdigest()


def bind_bundle(body: Mapping[str, object], bundle: EvidenceBundle) -> EvidenceBundle:
    """Select exactly the cited bundle lines and reject stale or mismatched drafts."""
    raw = body.get("lines")
    if body.get("canonical_product_id") != bundle.item_id or not isinstance(raw, list) or not raw:
        raise DataIntegrityError("[DATA] draft does not match an evidence bundle")
    by_id = {line.fact_id: line for line in bundle.lines}
    selected = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, dict):
            raise DataIntegrityError("[DATA] malformed draft line")
        fact = value.get("fact_id")
        if not isinstance(fact, str) or fact not in by_id or fact in seen:
            raise DataIntegrityError("[DATA] unknown or duplicated draft fact")
        line = by_id[fact]
        if body_hash(value) != body_hash(_line_to_draft(line)):
            raise DataIntegrityError("[DATA] draft evidence differs from its bundle")
        selected.append(line)
        seen.add(fact)
    return replace(bundle, lines=tuple(selected))


def matching_bundle(body: Mapping[str, object], bundles: Mapping[str, EvidenceBundle]) -> EvidenceBundle:
    """Resolve a draft to the matching evidence lane without mixing prices."""
    for bundle in bundles.values():
        if bundle.item_id != body.get("canonical_product_id"):
            continue
        try:
            return bind_bundle(body, bundle)
        except DataIntegrityError:
            continue
    raise DataIntegrityError("[DATA] draft is missing matching evidence")


def preflight(
    body: Mapping[str, object],
    bundle: EvidenceBundle,
    *,
    event: EventInfo,
    status: ParticipantStatus,
    lookup_state: ItemState,
    preview: PreviewResult,
    page_prices: Mapping[str, int | None],
    reserve: int,
    now: datetime,
    approved: bool = True,
) -> SubmitPlan:
    """Return allow/deny with stable reason codes; never raises for a deny."""
    reasons: list[str] = []
    item = str(body.get("canonical_product_id", bundle.item_id))
    if body.get("dataset_release_id") != event.dataset_release_id or body.get("ruleset_version") != event.ruleset_version:
        reasons.append("release_changed")
    if event.phase != "open" or not (event.opens_at <= now < event.closes_at):
        reasons.append("window_closed")
    if lookup_state is ItemState.CLAIMED_BY_ME:
        reasons.append("already_mine")
    elif lookup_state is not ItemState.AVAILABLE:
        reasons.append("not_available")
    if not preview.ok or preview.next_action != "submit" or preview.product_already_claimed or preview.claimed_by_me:
        reasons.append("preview_not_submittable")
    if preview.total_amount_krw != bundle.expected_total_krw:
        reasons.append("amount_changed")
    for line in bundle.lines:
        for evidence in line.price_evidence:
            if page_prices.get(evidence.source_url) != evidence.observed_price_krw:
                reasons.append("evidence_changed")
                break
        else:
            continue
        break
    if (status.remaining_claims_total is not None and status.remaining_claims_total <= reserve) or (
        status.remaining_claims_per_day is not None and status.remaining_claims_per_day <= reserve
    ):
        reasons.append("limit_reserve")
    if not approved:
        reasons.append("not_approved")
    return SubmitPlan(item_id=item, allowed=not reasons, reasons=tuple(reasons))


def _max_ratio(bundle: EvidenceBundle) -> Decimal:
    best = Decimal("1")
    for line in bundle.lines:
        if line.benchmark_unit_price_krw > 0:
            ratio = line.fact_unit_price_krw / line.benchmark_unit_price_krw
            if ratio > best:
                best = ratio
    return best


def _source_urls(bundle: EvidenceBundle) -> tuple[str, ...]:
    seen: list[str] = []
    for line in bundle.lines:
        for evidence in line.price_evidence:
            if evidence.source_url not in seen:
                seen.append(evidence.source_url)
    return tuple(seen)


def make_review_row(
    item_id: str, body: Mapping[str, object], bundle: EvidenceBundle, preview: PreviewResult, name: str
) -> ReviewRow:
    """Build one approval-table row from a draft, bundle, and preview."""
    return ReviewRow(
        item_id=item_id,
        name=name,
        body_sha256=body_hash(body),
        expected_total_krw=bundle.expected_total_krw,
        lines=len(bundle.lines),
        max_ratio=_max_ratio(bundle),
        source_urls=_source_urls(bundle),
        unit_notes=tuple(bundle.unit_notes),
        caveats=tuple(bundle.caveats),
        next_action=preview.next_action,
    )


def select_for_budget(
    rows: Sequence[ReviewRow], remaining_total: int | None, reserve: int
) -> tuple[list[ReviewRow], list[ReviewRow]]:
    """Split rows into selected and deferred under the claim-count reserve."""
    if remaining_total is None:
        return list(rows), []
    capacity = max(0, remaining_total - reserve)
    return list(rows[:capacity]), list(rows[capacity:])


def _observed_prices(bundle: EvidenceBundle) -> dict[str, int]:
    return {evidence.source_url: evidence.observed_price_krw for line in bundle.lines for evidence in line.price_evidence}


def _compute_rows(
    drafts: Sequence[tuple[str, Mapping[str, object], EvidenceBundle]],
    *,
    event: EventInfo,
    status: ParticipantStatus,
    competition: CompetitionClient,
    submission_client: SubmissionClient,
    now: datetime,
) -> tuple[list[ReviewRow], list[tuple[str, tuple[str, ...]]]]:
    ordered = sorted(drafts, key=lambda entry: (-entry[2].expected_total_krw, entry[0]))
    rows: list[ReviewRow] = []
    excluded: list[tuple[str, tuple[str, ...]]] = []
    for item_id, body, bundle in ordered:
        bundle = bind_bundle(body, bundle)
        state = competition.lookup_items([item_id])[0].state
        preview = submission_client.preview(body)
        plan = preflight(
            body, bundle, event=event, status=status, lookup_state=state, preview=preview,
            page_prices=_observed_prices(bundle), reserve=0, now=now, approved=True,
        )
        gated = tuple(reason for reason in plan.reasons if reason != "limit_reserve" and reason != "evidence_changed")
        if gated:
            excluded.append((item_id, gated))
            continue
        rows.append(make_review_row(item_id, body, bundle, preview, bundle.title))
    return rows, excluded


def _print_table(stream: TextIO, rows: Sequence[ReviewRow], total: int, challenge: str) -> None:
    stream.write("item | name | expected_krw | lines | max_ratio | urls | unit_notes | caveats\n")
    for row in rows:
        stream.write(
            f"{redact(row.item_id)} | {redact(row.name)} | {row.expected_total_krw} | {row.lines} | "
            f"{row.max_ratio} | {redact(', '.join(row.source_urls))} | "
            f"{redact('; '.join(row.unit_notes))} | {redact('; '.join(row.caveats))}\n"
        )
    stream.write(f"total={total} challenge={challenge}\n")


def run_review(
    drafts: Sequence[tuple[str, Mapping[str, object], EvidenceBundle]],
    *,
    event: EventInfo,
    status: ParticipantStatus,
    competition: CompetitionClient,
    submission_client: SubmissionClient,
    ledger: Ledger,
    now: datetime,
    limit: int | None = None,
    reserve: int = 5,
    out: TextIO | None = None,
) -> ReviewSet:
    """Preview drafts, print the approval table, and return the review set."""
    usable = sorted(drafts, key=lambda entry: (-entry[2].expected_total_krw, entry[0]))
    if limit is not None:
        usable = usable[:limit]
    rows, excluded = _compute_rows(usable, event=event, status=status, competition=competition,
                                   submission_client=submission_client, now=now)
    rows = sorted(rows, key=lambda row: (-row.expected_total_krw, row.item_id))
    budgets = [n for n in (status.remaining_claims_total, status.remaining_claims_per_day) if n is not None]
    selected, deferred = select_for_budget(rows, min(budgets) if budgets else None, reserve)
    for row in selected:
        ledger.append(
            LedgerEntry(at=now, kind="previewed", item_id=row.item_id, submission_id=None,
                        body_sha256=row.body_sha256, idempotency_key=None,
                        data={"total": row.expected_total_krw, "lines": row.lines})
        )
    stream = out if out is not None else sys.stdout
    for item_id, reasons in excluded:
        stream.write(f"excluded {redact(item_id)}: {','.join(reasons)}\n")
    for row in deferred:
        stream.write(f"deferred {redact(row.item_id)}: budget\n")
    review = build_review_set(selected, event, now)
    _print_table(stream, selected, review.total_krw, review.challenge)
    return review


def run_approve(
    drafts: Sequence[tuple[str, Mapping[str, object], EvidenceBundle]],
    *,
    event: EventInfo,
    status: ParticipantStatus,
    competition: CompetitionClient,
    submission_client: SubmissionClient,
    ledger: Ledger,
    now: datetime,
    challenge: str,
    note: str,
    kind: str = "submit",
    item: str | None = None,
    limit: int | None = None,
    reserve: int = 5,
    out: TextIO | None = None,
) -> int:
    """Record participant approval bound to exact body hashes; returns approved count."""
    if not note.strip():
        raise ConfigError("[SYS] invalid approval note: must be non-empty")
    if kind not in ("submit", "revise", "withdraw"):
        raise ConfigError("[SYS] invalid approval kind")
    if kind != "submit" and item is None:
        raise ConfigError("[SYS] action approval requires --item")
    usable = sorted(drafts, key=lambda entry: (-entry[2].expected_total_krw, entry[0]))
    if item is not None:
        usable = [entry for entry in usable if entry[0] == item]
    if limit is not None:
        usable = usable[:limit]
    targets: dict[str, SubmissionDetail] = {}
    if kind == "submit":
        rows, _ = _compute_rows(usable, event=event, status=status, competition=competition,
                                submission_client=submission_client, now=now)
    else:
        rows, targets = action_review(usable, submission_client, kind, out=out)
    rows = sorted(rows, key=lambda row: (-row.expected_total_krw, row.item_id))
    budgets = [n for n in (status.remaining_claims_total, status.remaining_claims_per_day) if n is not None]
    selected, _ = select_for_budget(rows, min(budgets) if budgets else None, reserve) if kind == "submit" else (rows, [])
    review = build_review_set(selected, event, now)
    if review.challenge != challenge:
        raise ConfigError("[SYS] approval challenge mismatch")
    for row in selected:
        ledger.append(
            LedgerEntry(at=now, kind="approved", item_id=row.item_id,
                        submission_id=targets[row.item_id].submission_id if row.item_id in targets else None,
                        body_sha256=row.body_sha256, idempotency_key=None,
                        data={"kind": kind, "note": note, "total": row.expected_total_krw})
        )
    return len(selected)


def action_hash(kind: str, detail: SubmissionDetail, body: Mapping[str, object]) -> str:
    """Bind an action to its target, current revision, and exact request body."""
    return body_hash({"kind": kind, "submission_id": detail.submission_id,
                      "revision": detail.revision, "body": dict(body) if kind == "revise" else None})


def action_review(
    drafts: Sequence[tuple[str, Mapping[str, object], EvidenceBundle]],
    client: SubmissionClient,
    kind: str,
    *,
    out: TextIO | None = None,
) -> tuple[list[ReviewRow], dict[str, SubmissionDetail]]:
    """Build action-specific rows with the current submission target and coverage."""
    stream = out if out is not None else sys.stdout
    own = client.list_my_submissions()
    rows: list[ReviewRow] = []
    targets: dict[str, SubmissionDetail] = {}
    for item, body, bundle in drafts:
        bundle = bind_bundle(body, bundle)
        matches = [detail for detail in own if detail.item_id == item and detail.state != "withdrawn"]
        if len(matches) != 1:
            raise ConfigError("[SYS] action requires exactly one current own submission")
        detail = client.get_submission(matches[0].submission_id)
        if detail.item_id != item or body.get("canonical_product_id") != item:
            raise ConfigError("[SYS] action product mismatch")
        preview = client.preview(body) if kind == "revise" else PreviewResult(
            True, bundle.expected_total_krw, {}, "withdraw", False, True, detail.submission_id, detail.revision)
        if kind == "revise" and (not preview.ok or preview.next_action != "revise_existing"
                                 or preview.total_amount_krw != bundle.expected_total_krw):
            raise ConfigError("[SYS] revision preview is not admissible")
        row = make_review_row(item, body, bundle, preview, bundle.title)
        row = replace(row, body_sha256=action_hash(kind, detail, body), next_action=kind)
        stream.write(f"{kind} {redact(item)} sid={redact(detail.submission_id)} "
                     f"revision={detail.revision} old={detail.amount_krw} new={row.expected_total_krw}\n")
        if kind == "withdraw":
            stream.write("withdrawing does not refund the used claim count; others may claim the item\n")
        rows.append(row)
        targets[item] = detail
    return rows, targets


def track_lint(
    submission_client: SubmissionClient,
    ledger: Ledger,
    *,
    item_id: str,
    submission_id: str,
    body_sha256: str,
    now: datetime,
    poll_seconds: float = 7.0,
    timeout_seconds: float = 600.0,
    sleep: Callable[[float], None] = time.sleep,
    out: TextIO | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Poll lint until settled, append the outcome, and return the final status."""
    stream = out if out is not None else sys.stdout
    attempts = max(1, int(timeout_seconds // poll_seconds) + 1)
    deadline = clock() + timeout_seconds
    detail: SubmissionDetail | None = None
    for attempt in range(attempts):
        if attempt and clock() >= deadline:
            break
        detail = submission_client.get_submission(submission_id)
        if detail.lint_status != "pending":
            break
        if attempt < attempts - 1:
            sleep(max(0.0, min(poll_seconds, deadline - clock())))
    final_status = detail.lint_status if detail is not None else "pending"
    ledger.append(
        LedgerEntry(at=now, kind="lint", item_id=item_id, submission_id=submission_id,
                    body_sha256=body_sha256, idempotency_key=None,
                    data=lint_data(detail) if detail is not None else {"status": "pending"})
    )
    if final_status == "suspect" and detail is not None:
        for line in detail.lint_lines:
            stream.write(f"suspect {redact(line.fact_id)}: {redact(line.reason)}\n")
        for hint in detail.lint_hints:
            stream.write(f"hint {redact(hint.code)}: {redact(hint.message)}\n")
        stream.write("action: page_not_read -> choose a publicly readable page; price mismatch -> re-read the page\n")
    elif final_status == "pending":
        stream.write(f"pending {redact(submission_id)}: lint window expired\n")
    return final_status


def lint_data(detail: SubmissionDetail) -> dict[str, object]:
    """Return credential-free lint metadata with fact identities."""
    return {"status": detail.lint_status, "revision": detail.revision, "amount": detail.amount_krw,
            "held": detail.held, "fact_ids": [line.fact_id for line in detail.lint_lines],
            "verdicts": [line.verdict for line in detail.lint_lines],
            "hints": [hint.code for hint in detail.lint_hints]}


def run_submit(
    drafts: Sequence[tuple[str, Mapping[str, object], EvidenceBundle]],
    *,
    event: EventInfo,
    status: ParticipantStatus,
    competition: CompetitionClient,
    submission_client: SubmissionClient,
    ledger: Ledger,
    fetch_price: Callable[[str], int | None],
    now: datetime,
    ttl: timedelta,
    reserve: int = 5,
    execute: bool = False,
    limit: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    new_key: Callable[[], str] = _default_key,
    poll_seconds: float = 7.0,
    timeout_seconds: float = 600.0,
    pause_seconds: float = 1.0,
    stop_after_consecutive_suspect: int = 2,
    clock: Callable[[], datetime] | None = None,
    out: TextIO | None = None,
) -> tuple[list[Receipt], int]:
    """Submit approved drafts sequentially; returns receipts and the process exit code."""
    stream = out if out is not None else sys.stdout
    ordered = sorted(drafts, key=lambda entry: (-entry[2].expected_total_krw, entry[0]))
    if limit is not None:
        ordered = ordered[:limit]
    if not execute:
        for item_id, _body, bundle in ordered:
            stream.write(f"would-submit {redact(item_id)} expected={bundle.expected_total_krw}\n")
        return [], 0
    receipts: list[Receipt] = []
    exit_code = 0
    consecutive_suspect = 0
    for item_id, body, bundle in ordered:
        body = json.loads(json.dumps(body, ensure_ascii=False, allow_nan=False))
        bundle = bind_bundle(body, bundle)
        moment = clock() if clock is not None else now
        now = moment
        digest = body_hash(body)
        if not verify_approval(ledger, item_id, digest, moment, ttl, kind="submit"):
            ledger.append(
                LedgerEntry(at=now, kind="refused", item_id=item_id, submission_id=None,
                            body_sha256=digest, idempotency_key=None, data={"reasons": ["not_approved"]})
            )
            exit_code = 2
            continue
        existing_keys = [entry.idempotency_key for entry in ledger.entries()
                         if entry.kind == "key" and entry.item_id == item_id
                         and entry.body_sha256 == digest and entry.data.get("for_kind") == "submit"]
        if existing_keys:
            outcome = submission_client.resolve_request(str(existing_keys[-1]))
            if outcome.state == "pending":
                return receipts, 3
            if outcome.receipt is not None:
                receipt = outcome.receipt
                if receipt.submission_state != "accepted":
                    raise DataIntegrityError("[DATA] unexpected recovered submission receipt")
                if not any(entry.kind == "submitted" and entry.idempotency_key == existing_keys[-1]
                           for entry in ledger.entries()):
                    ledger.append(LedgerEntry(moment, "submitted", item_id, receipt.submission_id,
                                              digest, existing_keys[-1], {"amount": receipt.amount_krw}))
                stream.write(f"recovered {redact(item_id)} sid={redact(receipt.submission_id)} "
                             f"amount={receipt.amount_krw}\n")
                verdict = track_lint(submission_client, ledger, item_id=item_id, submission_id=receipt.submission_id,
                                     body_sha256=digest, now=moment, poll_seconds=poll_seconds,
                                     timeout_seconds=timeout_seconds, sleep=sleep, out=stream)
                consecutive_suspect = consecutive_suspect + 1 if verdict == "suspect" else 0
                if consecutive_suspect >= stop_after_consecutive_suspect:
                    return receipts, 2
                continue
        state = competition.lookup_items([item_id])[0].state
        if state is not ItemState.AVAILABLE:
            reason = "already_mine" if state is ItemState.CLAIMED_BY_ME else "not_available"
            ledger.append(
                LedgerEntry(at=now, kind="refused", item_id=item_id, submission_id=None,
                            body_sha256=digest, idempotency_key=None, data={"reasons": [reason]})
            )
            exit_code = 2
            continue
        preview = submission_client.preview(body)
        prices = {url: fetch_price(url) for url in _source_urls(bundle)}
        moment = clock() if clock is not None else now
        now = moment
        current_event = competition.get_event() if clock is not None else event
        if clock is not None:
            fresh = competition.get_me()
            remaining_total = [n for n in (status.remaining_claims_total, fresh.remaining_claims_total) if n is not None]
            remaining_daily = [n for n in (status.remaining_claims_per_day, fresh.remaining_claims_per_day) if n is not None]
            status = replace(fresh, remaining_claims_total=min(remaining_total) if remaining_total else None,
                             remaining_claims_per_day=min(remaining_daily) if remaining_daily else None)
        plan = preflight(body, bundle, event=current_event, status=status, lookup_state=state,
                         preview=preview, page_prices=prices, reserve=reserve, now=moment,
                         approved=verify_approval(ledger, item_id, digest, moment, ttl))
        if not plan.allowed:
            ledger.append(
                LedgerEntry(at=now, kind="refused", item_id=item_id, submission_id=None,
                            body_sha256=digest, idempotency_key=None, data={"reasons": list(plan.reasons)})
            )
            exit_code = 2
            continue
        key = ledger.key_for("submit", item_id, digest, new_key)
        try:
            receipt = submission_client.submit(body, idempotency_key=key)
        except SubmissionPending:
            ledger.append(
                LedgerEntry(at=now, kind="lint", item_id=item_id, submission_id=None,
                            body_sha256=digest, idempotency_key=key, data={"status": "pending"})
            )
            return receipts, 3
        except NetworkError:
            return receipts, 3
        except ApiError as exc:
            if exc.code in _STOP_CODES:
                ledger.append(
                    LedgerEntry(at=now, kind="refused", item_id=item_id, submission_id=None,
                                body_sha256=digest, idempotency_key=key, data={"reasons": [exc.code]})
                )
                return receipts, 2
            ledger.append(
                LedgerEntry(at=now, kind="refused", item_id=item_id, submission_id=None,
                            body_sha256=digest, idempotency_key=key, data={"reasons": [exc.code]})
            )
            exit_code = 2
            continue
        if any(saved.submission_id == receipt.submission_id for saved in receipts):
            raise DataIntegrityError("[DATA] duplicate receipt for distinct requests")
        receipts.append(receipt)
        status = replace(status,
                         remaining_claims_total=None if status.remaining_claims_total is None else status.remaining_claims_total - 1,
                         remaining_claims_per_day=None if status.remaining_claims_per_day is None else status.remaining_claims_per_day - 1)
        ledger.append(
            LedgerEntry(at=now, kind="submitted", item_id=item_id, submission_id=receipt.submission_id,
                        body_sha256=digest, idempotency_key=key,
                        data={"submission_id": receipt.submission_id, "amount": receipt.amount_krw,
                              "status": receipt.lint_status})
        )
        stream.write(f"submitted {redact(item_id)} sid={redact(receipt.submission_id)} "
                     f"revision={receipt.revision} amount={receipt.amount_krw}\n")
        final_status = track_lint(submission_client, ledger, item_id=item_id,
                                  submission_id=receipt.submission_id, body_sha256=digest, now=now,
                                  poll_seconds=poll_seconds, timeout_seconds=timeout_seconds,
                                  sleep=sleep, out=stream)
        if final_status == "suspect":
            consecutive_suspect += 1
            if consecutive_suspect >= stop_after_consecutive_suspect:
                exit_code = 2
                break
        else:
            consecutive_suspect = 0
        sleep(pause_seconds)
    return receipts, exit_code


def run_status(
    submission_client: SubmissionClient,
    ledger: Ledger,
    now: datetime,
    out: TextIO | None = None,
) -> list[SubmissionDetail]:
    """List own submissions and record lint changes; read-only."""
    details = submission_client.list_my_submissions()
    seen = {entry.submission_id: dict(entry.data) for entry in ledger.entries() if entry.kind == "lint"}
    stream = out if out is not None else sys.stdout
    for detail in details:
        state = lint_data(detail)
        if seen.get(detail.submission_id) != state:
            ledger.append(
                LedgerEntry(at=now, kind="lint", item_id=detail.item_id, submission_id=detail.submission_id,
                            body_sha256=None, idempotency_key=None,
                            data=state)
            )
        stream.write(
            f"{redact(detail.submission_id)} rev={detail.revision} "
            f"amount={detail.amount_krw} held={detail.held} lint={detail.lint_status} "
            f"hints={redact(','.join(hint.code for hint in detail.lint_hints))}\n"
        )
    return details


def run_revise(
    submission_id: str,
    item_id: str,
    body: Mapping[str, object],
    *,
    submission_client: SubmissionClient,
    ledger: Ledger,
    now: datetime,
    ttl: timedelta,
    execute: bool = False,
    new_key: Callable[[], str] = _default_key,
    out: TextIO | None = None,
    bundle: EvidenceBundle | None = None,
    fetch_price: Callable[[str], int | None] | None = None,
    event: EventInfo | None = None,
    clock: Callable[[], datetime] | None = None,
    event_provider: Callable[[], EventInfo] | None = None,
) -> Receipt | None:
    """Revise a submission guarded by its own approval and current revision."""
    stream = out if out is not None else sys.stdout
    if not execute:
        stream.write(f"would-revise {redact(submission_id)} body_sha256={body_hash(body)}\n")
        return None
    body = json.loads(json.dumps(body, ensure_ascii=False, allow_nan=False))
    detail = submission_client.get_submission(submission_id)
    digest = action_hash("revise", detail, body)
    if not verify_approval(ledger, item_id, digest, now, ttl, kind="revise"):
        ledger.append(
            LedgerEntry(at=now, kind="refused", item_id=item_id, submission_id=submission_id,
                        body_sha256=digest, idempotency_key=None, data={"reasons": ["not_approved"]})
        )
        return None
    if detail.item_id != item_id:
        ledger.append(
            LedgerEntry(at=now, kind="refused", item_id=item_id, submission_id=submission_id,
                        body_sha256=digest, idempotency_key=None, data={"reasons": ["product_mismatch"]})
        )
        return None
    if body.get("canonical_product_id") != item_id:
        ledger.append(
            LedgerEntry(at=now, kind="refused", item_id=item_id, submission_id=submission_id,
                        body_sha256=digest, idempotency_key=None, data={"reasons": ["product_mismatch"]})
        )
        return None
    preview = submission_client.preview(body)
    if not preview.ok or preview.next_action != "revise_existing":
        raise ConfigError("[SYS] revision preview is not admissible")
    if bundle is None or fetch_price is None or event is None:
        raise ConfigError("[SYS] revision requires fresh evidence and event")
    bundle = bind_bundle(body, bundle)
    if (preview.total_amount_krw != bundle.expected_total_krw
            or any(fetch_price(e.source_url) != e.observed_price_krw
                   for line in bundle.lines for e in line.price_evidence)):
        raise ConfigError("[SYS] revision evidence or amount changed")
    now = clock() if clock is not None else now
    event = event_provider() if event_provider is not None else event
    if not verify_approval(ledger, item_id, digest, now, ttl, kind="revise"):
        raise ConfigError("[SYS] revision approval expired")
    if (body.get("dataset_release_id") != event.dataset_release_id
                             or body.get("ruleset_version") != event.ruleset_version
                             or event.phase != "open" or not event.opens_at <= now < event.closes_at):
        raise ConfigError("[SYS] revision event changed")
    key = ledger.key_for("revise", item_id, digest, new_key)
    try:
        receipt = submission_client.revise(submission_id, body, if_match=detail.revision, idempotency_key=key)
    except ApiError as exc:
        if exc.code == "REVISION_CONFLICT":
            submission_client.get_submission(submission_id)
            return None
        raise
    ledger.append(
        LedgerEntry(at=now, kind="revised", item_id=item_id, submission_id=receipt.submission_id,
                    body_sha256=digest, idempotency_key=key,
                        data={"submission_id": receipt.submission_id, "amount": receipt.amount_krw,
                              "status": receipt.lint_status})
    )
    stream.write(f"revised {redact(item_id)} sid={redact(receipt.submission_id)} "
                 f"revision={receipt.revision} amount={receipt.amount_krw}\n")
    track_lint(submission_client, ledger, item_id=item_id, submission_id=receipt.submission_id,
               body_sha256=digest, now=now)
    return receipt


def run_withdraw(
    submission_id: str,
    item_id: str,
    body: Mapping[str, object],
    *,
    submission_client: SubmissionClient,
    ledger: Ledger,
    now: datetime,
    ttl: timedelta,
    execute: bool = False,
    new_key: Callable[[], str] = _default_key,
    out: TextIO | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Receipt | None:
    """Withdraw a submission guarded by its own approval."""
    stream = out if out is not None else sys.stdout
    stream.write("withdrawing does not refund the used claim count; others may claim the item\n")
    if not execute:
        stream.write(f"would-withdraw {redact(submission_id)}\n")
        return None
    detail = submission_client.get_submission(submission_id)
    if detail.item_id != item_id:
        raise ConfigError("[SYS] withdrawal product mismatch")
    digest = action_hash("withdraw", detail, body)
    now = clock() if clock is not None else now
    if not verify_approval(ledger, item_id, digest, now, ttl, kind="withdraw"):
        ledger.append(
            LedgerEntry(at=now, kind="refused", item_id=item_id, submission_id=submission_id,
                        body_sha256=digest, idempotency_key=None, data={"reasons": ["not_approved"]})
        )
        return None
    key = ledger.key_for("withdraw", item_id, digest, new_key)
    receipt = submission_client.withdraw(submission_id, idempotency_key=key)
    ledger.append(
        LedgerEntry(at=now, kind="withdrawn", item_id=item_id, submission_id=receipt.submission_id,
                    body_sha256=digest, idempotency_key=key, data={"submission_id": receipt.submission_id})
    )
    stream.write(f"withdrawn {redact(item_id)} sid={redact(receipt.submission_id)}\n")
    return receipt


def submission_for_item(ledger: Ledger, item_id: str) -> str | None:
    """Return the latest submitted submission id for an item, if any."""
    found: str | None = None
    for entry in ledger.entries():
        if entry.kind == "submitted" and entry.item_id == item_id and entry.submission_id is not None:
            found = entry.submission_id
    return found


def load_drafts(drafts_dir: Path) -> list[tuple[str, dict[str, object]]]:
    """Load draft bodies from a directory ordered by file name."""
    out: list[tuple[str, dict[str, object]]] = []
    if not drafts_dir.is_dir():
        return out
    for path in sorted(drafts_dir.rglob("*.json")):
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError, OSError):
            raise DataIntegrityError(f"[DATA] malformed draft at {path.name}") from None
        if not isinstance(decoded, dict):
            raise DataIntegrityError(f"[DATA] malformed draft at {path.name}")
        item = decoded.get("canonical_product_id")
        if not isinstance(item, str) or not item:
            raise DataIntegrityError(f"[DATA] malformed draft at {path.name}: missing product id")
        out.append((item, decoded))
    return out


def _bundle_from_json(data: Mapping[str, object], *, now: datetime) -> EvidenceBundle:
    from datetime import date

    from src.savetax.evidence import LineProposal, PriceEvidence, RejectedFact

    def _req_str(field: str) -> str:
        value = data.get(field)
        if not isinstance(value, str):
            raise DataIntegrityError(f"[DATA] malformed bundle at {field}")
        return value

    try:
        lane = Lane(str(data.get("lane", "cross_contract")))
    except ValueError:
        raise DataIntegrityError("[DATA] malformed bundle at lane") from None
    raw_lines = data.get("lines")
    if not isinstance(raw_lines, list):
        raise DataIntegrityError("[DATA] malformed bundle at lines")
    lines: list[LineProposal] = []
    for entry in raw_lines:
        if not isinstance(entry, dict):
            raise DataIntegrityError("[DATA] malformed bundle at lines")
        if any(not isinstance(entry.get(field), str) or not entry[field]
               for field in ("fact_id", "comparison_basis_id", "unit", "tax_basis", "request_date",
                             "benchmark_unit_price_krw", "fact_unit_price_krw", "quantity", "comparable_amount_krw")):
            raise DataIntegrityError("[DATA] malformed bundle line field types")
        evidences: list[PriceEvidence] = []
        raw_evidence = entry.get("price_evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise DataIntegrityError("[DATA] malformed bundle at price_evidence")
        for item in raw_evidence:
            if not isinstance(item, dict):
                raise DataIntegrityError("[DATA] malformed bundle at price_evidence")
            price = item.get("observed_price_krw")
            if not isinstance(price, int) or isinstance(price, bool):
                raise DataIntegrityError("[DATA] malformed bundle at observed_price_krw")
            observed_at = item.get("observed_at")
            if not isinstance(observed_at, str):
                raise DataIntegrityError("[DATA] malformed bundle at observed_at")
            if (not isinstance(item.get("vat_included"), bool)
                    or not isinstance(item.get("shipping_included"), bool)
                    or not isinstance(item.get("quote"), str)
                    or not isinstance(item.get("source_url"), str)):
                raise DataIntegrityError("[DATA] malformed bundle citation types")
            evidences.append(
                PriceEvidence(
                    source_url=str(item.get("source_url")),
                    quote=str(item.get("quote")),
                    observed_price_krw=price,
                    observed_at=date.fromisoformat(observed_at),
                    vat_included=bool(item.get("vat_included", True)),
                    shipping_included=bool(item.get("shipping_included", True)),
                    note=item.get("note") if isinstance(item.get("note"), str) else None,
                )
            )
        expected = entry.get("expected_krw")
        if not isinstance(expected, int) or isinstance(expected, bool):
            raise DataIntegrityError("[DATA] malformed bundle at expected_krw")
        lines.append(
            LineProposal(
                fact_id=str(entry.get("fact_id")),
                benchmark_unit_price_krw=Decimal(str(entry.get("benchmark_unit_price_krw"))),
                comparison_basis_id=str(entry.get("comparison_basis_id")),
                unit=str(entry.get("unit")),
                tax_basis=str(entry.get("tax_basis")),
                fact_unit_price_krw=Decimal(str(entry.get("fact_unit_price_krw"))),
                quantity=Decimal(str(entry.get("quantity"))),
                comparable_amount_krw=Decimal(str(entry.get("comparable_amount_krw"))),
                request_date=date.fromisoformat(str(entry.get("request_date"))),
                expected_krw=expected,
                price_evidence=tuple(evidences),
            )
        )
        line = lines[-1]
        if (not line.quantity.is_finite() or line.quantity <= 0
                or not line.fact_unit_price_krw.is_finite() or line.fact_unit_price_krw <= 0
                or line.expected_krw <= 0 or line.expected_krw != expected_amount_krw(
                    [(line.comparable_amount_krw, line.quantity)], line.benchmark_unit_price_krw)):
            raise DataIntegrityError("[DATA] inconsistent bundle line amount")
    if len({line.fact_id for line in lines}) != len(lines):
        raise DataIntegrityError("[DATA] duplicate bundle facts")
    raw_rejected = data.get("rejected", [])
    if not isinstance(raw_rejected, list) or any(not isinstance(item, dict)
                                                or not isinstance(item.get("fact_id"), str)
                                                or not isinstance(item.get("reason"), str) for item in raw_rejected):
        raise DataIntegrityError("[DATA] malformed bundle at rejected")
    rejected = tuple(
        RejectedFact(fact_id=str(item.get("fact_id")), reason=str(item.get("reason")))
        for item in raw_rejected if isinstance(item, dict)
    )
    generated_raw = data.get("generated_at")
    generated = now if not isinstance(generated_raw, str) else datetime.fromisoformat(generated_raw)
    unit_notes = data.get("unit_notes", [])
    caveats = data.get("caveats", [])
    if (not isinstance(unit_notes, list) or not isinstance(caveats, list)
            or any(not isinstance(item, str) for item in [*unit_notes, *caveats])
            or not isinstance(data.get("review_required", False), bool)
            or generated.tzinfo is None):
        raise DataIntegrityError("[DATA] malformed bundle review metadata")
    return EvidenceBundle(
        item_id=_req_str("item_id"),
        lane=lane,
        title=_req_str("title"),
        claim=str(data.get("claim", "")),
        comparison_scope=str(data.get("comparison_scope", "")),
        reproduction_notes=str(data.get("reproduction_notes", "")),
        lines=tuple(lines),
        rejected=rejected,
        caveats=tuple(str(item) for item in caveats) if isinstance(caveats, list) else (),
        review_required=bool(data.get("review_required", False)),
        generated_at=generated,
        unit_notes=tuple(str(item) for item in unit_notes) if isinstance(unit_notes, list) else (),
    )


def load_bundles(paths: Sequence[Path], *, now: datetime) -> dict[str, EvidenceBundle]:
    """Load bundles from JSONL files keyed by item id."""
    out: dict[str, EvidenceBundle] = {}
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError):
            raise DataIntegrityError(f"[DATA] bundle file cannot be read at {path.name}") from None
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except ValueError:
                raise DataIntegrityError(f"[DATA] malformed bundle line at {path.name}") from None
            if not isinstance(decoded, dict):
                raise DataIntegrityError(f"[DATA] malformed bundle line at {path.name}")
            try:
                bundle = _bundle_from_json(decoded, now=now)
            except (ValueError, ArithmeticError, TypeError, DomainValidationError):
                raise DataIntegrityError("[DATA] malformed bundle values") from None
            key = bundle.item_id
            if key in out and out[key].lane != bundle.lane:
                key = f"{bundle.item_id}:{bundle.lane.value}"
            if key in out and out[key] != bundle:
                raise DataIntegrityError("[DATA] conflicting bundles for one item")
            out[key] = bundle
    return out
