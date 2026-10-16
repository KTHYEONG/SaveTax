"""Read-only client for the competition API."""

from __future__ import annotations

import enum
import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from src.savetax.config import SaveTaxConfig
from src.savetax.errors import ApiError, ConfigError, DataIntegrityError
from src.savetax.http import HttpClient, HttpResponse
from src.savetax.redaction import redact

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")
_RETRY_503_CODES = frozenset({"CATALOG_UNAVAILABLE", "REFORM_CHECK_UNAVAILABLE"})
_RETRY_429_CODES = frozenset({"RATE_LIMITED", "NON_JSON_ERROR"})  # edge limiter replies without a JSON body


class ItemState(enum.StrEnum):
    """Claim availability of an item as reported by the competition API."""

    AVAILABLE = "available"
    CLAIMED = "claimed"
    CLAIMED_BY_ME = "claimed_by_me"
    REFORM_FOUND = "reform_found"
    ON_HOLD = "on_hold"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class EventInfo:
    """Current event identity and submission window (UTC)."""

    dataset_release_id: str
    ruleset_version: str
    phase: str
    opens_at: datetime
    closes_at: datetime
    evidence_format: str
    item_catalog: str


@dataclass(frozen=True)
class ParticipantStatus:
    """Public handle and remaining claim budget; `None` means unlimited."""

    handle: str
    registered: bool
    claims_used: int
    claims_today: int
    remaining_claims_total: int | None
    remaining_claims_per_day: int | None


@dataclass(frozen=True)
class LookupResult:
    """Pre-registration availability of one requested id."""

    input: str
    kind: str
    item_id: str | None
    state: ItemState


@dataclass(frozen=True)
class Fact:
    """One delivery-request line with its comparison basis; amounts in KRW."""

    fact_id: str
    listing_id: str
    contract_no: str
    request_no: str
    line_no: str
    request_date: date | None
    unit: str
    tax_basis: str
    basis_id: str
    period_kind: str
    unit_price_krw: Decimal
    quantity: Decimal
    comparable_amount_krw: Decimal
    scorable: bool
    exclusion_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ItemDetail:
    """Frozen item snapshot and its facts; snapshot fields are `None` when withheld."""

    item_id: str
    state: ItemState
    name: str | None
    model: str | None
    manufacturer: str | None
    category: str | None
    source_release: str | None
    facts_total: int | None
    facts_truncated: bool
    facts: tuple[Fact, ...]


def _decode_object(body: bytes, path: str) -> dict[str, Any]:
    try:
        decoded = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise DataIntegrityError(f"[DATA] malformed payload at {path}: body is not JSON") from None
    if not isinstance(decoded, dict):
        raise DataIntegrityError(f"[DATA] malformed payload at {path}: expected a JSON object")
    return decoded


def _require_str(obj: Mapping[str, Any], field: str) -> str:
    value = obj.get(field)
    if not isinstance(value, str) or not value:
        raise DataIntegrityError(f"[DATA] malformed payload at {field}: expected a non-empty string")
    return value


def _require_int(obj: Mapping[str, Any], field: str) -> int:
    value = obj.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataIntegrityError(f"[DATA] malformed payload at {field}: expected an integer")
    return value


def _parse_decimal(value: Any, field: str) -> Decimal:
    if not isinstance(value, str):
        raise DataIntegrityError(f"[DATA] malformed payload at {field}: expected a numeric string")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation:
        raise DataIntegrityError(f"[DATA] malformed payload at {field}: not a decimal") from None
    if not parsed.is_finite():
        raise DataIntegrityError(f"[DATA] malformed payload at {field}: non-finite amount")
    return parsed


def _parse_moment(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)", value
    ):
        raise DataIntegrityError(f"[DATA] malformed payload at {field}: expected RFC 3339 UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise DataIntegrityError(f"[DATA] malformed payload at {field}: bad timestamp") from None
    return parsed.astimezone(UTC)


def _parse_request_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise DataIntegrityError("[DATA] malformed payload at request_date: expected YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise DataIntegrityError("[DATA] malformed payload at request_date: bad calendar date") from None


def _parse_state(value: Any) -> ItemState:
    try:
        return ItemState(str(value))
    except ValueError:
        raise DataIntegrityError(f"[DATA] malformed payload at state: unknown state {value!r}") from None


def _flat_error(service: str, status: int, body: bytes) -> ApiError:
    try:
        decoded = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        decoded = None
    if not isinstance(decoded, dict):
        return ApiError(service, status, "NON_JSON_ERROR", "non-JSON error response", False)
    code = decoded.get("code")
    message = decoded.get("message")
    retryable = decoded.get("retryable")
    return ApiError(
        service,
        status,
        code if isinstance(code, str) and code else "UNKNOWN_ERROR",
        message if isinstance(message, str) and message else "",
        retryable is True,
    )


def _check_id(value: str) -> None:
    if not _ID_RE.fullmatch(value):
        raise ConfigError(f"[SYS] invalid id: {value!r} must match ^[A-Za-z0-9_]{{1,40}}$")


class CompetitionClient:
    """Read-only client for `https://hackathon.savetax.now/v1/events/current`."""

    def __init__(
        self,
        http: HttpClient,
        config: SaveTaxConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http
        self._config = config
        self._sleep = sleep

    def _get(self, url: str, *, authenticated: bool) -> HttpResponse:
        retried: set[int] = set()
        while True:
            response = self._http.get(url, authenticated=authenticated)
            if response.status == 200:
                return response
            err = _flat_error("competition", response.status, response.body)
            if response.status not in retried:
                if response.status == 429 and err.code in _RETRY_429_CODES:
                    retried.add(429)
                    self._sleep(self._config.rate_limit_max_wait_seconds)
                    continue
                if response.status == 503 and err.code in _RETRY_503_CODES:
                    retried.add(503)
                    self._sleep(self._config.retry_cooldown_seconds)
                    continue
            raise err

    def _base(self) -> str:
        return self._config.competition_base_url.rstrip("/")

    def get_event(self) -> EventInfo:
        """Fetch the event without credentials; raises ApiError or DataIntegrityError."""
        resp = self._get(f"{self._base()}/events/current", authenticated=False)
        body = _decode_object(resp.body, "event")
        return EventInfo(
            dataset_release_id=_require_str(body, "dataset_release_id"),
            ruleset_version=_require_str(body, "ruleset_version"),
            phase=_require_str(body, "phase"),
            opens_at=_parse_moment(body.get("opens_at"), "opens_at"),
            closes_at=_parse_moment(body.get("closes_at"), "closes_at"),
            evidence_format=_require_str(body, "evidence_format"),
            item_catalog=_require_str(body, "item_catalog"),
        )

    def get_me(self) -> ParticipantStatus:
        """Fetch the participant's handle and remaining limits."""
        resp = self._get(f"{self._base()}/events/current/me", authenticated=True)
        body = _decode_object(resp.body, "me")
        remaining = body.get("remaining")
        if not isinstance(remaining, dict):
            raise DataIntegrityError("[DATA] malformed payload at remaining: expected an object")
        for field in ("claims_total", "claims_per_day"):
            if field not in remaining:
                raise DataIntegrityError(f"[DATA] missing field remaining.{field}")
        total = remaining["claims_total"]
        per_day = remaining["claims_per_day"]
        for label, val in (("remaining.claims_total", total), ("remaining.claims_per_day", per_day)):
            if val is not None and (isinstance(val, bool) or not isinstance(val, int)):
                raise DataIntegrityError(f"[DATA] malformed payload at {label}: expected int or null")
        handle = body.get("handle")
        if not isinstance(handle, str) or not handle:
            raise DataIntegrityError("[DATA] malformed payload at handle: expected a non-empty string")
        registered = body.get("registered")
        if not isinstance(registered, bool):
            raise DataIntegrityError("[DATA] malformed payload at registered: expected a boolean")
        return ParticipantStatus(
            handle=handle,
            registered=registered,
            claims_used=_require_int(body, "claims_used"),
            claims_today=_require_int(body, "claims_today"),
            remaining_claims_total=total,
            remaining_claims_per_day=per_day,
        )

    def lookup_items(self, ids: Sequence[str]) -> list[LookupResult]:
        """Return availability for every id in input order; raises ConfigError on an invalid id."""
        items = list(ids)
        if not items:
            return []
        for value in items:
            _check_id(value)
        ordered: dict[str, LookupResult] = {}
        for start in range(0, len(items), 100):
            chunk = items[start : start + 100]
            url = f"{self._base()}/events/current/items/lookup?ids={','.join(chunk)}"
            resp = self._get(url, authenticated=True)
            body = _decode_object(resp.body, "lookup")
            entries = body.get("items")
            if not isinstance(entries, list):
                raise DataIntegrityError("[DATA] malformed payload at items: expected a list")
            by_input: dict[str, dict[str, Any]] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    raise DataIntegrityError("[DATA] malformed payload at items: expected objects")
                raw_input = entry.get("input")
                if not isinstance(raw_input, str):
                    raise DataIntegrityError("[DATA] malformed payload at items.input: expected a string")
                if raw_input in by_input and by_input[raw_input] != entry:
                    raise DataIntegrityError("[DATA] malformed payload at items.input: conflicting duplicate")
                by_input.setdefault(raw_input, entry)
            for wanted in chunk:
                entry = by_input.get(wanted)
                if entry is None:
                    raise DataIntegrityError(f"[DATA] malformed payload at items: missing id {wanted!r}")
                kind = entry.get("kind")
                if not isinstance(kind, str) or not kind:
                    raise DataIntegrityError("[DATA] malformed payload at items.kind: expected a string")
                raw_item = entry.get("item_id")
                if raw_item is not None and not isinstance(raw_item, str):
                    raise DataIntegrityError("[DATA] malformed payload at items.item_id: expected a string or null")
                ordered[wanted] = LookupResult(
                    input=wanted,
                    kind=kind,
                    item_id=raw_item,
                    state=_parse_state(entry.get("state")),
                )
        counts: dict[str, int] = {}
        for wanted in items:
            result = ordered[wanted]
            counts[result.state.value] = counts.get(result.state.value, 0) + 1
        histogram = ",".join(f"{name}:{counts[name]}" for name in sorted(counts))
        logger.info("[REPORT] lookup n=%d states=%s", len(items), histogram)
        return [ordered[wanted] for wanted in items]

    def get_item(self, item_id: str) -> ItemDetail:
        """Fetch one item's snapshot and facts; raises ApiError or DataIntegrityError."""
        _check_id(item_id)
        resp = self._get(f"{self._base()}/events/current/items/{item_id}", authenticated=True)
        body = _decode_object(resp.body, "item")
        raw_item = body.get("item_id")
        if not isinstance(raw_item, str) or not raw_item:
            raise DataIntegrityError("[DATA] malformed payload at item_id: expected a string")
        availability = body.get("availability")
        if not isinstance(availability, dict):
            raise DataIntegrityError("[DATA] malformed payload at availability: expected an object")
        state = _parse_state(availability.get("state"))
        if "snapshot" not in body:
            raise DataIntegrityError("[DATA] missing field snapshot")
        snapshot = body["snapshot"]
        if snapshot is not None and not isinstance(snapshot, dict):
            raise DataIntegrityError("[DATA] malformed payload at snapshot: expected an object or null")
        raw_facts = body.get("facts")
        if not isinstance(raw_facts, list):
            raise DataIntegrityError("[DATA] malformed payload at facts: expected a list")
        facts = tuple(self._parse_fact(entry) for entry in raw_facts) if snapshot is not None else ()
        if snapshot is None:
            detail = ItemDetail(
                item_id=raw_item,
                state=state,
                name=None,
                model=None,
                manufacturer=None,
                category=None,
                source_release=None,
                facts_total=None,
                facts_truncated=False,
                facts=facts,
            )
        else:
            facts_total = snapshot.get("facts_total")
            if not isinstance(facts_total, int) or isinstance(facts_total, bool):
                raise DataIntegrityError("[DATA] malformed payload at snapshot.facts_total: expected an integer")
            facts_truncated = snapshot.get("facts_truncated")
            if not isinstance(facts_truncated, bool):
                raise DataIntegrityError("[DATA] malformed payload at snapshot.facts_truncated: expected a boolean")
            detail = ItemDetail(
                item_id=raw_item,
                state=state,
                name=self._opt_snapshot_text(snapshot, "name"),
                model=self._opt_snapshot_text(snapshot, "model"),
                manufacturer=self._opt_snapshot_text(snapshot, "manufacturer"),
                category=self._opt_snapshot_text(snapshot, "category"),
                source_release=self._opt_snapshot_text(snapshot, "source_release"),
                facts_total=facts_total,
                facts_truncated=facts_truncated,
                facts=facts,
            )
        scorable = sum(1 for fact in detail.facts if fact.scorable)
        logger.info(
            "[DATA] item_id=%s facts=%d scorable=%d truncated=%s",
            redact(detail.item_id),
            len(detail.facts),
            scorable,
            detail.facts_truncated,
        )
        return detail

    @staticmethod
    def _opt_snapshot_text(snapshot: Mapping[str, Any], field: str) -> str | None:
        if field not in snapshot:
            raise DataIntegrityError(f"[DATA] missing field snapshot.{field}")
        value = snapshot[field]
        if value is None:
            return None
        if not isinstance(value, str):
            raise DataIntegrityError(f"[DATA] malformed payload at snapshot.{field}: expected a string or null")
        return value

    @staticmethod
    def _parse_fact(entry: Any) -> Fact:
        if not isinstance(entry, dict):
            raise DataIntegrityError("[DATA] malformed payload at facts: expected objects")
        basis = _require_fact_str(entry, "basis_id")
        if "request_date" not in entry:
            raise DataIntegrityError("[DATA] missing field facts.request_date")
        reasons = entry.get("exclusion_reasons")
        if not isinstance(reasons, list) or any(not isinstance(r, str) for r in reasons):
            raise DataIntegrityError("[DATA] malformed payload at facts.exclusion_reasons: expected a string list")
        scorable = entry.get("scorable")
        if not isinstance(scorable, bool):
            raise DataIntegrityError("[DATA] malformed payload at facts.scorable: expected a boolean")
        return Fact(
            fact_id=_require_fact_str(entry, "fact_id"),
            listing_id=_require_fact_str(entry, "listing_id"),
            contract_no=_require_fact_str(entry, "contract_no"),
            request_no=_require_fact_str(entry, "request_no"),
            line_no=_require_fact_str(entry, "line_no"),
            request_date=_parse_request_date(entry.get("request_date")),
            unit=_require_fact_str(entry, "unit"),
            tax_basis=_require_fact_str(entry, "tax_basis"),
            basis_id=basis,
            period_kind=_require_fact_str(entry, "period_kind"),
            unit_price_krw=_parse_decimal(entry.get("unit_price_krw"), "facts.unit_price_krw"),
            quantity=_parse_decimal(entry.get("quantity"), "facts.quantity"),
            comparable_amount_krw=_parse_decimal(
                entry.get("comparable_amount_krw"), "facts.comparable_amount_krw"
            ),
            scorable=scorable,
            exclusion_reasons=tuple(reasons),
        )


def _require_fact_str(entry: Mapping[str, Any], field: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise DataIntegrityError(f"[DATA] malformed payload at facts.{field}: expected a non-empty string")
    return value
