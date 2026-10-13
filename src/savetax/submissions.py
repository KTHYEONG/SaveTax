"""Idempotent submission workflow over the write and read transports."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from src.savetax.competition import CompetitionClient
from src.savetax.config import SaveTaxConfig
from src.savetax.errors import ApiError, ConfigError, DataIntegrityError, NetworkError, SubmissionPending
from src.savetax.http import HttpClient
from src.savetax.redaction import redact
from src.savetax.writehttp import WriteHttpClient

logger = logging.getLogger(__name__)

_SID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_KEY_RE = re.compile(r"^[\x21-\x7e]{1,128}$")
_NEXT_ACTIONS = frozenset({"submit", "revise_existing", "pick_another_product"})
_LINT_STATUSES = frozenset({"pending", "clean", "suspect", "unreadable"})
_NON_RETRYABLE = frozenset(
    {
        "REFORM_ALREADY_FOUND",
        "PRODUCT_ALREADY_CLAIMED",
        "PARTICIPANT_LIMIT",
        "REVISION_LIMIT",
        "REVISION_CONFLICT",
        "SUBMISSION_NOT_EDITABLE",
        "IDEMPOTENCY_KEY_REUSED",
        "NO_POSITIVE_AMOUNT",
        "FACT_NOT_SCORABLE",
        "FACT_PRODUCT_MISMATCH",
        "COMPARISON_BASIS_MISMATCH",
        "DATASET_VERSION_MISMATCH",
        "SCHEMA_INVALID",
        "BODY_TOO_LARGE",
        "IF_MATCH_REQUIRED",
        "IF_MATCH_INVALID",
    }
)


@dataclass(frozen=True)
class PreviewResult:
    """Server-side dry run outcome for a draft body."""

    ok: bool
    total_amount_krw: int
    line_amounts_krw: Mapping[str, int]
    next_action: str
    product_already_claimed: bool
    claimed_by_me: bool
    existing_submission_id: str | None
    current_revision: int | None


@dataclass(frozen=True)
class Receipt:
    """Acceptance receipt of a submission or revision."""

    submission_id: str
    claim_id: str | None
    revision: int
    submission_state: str
    admitted_at: datetime | None
    amount_krw: int
    lint_status: str
    api_path: str | None


@dataclass(frozen=True)
class LintHint:
    """One lint hint attached to a submission."""

    fact_id: str | None
    code: str
    message: str


@dataclass(frozen=True)
class LintLine:
    """One lint verdict attached to a fact."""

    fact_id: str
    verdict: str
    reason: str


@dataclass(frozen=True)
class SubmissionDetail:
    """Own submission with lint outcome."""

    submission_id: str
    item_id: str | None
    state: str
    revision: int
    amount_krw: int | None
    held: bool
    lint_status: str
    lint_lines: tuple[LintLine, ...]
    lint_hints: tuple[LintHint, ...]


@dataclass(frozen=True)
class RequestOutcome:
    """Resolution of an idempotency key."""

    key: str
    state: str
    receipt: Receipt | None


def _decode_object(body: bytes, field: str) -> dict[str, Any]:
    try:
        decoded = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise DataIntegrityError(f"[DATA] malformed payload at {field}: body is not JSON") from None
    if not isinstance(decoded, dict):
        raise DataIntegrityError(f"[DATA] malformed payload at {field}: expected a JSON object")
    return decoded


def _flat_error(service: str, status: int, body: bytes) -> ApiError:
    try:
        decoded = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        decoded = None
    if not isinstance(decoded, dict):
        return ApiError(service, status, "UNKNOWN_ERROR", "non-JSON error response", False)
    raw_code = decoded.get("code")
    code = raw_code if isinstance(raw_code, str) and raw_code else "UNKNOWN_ERROR"
    raw_message = decoded.get("message")
    message = raw_message if isinstance(raw_message, str) else ""
    retryable = decoded.get("retryable") is True
    request_id = decoded.get("request_id")
    context = decoded.get("context")
    return ApiError(
        service,
        status,
        code,
        message,
        retryable,
        request_id=request_id if isinstance(request_id, str) else None,
        context=context if isinstance(context, dict) else None,
    )


def _retryable(status: int, code: str) -> bool:
    if code in _NON_RETRYABLE:
        return False
    if status == 429:
        return code == "RATE_LIMITED"
    return 500 <= status <= 599


def _check_sid(value: object) -> None:
    if not isinstance(value, str) or not _SID_RE.fullmatch(value):
        raise ConfigError("[SYS] invalid submission id")


def _check_key(value: str) -> None:
    if not _KEY_RE.fullmatch(value):
        raise ConfigError("[SYS] invalid idempotency key")


def _parse_krw(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise DataIntegrityError(f"[DATA] malformed payload at {field}: expected an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?[0-9]{1,100}", value):
        return int(value)
    raise DataIntegrityError(f"[DATA] malformed payload at {field}: expected an integer")


def _opt_krw(data: Mapping[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        if name in data and data[name] is not None:
            return _parse_krw(data[name], name)
    return None


def _parse_moment_or_none(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)", value
    ):
        raise DataIntegrityError("[DATA] malformed payload at admitted_at: expected RFC 3339 UTC")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        raise DataIntegrityError("[DATA] malformed payload at admitted_at: bad timestamp") from None


def _parse_receipt(data: Mapping[str, Any], api_path: str | None) -> Receipt:
    raw_sid = data.get("submission_id", data.get("id"))
    if not isinstance(raw_sid, str) or not raw_sid:
        raise DataIntegrityError("[DATA] malformed receipt at submission_id: expected a non-empty string")
    _check_sid(raw_sid)
    raw_rev = data.get("revision", data.get("current_revision"))
    if isinstance(raw_rev, bool) or not isinstance(raw_rev, int) or raw_rev < 1:
        raise DataIntegrityError("[DATA] malformed receipt at revision: expected an integer")
    raw_state = data.get("submission_state", data.get("state"))
    if not isinstance(raw_state, str) or not raw_state:
        raise DataIntegrityError("[DATA] malformed receipt at submission_state: expected a non-empty string")
    raw_claim = data.get("claim_id")
    claim_id = raw_claim if isinstance(raw_claim, str) and raw_claim else None
    amount = _opt_krw(data, ("amount_krw", "amount", "total_amount_krw"))
    if amount is None:
        raise DataIntegrityError("[DATA] malformed receipt at amount_krw: expected an integer")
    lint_raw = data.get("lint_status")
    if lint_raw is None and isinstance(data.get("lint"), dict):
        lint_raw = data["lint"].get("status", data["lint"].get("lint_status"))
    lint_status = lint_raw if isinstance(lint_raw, str) and lint_raw else "pending"
    if lint_status not in _LINT_STATUSES:
        raise DataIntegrityError("[DATA] malformed receipt at lint_status")
    response_path = data.get("api_path", api_path)
    if response_path is not None and not isinstance(response_path, str):
        raise DataIntegrityError("[DATA] malformed receipt at api_path")
    return Receipt(
        submission_id=raw_sid,
        claim_id=claim_id,
        revision=raw_rev,
        submission_state=raw_state,
        admitted_at=_parse_moment_or_none(data.get("admitted_at")),
        amount_krw=amount,
        lint_status=lint_status,
        api_path=response_path,
    )


def _parse_detail(data: Mapping[str, Any]) -> SubmissionDetail:
    if not isinstance(data, Mapping):
        raise DataIntegrityError("[DATA] malformed submission: expected an object")
    raw_sid = data.get("submission_id", data.get("id"))
    if not isinstance(raw_sid, str) or not raw_sid:
        raise DataIntegrityError("[DATA] malformed submission at submission_id: expected a string")
    _check_sid(raw_sid)
    raw_state = data.get("submission_state", data.get("state", ""))
    state = raw_state if isinstance(raw_state, str) else ""
    raw_rev = data.get("current_revision", data.get("revision", 0))
    if isinstance(raw_rev, bool) or not isinstance(raw_rev, int):
        raise DataIntegrityError("[DATA] malformed submission at revision: expected an integer")
    raw_item = data.get("item_id", data.get("canonical_product_id"))
    item_id = raw_item if isinstance(raw_item, str) and raw_item else None
    if item_id is not None:
        _check_sid(item_id)
    amount = None
    if any(k in data for k in ("amount_krw", "amount", "total_amount_krw")):
        amount = _opt_krw(data, ("amount_krw", "amount", "total_amount_krw"))
    lint_raw: Any = data.get("lint_status")
    if lint_raw is None and isinstance(data.get("lint"), dict):
        nested: Any = data["lint"]
        lint_raw = nested.get("status", nested.get("lint_status"))
    if not isinstance(lint_raw, str) or not lint_raw:
        raise DataIntegrityError("[DATA] malformed submission at lint_status: expected a string")
    if lint_raw not in _LINT_STATUSES:
        raise DataIntegrityError(f"[DATA] malformed submission at lint_status: unknown {lint_raw!r}")
    held_flag = data.get("held") is True or (state == "accepted" and lint_raw == "suspect")
    return SubmissionDetail(
        submission_id=raw_sid,
        item_id=item_id,
        state=state,
        revision=raw_rev,
        amount_krw=amount,
        held=held_flag,
        lint_status=lint_raw,
        lint_lines=_parse_lines(data),
        lint_hints=_parse_hints(data),
    )


def _parse_lines(data: Mapping[str, Any]) -> tuple[LintLine, ...]:
    raw: Any = data.get("lint_lines", data.get("lines"))
    if raw is None and isinstance(data.get("lint"), dict):
        raw = data["lint"].get("lines")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise DataIntegrityError("[DATA] malformed submission at lint_lines: expected a list")
    out: list[LintLine] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise DataIntegrityError("[DATA] malformed submission at lint_lines: expected objects")
        fact_id = entry.get("fact_id", entry.get("id"))
        if not isinstance(fact_id, str) or not fact_id:
            raise DataIntegrityError("[DATA] malformed submission at lint_lines.fact_id")
        verdict = entry.get("verdict", "")
        reason = entry.get("reason", "")
        out.append(
            LintLine(
                fact_id=fact_id,
                verdict=verdict if isinstance(verdict, str) else "",
                reason=reason if isinstance(reason, str) else "",
            )
        )
    return tuple(out)


def _parse_hints(data: Mapping[str, Any]) -> tuple[LintHint, ...]:
    raw: Any = data.get("lint_hints", data.get("hints"))
    if raw is None and isinstance(data.get("lint"), dict):
        raw = data["lint"].get("hints")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise DataIntegrityError("[DATA] malformed submission at lint_hints: expected a list")
    out: list[LintHint] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise DataIntegrityError("[DATA] malformed submission at lint_hints: expected objects")
        code = entry.get("code")
        if not isinstance(code, str) or not code:
            raise DataIntegrityError("[DATA] malformed submission at lint_hints.code")
        message = entry.get("message", "")
        fact_id = entry.get("fact_id")
        out.append(
            LintHint(
                fact_id=fact_id if isinstance(fact_id, str) else None,
                code=code,
                message=message if isinstance(message, str) else "",
            )
        )
    return tuple(out)


class SubmissionClient:
    """Idempotent submit, revise, withdraw and read-back over competition transports."""

    def __init__(
        self,
        write: WriteHttpClient,
        read: CompetitionClient,
        http: HttpClient,
        config: SaveTaxConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._write = write
        self._read = read
        self._http = http
        self._config = config
        self._sleep = sleep

    def _base(self) -> str:
        return self._config.competition_base_url.rstrip("/")

    def preview(self, body: Mapping[str, object]) -> PreviewResult:
        """Dry-run a draft body without side effects."""
        if "canonical_product_id" in body:
            _check_sid(body["canonical_product_id"])
        response = self._write.post_json("/submissions/preview", body, idempotency_key="preview")
        if response.status != 200:
            raise _flat_error("competition", response.status, response.body)
        data = _decode_object(response.body, "preview")
        ok = data.get("ok")
        if not isinstance(ok, bool):
            raise DataIntegrityError("[DATA] malformed preview at ok: expected a boolean")
        next_action = data.get("next_action")
        if not isinstance(next_action, str) or next_action not in _NEXT_ACTIONS:
            raise DataIntegrityError("[DATA] malformed preview at next_action: unknown action")
        total = _opt_krw(data, ("total_amount_krw", "total_amount", "amount_krw"))
        if total is None:
            raise DataIntegrityError("[DATA] malformed preview at total_amount_krw: expected an integer")
        lines = self._preview_lines(data)
        claimed = data.get("product_already_claimed") is True
        by_me = data.get("claimed_by_me") is True
        existing = data.get("existing_submission_id")
        if isinstance(existing, str) and existing:
            _check_sid(existing)
        revision_raw = data.get("current_revision")
        result = PreviewResult(
            ok=ok,
            total_amount_krw=total,
            line_amounts_krw=lines,
            next_action=str(next_action),
            product_already_claimed=claimed,
            claimed_by_me=by_me,
            existing_submission_id=existing if isinstance(existing, str) and existing else None,
            current_revision=revision_raw
            if isinstance(revision_raw, int) and not isinstance(revision_raw, bool)
            else None,
        )
        item = body.get("canonical_product_id") if isinstance(body, Mapping) else None
        logger.info(
            "[SUBMIT] preview item=%s ok=%s total=%d next=%s",
            redact(str(item)) if isinstance(item, str) else "unknown",
            ok,
            total,
            next_action,
        )
        return result

    @staticmethod
    def _preview_lines(data: Mapping[str, Any]) -> dict[str, int]:
        raw: Any = data.get("line_amounts_krw", data.get("lines", data.get("line_amounts")))
        if raw is None:
            return {}
        if isinstance(raw, dict):
            parsed: dict[str, int] = {}
            for key, value in raw.items():
                parsed[str(key)] = _parse_krw(value, "lines")
            return parsed
        if isinstance(raw, list):
            parsed_list: dict[str, int] = {}
            for entry in raw:
                if not isinstance(entry, dict):
                    raise DataIntegrityError("[DATA] malformed preview at lines: expected objects")
                fid = entry.get("fact_id", entry.get("id"))
                if not isinstance(fid, str) or not fid:
                    raise DataIntegrityError("[DATA] malformed preview at lines.fact_id")
                amount = _parse_krw(
                    entry.get("score_amount_krw", entry.get("amount_krw", entry.get("amount"))), "lines"
                )
                if fid in parsed_list and parsed_list[fid] != amount:
                    raise DataIntegrityError("[DATA] conflicting preview amounts for one fact")
                parsed_list[fid] = amount
            return parsed_list
        raise DataIntegrityError("[DATA] malformed preview at lines: expected an object or list")

    def submit(self, body: Mapping[str, object], *, idempotency_key: str) -> Receipt:
        """Submit a draft body exactly once per idempotency key."""
        _check_key(idempotency_key)
        for field in ("canonical_product_id", "dataset_release_id", "ruleset_version"):
            value = body.get(field)
            if not isinstance(value, str) or not value:
                raise ConfigError(f"[SYS] invalid submit body: {field} must be a non-empty string")
        _check_sid(body["canonical_product_id"])
        receipt = self._send_with_retry("/submissions", body, idempotency_key, None, 201, "accepted")
        item = body.get("canonical_product_id")
        logger.info(
            "[SUBMIT] submitted item=%s sid=%s revision=%d amount=%d",
            redact(str(item)),
            redact(receipt.submission_id),
            receipt.revision,
            receipt.amount_krw,
        )
        return receipt

    def revise(self, submission_id: str, body: Mapping[str, object], *, if_match: int, idempotency_key: str) -> Receipt:
        """Replace a submission body guarded by its current revision."""
        _check_sid(submission_id)
        _check_key(idempotency_key)
        if "canonical_product_id" in body:
            _check_sid(body["canonical_product_id"])
        if isinstance(if_match, bool) or not isinstance(if_match, int) or if_match < 1:
            raise ConfigError("[SYS] invalid if_match: must be >= 1")
        return self._send_with_retry(
            f"/submissions/{submission_id}/revisions", body, idempotency_key, if_match, 200, None
        )

    def withdraw(self, submission_id: str, *, idempotency_key: str) -> Receipt:
        """Withdraw a submission without sending a body."""
        _check_sid(submission_id)
        _check_key(idempotency_key)
        receipt = self._send_with_retry(
            f"/submissions/{submission_id}/withdraw", None, idempotency_key, None, 200, "withdrawn"
        )
        logger.info("[SUBMIT] withdrawn sid=%s", redact(receipt.submission_id))
        return receipt

    def _send_with_retry(
        self,
        path: str,
        body: Mapping[str, object] | None,
        key: str,
        if_match: int | None,
        expected: int,
        expect_state: str | None,
    ) -> Receipt:
        try:
            frozen_body = (
                json.loads(json.dumps(body, ensure_ascii=False, sort_keys=True, allow_nan=False))
                if body is not None
                else None
            )
        except (TypeError, ValueError):
            raise DataIntegrityError("[DATA] submission body is not JSON serializable") from None
        last: NetworkError | ApiError
        for attempt in range(3):
            try:
                response = self._write.post_json(path, frozen_body, idempotency_key=key, if_match=if_match)
            except NetworkError as exc:
                last = exc
                if attempt < 2:
                    self._sleep(self._config.retry_cooldown_seconds)
                    continue
                break
            if response.status == expected:
                receipt = _parse_receipt(_decode_object(response.body, "receipt"), path)
                if expect_state is not None and receipt.submission_state != expect_state:
                    raise DataIntegrityError(f"[DATA] unexpected receipt state: {redact(receipt.submission_state)}")
                return receipt
            err = _flat_error("competition", response.status, response.body)
            if err.code in _NON_RETRYABLE or not _retryable(response.status, err.code):
                raise err
            last = err
            if attempt < 2:
                self._sleep(self._config.retry_cooldown_seconds)
        outcome = self.resolve_request(key)
        if outcome.state == "receipt" and outcome.receipt is not None:
            if expect_state is not None and outcome.receipt.submission_state != expect_state:
                raise DataIntegrityError("[DATA] unexpected resolved receipt state")
            return outcome.receipt
        if outcome.state == "pending":
            raise SubmissionPending(key)
        raise last

    def get_submission(self, submission_id: str) -> SubmissionDetail:
        """Fetch one own submission with its lint outcome."""
        _check_sid(submission_id)
        url = f"{self._base()}/events/current/submissions/{submission_id}"
        response = self._http.get(url, authenticated=True)
        if response.status != 200:
            raise _flat_error("competition", response.status, response.body)
        detail = _parse_detail(_decode_object(response.body, "submission"))
        logger.info("[REPORT] lint sid=%s status=%s", redact(detail.submission_id), redact(detail.lint_status))
        return detail

    def list_my_submissions(self) -> list[SubmissionDetail]:
        """List own submissions with tolerant parsing."""
        url = f"{self._base()}/events/current/me/submissions"
        response = self._http.get(url, authenticated=True)
        if response.status != 200:
            raise _flat_error("competition", response.status, response.body)
        data = _decode_object(response.body, "submissions")
        raw: Any = data.get("submissions", data.get("items", data.get("data")))
        if raw is None and isinstance(data.get("submission"), dict):
            raw = [data["submission"]]
        if not isinstance(raw, list):
            raise DataIntegrityError("[DATA] malformed payload at submissions: expected a list")
        details = [_parse_detail(entry) for entry in raw]
        for detail in details:
            logger.info("[REPORT] lint sid=%s status=%s", redact(detail.submission_id), redact(detail.lint_status))
        return details

    def resolve_request(self, idempotency_key: str) -> RequestOutcome:
        """Resolve an idempotency key to receipt, pending, or never admitted."""
        _check_key(idempotency_key)
        encoded_key = quote(idempotency_key, safe="").replace(".", "%2E")
        url = f"{self._base()}/events/current/submission-requests/{encoded_key}"
        response = self._http.get(url, authenticated=True)
        if response.status == 200:
            receipt = _parse_receipt(_decode_object(response.body, "submission-request"), None)
            return RequestOutcome(key=idempotency_key, state="receipt", receipt=receipt)
        if response.status == 202:
            return RequestOutcome(key=idempotency_key, state="pending", receipt=None)
        if response.status == 404:
            err = _flat_error("competition", response.status, response.body)
            if err.code == "IDEMPOTENCY_KEY_NOT_FOUND":
                return RequestOutcome(key=idempotency_key, state="not_found", receipt=None)
            raise err
        raise _flat_error("competition", response.status, response.body)
