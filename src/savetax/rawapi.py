"""Read-only client for the raw data API."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from src.savetax.config import SaveTaxConfig
from src.savetax.errors import ApiError, ConfigError, DataIntegrityError
from src.savetax.http import HttpClient
from src.savetax.redaction import redact

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")


@dataclass(frozen=True)
class RateLimit:
    """Budget reported by the raw data API response headers."""

    limit: int
    remaining: int
    reset_seconds: int


@dataclass(frozen=True)
class ListingDetail:
    """One procurement listing with delivery conditions absent from the dataset."""

    listing_id: str
    item_id: str
    unit: str | None
    price_krw: Decimal
    price_kind: str
    contract_no: str
    contract_type: str | None
    contract_begin: date | None
    contract_end: date | None
    vendor_name: str | None
    manufacturer: str | None
    model: str | None
    category: str | None
    delivery_terms: str | None
    supply_region: str | None
    release: str | None


def _opt_text(data: Mapping[str, Any], field: str) -> str | None:
    if field not in data:
        return None
    value = data[field]
    if value is None:
        return None
    if not isinstance(value, str):
        raise DataIntegrityError(f"[DATA] malformed listing at {field}: expected a string or null")
    if value.strip() == "":
        return None
    return value


def _req_text(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise DataIntegrityError(f"[DATA] malformed listing at {field}: expected a non-empty string")
    return value


def _parse_price(value: Any) -> Decimal:
    if not isinstance(value, str):
        raise DataIntegrityError("[DATA] malformed listing at price: expected a numeric string")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation:
        raise DataIntegrityError("[DATA] malformed listing at price: not a decimal") from None
    if not parsed.is_finite():
        raise DataIntegrityError("[DATA] malformed listing at price: non-finite amount")
    return parsed


def _parse_ymd(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        return None
    try:
        return date(int(value[0:4]), int(value[4:6]), int(value[6:8]))
    except ValueError:
        return None


def _nested_error(status: int, body: bytes) -> ApiError:
    try:
        decoded = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        decoded = None
    code = "NON_JSON_ERROR"
    message = "non-JSON error response"
    if isinstance(decoded, dict):
        error = decoded.get("error")
        if isinstance(error, dict):
            raw_code = error.get("code")
            raw_message = error.get("message")
            if isinstance(raw_code, str) and raw_code:
                code = raw_code
            if isinstance(raw_message, str):
                message = raw_message
    retryable = status == 429 or 500 <= status <= 599
    return ApiError("raw", status, code, message, retryable)


def _int_header(headers: Mapping[str, str], name: str) -> int | None:
    try:
        return int(str(headers.get(name, "")).strip())
    except (ValueError, TypeError):
        return None


def _float_header(headers: Mapping[str, str], name: str) -> float | None:
    try:
        parsed = float(str(headers.get(name, "")).strip())
    except (ValueError, TypeError):
        return None
    if parsed != parsed or parsed == float("inf") or parsed == float("-inf"):
        return None
    return parsed


class RawApiClient:
    """Read-only client for `https://api.savetax.now/v1`."""

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
        self._rate_limit: RateLimit | None = None

    @property
    def last_rate_limit(self) -> RateLimit | None:
        """Return the budget parsed from the most recent response headers."""
        return self._rate_limit

    def get_listing(self, listing_id: str) -> ListingDetail | None:
        """Fetch one listing; returns None when the catalog does not serve it."""
        if not _ID_RE.fullmatch(listing_id):
            raise ConfigError(f"[SYS] invalid listing_id: {listing_id!r} must match ^[A-Za-z0-9_]{{1,40}}$")
        url = f"{self._config.raw_api_base_url.rstrip('/')}/listings/{listing_id}"
        first = self._http.get(url, authenticated=True)
        self._observe_rate_limit(first.headers)
        if first.status == 200:
            return self._parse_listing(listing_id, first.body)
        if first.status == 404 and _nested_code(first.body) == "not_found":
            self._log(listing_id, None)
            return None
        if first.status != 429:
            raise _nested_error(first.status, first.body)
        self._sleep(self._throttle_wait(first.headers))
        second = self._http.get(url, authenticated=True)
        self._observe_rate_limit(second.headers)
        if second.status == 200:
            return self._parse_listing(listing_id, second.body)
        if second.status == 404 and _nested_code(second.body) == "not_found":
            self._log(listing_id, None)
            return None
        raise _nested_error(second.status, second.body)

    def _throttle_wait(self, headers: Mapping[str, str]) -> float:
        retry_after = _float_header(headers, "retry-after")
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self._config.rate_limit_max_wait_seconds)
        reset = _int_header(headers, "x-ratelimit-reset")
        if reset is not None and reset >= 0:
            return float(min(reset, self._config.rate_limit_max_wait_seconds))
        return min(self._config.retry_cooldown_seconds, self._config.rate_limit_max_wait_seconds)

    def _observe_rate_limit(self, headers: Mapping[str, str]) -> None:
        limit = _int_header(headers, "x-ratelimit-limit")
        remaining = _int_header(headers, "x-ratelimit-remaining")
        reset = _int_header(headers, "x-ratelimit-reset")
        if limit is not None and remaining is not None and reset is not None:
            self._rate_limit = RateLimit(limit=limit, remaining=remaining, reset_seconds=reset)
        else:
            self._rate_limit = None

    def _parse_listing(self, listing_id: str, body: bytes) -> ListingDetail:
        try:
            decoded = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise DataIntegrityError("[DATA] malformed listing: body is not JSON") from None
        if not isinstance(decoded, dict):
            raise DataIntegrityError("[DATA] malformed listing: expected a JSON object")
        data = decoded.get("data")
        if not isinstance(data, dict):
            raise DataIntegrityError("[DATA] malformed listing at data: expected an object")
        meta = decoded.get("meta")
        if meta is not None and not isinstance(meta, dict):
            raise DataIntegrityError("[DATA] malformed listing at meta: expected an object or null")
        release: str | None = None
        if isinstance(meta, dict):
            raw_release = meta.get("release")
            if raw_release is not None:
                if not isinstance(raw_release, str) or not raw_release:
                    raise DataIntegrityError("[DATA] malformed listing at meta.release: expected a string")
                release = raw_release
        contract = data.get("contract")
        if not isinstance(contract, dict):
            raise DataIntegrityError("[DATA] malformed listing at contract: expected an object")
        vendor = data.get("vendor")
        vendor_name: str | None = None
        if vendor is not None:
            if not isinstance(vendor, dict):
                raise DataIntegrityError("[DATA] malformed listing at vendor: expected an object or null")
            vendor_name = _opt_text(vendor, "name")
        detail = ListingDetail(
            listing_id=_req_text(data, "id"),
            item_id=_req_text(data, "itemId"),
            unit=_opt_text(data, "unit"),
            price_krw=_parse_price(data.get("price")),
            price_kind=_req_text(data, "priceKind"),
            contract_no=_req_text(contract, "no"),
            contract_type=_opt_text(contract, "type"),
            contract_begin=_parse_ymd(contract.get("beginYmd")),
            contract_end=_parse_ymd(contract.get("endYmd")),
            vendor_name=vendor_name,
            manufacturer=_opt_text(data, "manufacturer"),
            model=_opt_text(data, "model"),
            category=_opt_text(data, "category"),
            delivery_terms=_opt_text(data, "deliveryTerms"),
            supply_region=_opt_text(data, "supplyRegion"),
            release=release,
        )
        self._log(listing_id, release)
        return detail

    def _log(self, listing_id: str, release: str | None) -> None:
        limit = self._rate_limit
        logger.info(
            "[DATA] listing_id=%s release=%s remaining=%s",
            redact(listing_id),
            redact(release) if release is not None else "unknown",
            limit.remaining if limit is not None else "unknown",
        )


def _nested_code(body: bytes) -> str | None:
    try:
        decoded = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(decoded, dict):
        error = decoded.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            if isinstance(code, str):
                return code
    return None
