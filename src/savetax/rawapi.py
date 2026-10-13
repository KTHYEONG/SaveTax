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
_ITEM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,20}$")


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
    contract_date: date | None = None
    change_order: str | None = None


@dataclass(frozen=True)
class DeliveryRow:
    """One final delivery-request line of a (contract, item); amounts in KRW."""

    request_no: str
    line_no: str
    request_date: date
    agency_name: str
    unit: str | None
    quantity: Decimal
    unit_price_krw: Decimal
    contract_unit_price_krw: Decimal
    amount_krw: Decimal


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


def _parse_contract_date(value: Any) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DataIntegrityError("[DATA] malformed listing at contractTerms.contractDate: expected a date string or null")
    stripped = value.strip()
    if stripped == "":
        return None
    candidates: list[tuple[int, int, int]] = []
    if re.fullmatch(r"\d{8}", stripped):
        candidates.append((int(stripped[0:4]), int(stripped[4:6]), int(stripped[6:8])))
    elif re.fullmatch(r"[0-9]{4}/[0-9]{2}/[0-9]{2}", stripped):
        parts = stripped.split("/")
        candidates.append((int(parts[0]), int(parts[1]), int(parts[2])))
    elif re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", stripped):
        parts = stripped.split("-")
        candidates.append((int(parts[0]), int(parts[1]), int(parts[2])))
    else:
        raise DataIntegrityError(f"[DATA] malformed listing at contractTerms.contractDate: {stripped!r} is not a date")
    try:
        return date(*candidates[0])
    except ValueError:
        raise DataIntegrityError(f"[DATA] malformed listing at contractTerms.contractDate: {stripped!r} is not a date") from None


def _normalize_change_order(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DataIntegrityError(f"[DATA] malformed listing at {field}: expected digits or null")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        if not re.fullmatch(r"[0-9]+", stripped):
            raise DataIntegrityError(f"[DATA] malformed listing at {field}: expected digits or null")
        return stripped.lstrip("0") or "0"
    raise DataIntegrityError(f"[DATA] malformed listing at {field}: expected digits or null")


def _parse_decimal_amount(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise DataIntegrityError(f"[DATA] malformed delivery row at {field}: expected a finite number")
    if isinstance(value, int):
        parsed = Decimal(value)
    elif isinstance(value, (float, Decimal)):
        parsed = Decimal(str(value))
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            raise DataIntegrityError(f"[DATA] malformed delivery row at {field}: not a decimal")
        try:
            parsed = Decimal(stripped)
        except InvalidOperation:
            raise DataIntegrityError(f"[DATA] malformed delivery row at {field}: not a decimal") from None
    else:
        raise DataIntegrityError(f"[DATA] malformed delivery row at {field}: expected a finite number")
    if not parsed.is_finite():
        raise DataIntegrityError(f"[DATA] malformed delivery row at {field}: non-finite amount")
    return parsed


def _parse_delivery_request_date(value: Any) -> date:
    if not isinstance(value, str):
        raise DataIntegrityError("[DATA] malformed delivery row at requestDate: expected YYYY-MM-DD")
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value.strip())
    if match is None:
        raise DataIntegrityError("[DATA] malformed delivery row at requestDate: expected YYYY-MM-DD")
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        raise DataIntegrityError("[DATA] malformed delivery row at requestDate: not a date") from None


def _parse_delivery_row(row: Any) -> DeliveryRow | None:
    if not isinstance(row, dict):
        raise DataIntegrityError("[DATA] malformed delivery row at row: expected an object")
    final = row.get("finalRequest")
    if "finalRequest" in row and final != "Y":
        return None
    request_no = row.get("requestNo")
    if not isinstance(request_no, str) or not request_no.strip():
        raise DataIntegrityError("[DATA] malformed delivery row at requestNo: expected a non-empty string")
    raw_line = row.get("lineNo")
    if isinstance(raw_line, bool):
        raise DataIntegrityError("[DATA] malformed delivery row at lineNo: expected digits")
    if isinstance(raw_line, int):
        line_no = str(raw_line)
    elif isinstance(raw_line, str):
        stripped = raw_line.strip()
        if stripped == "" or not re.fullmatch(r"[0-9]+", stripped):
            raise DataIntegrityError("[DATA] malformed delivery row at lineNo: expected digits")
        line_no = stripped.lstrip("0") or "0"
    else:
        raise DataIntegrityError("[DATA] malformed delivery row at lineNo: expected digits")
    request_date = _parse_delivery_request_date(row.get("requestDate"))
    agency = row.get("agencyName")
    if not isinstance(agency, str):
        raise DataIntegrityError("[DATA] malformed delivery row at agencyName: expected a string")
    unit_raw = row.get("unit")
    if unit_raw is None:
        unit: str | None = None
    elif isinstance(unit_raw, str):
        unit = unit_raw.strip() or None
    else:
        raise DataIntegrityError("[DATA] malformed delivery row at unit: expected text or null")
    quantity = _parse_decimal_amount(row.get("quantity"), "quantity")
    unit_price = _parse_decimal_amount(row.get("unitPrice"), "unitPrice")
    contract_unit_price = _parse_decimal_amount(row.get("contractUnitPrice"), "contractUnitPrice")
    amount = _parse_decimal_amount(row.get("amount"), "amount")
    return DeliveryRow(
        request_no=request_no,
        line_no=line_no,
        request_date=request_date,
        agency_name=agency,
        unit=unit,
        quantity=quantity,
        unit_price_krw=unit_price,
        contract_unit_price_krw=contract_unit_price,
        amount_krw=amount,
    )


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
        terms_raw = data.get("contractTerms")
        if terms_raw is None or "contractTerms" not in data:
            terms: dict[str, Any] | None = None
        elif isinstance(terms_raw, dict):
            terms = terms_raw
        else:
            raise DataIntegrityError("[DATA] malformed listing at contractTerms: expected an object or null")
        if terms is None:
            contract_date: date | None = None
        else:
            contract_date = _parse_contract_date(terms.get("contractDate"))
        terms_co = terms.get("changeOrder") if terms is not None else None
        if terms_co is None or (isinstance(terms_co, str) and terms_co.strip() == ""):
            change_order = _normalize_change_order(contract.get("changeOrder"), "contract.changeOrder")
        else:
            change_order = _normalize_change_order(terms_co, "contractTerms.changeOrder")
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
            contract_date=contract_date,
            change_order=change_order,
        )
        self._log(listing_id, release)
        return detail

    def get_item_attributes(self, item_id: str) -> Mapping[str, str] | None:
        """Fetch labelled attributes of one item from GET /items/{id}; None when the catalog does not serve it."""
        if not _ITEM_ID_RE.fullmatch(item_id):
            raise ConfigError(f"[SYS] invalid item_id: {item_id!r} must match ^[A-Za-z0-9_-]{{1,20}}$")
        url = f"{self._config.raw_api_base_url.rstrip('/')}/items/{item_id}"
        first = self._http.get(url, authenticated=True)
        self._observe_rate_limit(first.headers)
        if first.status == 200:
            return self._parse_item_attributes(item_id, first.body)
        if first.status == 404 and _nested_code(first.body) == "not_found":
            self._log(item_id, None)
            return None
        if first.status != 429:
            raise _nested_error(first.status, first.body)
        self._sleep(self._throttle_wait(first.headers))
        second = self._http.get(url, authenticated=True)
        self._observe_rate_limit(second.headers)
        if second.status == 200:
            return self._parse_item_attributes(item_id, second.body)
        if second.status == 404 and _nested_code(second.body) == "not_found":
            self._log(item_id, None)
            return None
        raise _nested_error(second.status, second.body)

    def _parse_item_attributes(self, item_id: str, body: bytes) -> dict[str, str]:
        try:
            decoded = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise DataIntegrityError("[DATA] malformed item: body is not JSON") from None
        if not isinstance(decoded, dict):
            raise DataIntegrityError("[DATA] malformed item: expected a JSON object")
        data = decoded.get("data")
        if not isinstance(data, dict):
            raise DataIntegrityError("[DATA] malformed item at data: expected an object")
        raw_id = data.get("id")
        if not isinstance(raw_id, str) or raw_id != item_id:
            raise DataIntegrityError("[DATA] malformed item at data.id: identity mismatch")
        raw_attrs = data.get("attributes")
        if raw_attrs is None:
            return {}
        if not isinstance(raw_attrs, list):
            raise DataIntegrityError("[DATA] malformed item at data.attributes: expected a list")
        out: dict[str, str] = {}
        for entry in raw_attrs:
            if not isinstance(entry, dict):
                raise DataIntegrityError("[DATA] malformed item at data.attributes: expected objects")
            name = entry.get("name")
            value = entry.get("value")
            if not isinstance(name, str) or not isinstance(value, str):
                continue
            name_s = name.strip()
            value_s = value.strip()
            if not name_s or not value_s:
                continue
            if name_s in out and out[name_s] != value_s:
                raise DataIntegrityError("[DATA] malformed item at data.attributes: conflicting duplicate")
            out.setdefault(name_s, value_s)
        self._log(item_id, None)
        return out

    def list_deliveries(self, listing_id: str) -> tuple[DeliveryRow, ...] | None:
        """Fetch every final delivery row of the listing's (contract, item); None when the listing is not served."""
        if not _ID_RE.fullmatch(listing_id):
            raise ConfigError(f"[SYS] invalid listing_id: {listing_id!r} must match ^[A-Za-z0-9_]{{1,40}}$")
        base = self._config.raw_api_base_url.rstrip("/")
        page_size = self._config.delivery_page_size
        cap = self._config.delivery_rows_max
        out: list[DeliveryRow] = []
        seen: set[tuple[str, str]] = set()
        offset = 0
        first = True
        while True:
            url = f"{base}/listings/{listing_id}/deliveries?final=1&limit={page_size}&offset={offset}"
            response = self._http.get(url, authenticated=True)
            self._observe_rate_limit(response.headers)
            if response.status == 429:
                self._sleep(self._throttle_wait(response.headers))
                response = self._http.get(url, authenticated=True)
                self._observe_rate_limit(response.headers)
                if response.status != 200:
                    if first and response.status == 404 and _nested_code(response.body) == "not_found":
                        self._log_deliveries(listing_id, 0)
                        return None
                    raise _nested_error(response.status, response.body)
            elif response.status != 200:
                if first and response.status == 404 and _nested_code(response.body) == "not_found":
                    self._log_deliveries(listing_id, 0)
                    return None
                raise _nested_error(response.status, response.body)
            data = self._decode_deliveries(response.body)
            if first and data.get("hasDeliveries") is False:
                self._log_deliveries(listing_id, 0)
                return ()
            raw_rows = data.get("rows")
            if not isinstance(raw_rows, list):
                raise DataIntegrityError("[DATA] malformed delivery row at rows: expected a list")
            if first and len(raw_rows) == 0:
                self._log_deliveries(listing_id, 0)
                return ()
            received = len(raw_rows)
            if offset + received > cap:
                raise DataIntegrityError("[DATA] delivery rows exceed limit")
            for entry in raw_rows:
                parsed = _parse_delivery_row(entry)
                if parsed is None:
                    continue
                key = (parsed.request_no, parsed.line_no)
                if key in seen:
                    raise DataIntegrityError("[DATA] malformed delivery row at requestNo: duplicate order line")
                seen.add(key)
                out.append(parsed)
            page = data.get("page")
            if not isinstance(page, dict) or not isinstance(page.get("hasMore"), bool):
                raise DataIntegrityError("[DATA] malformed delivery row at page.hasMore")
            has_more = page["hasMore"]
            if not has_more:
                break
            if received == 0:
                raise DataIntegrityError("[DATA] malformed delivery row at page: no paging progress")
            offset += received
            first = False
        self._log_deliveries(listing_id, len(out))
        return tuple(out)

    @staticmethod
    def _decode_deliveries(body: bytes) -> dict[str, Any]:
        try:
            decoded = json.loads(body, parse_float=Decimal)
        except (ValueError, UnicodeDecodeError):
            raise DataIntegrityError("[DATA] malformed delivery row at data: body is not JSON") from None
        if not isinstance(decoded, dict):
            raise DataIntegrityError("[DATA] malformed delivery row at data: expected a JSON object")
        data = decoded.get("data")
        if not isinstance(data, dict):
            raise DataIntegrityError("[DATA] malformed delivery row at data: expected an object")
        return data

    def _log_deliveries(self, listing_id: str, count: int) -> None:
        limit = self._rate_limit
        logger.info(
            "[DATA] deliveries listing_id=%s rows=%d remaining=%s",
            redact(listing_id),
            count,
            limit.remaining if limit is not None else "unknown",
        )

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
