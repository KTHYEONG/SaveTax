"""Public listing delivery-row table parsing."""

from __future__ import annotations

import html as html_module
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from src.savetax.errors import ConfigError, DataIntegrityError

__all__ = ["DeliveryPage", "PageRow", "delivery_page_url", "parse_delivery_rows"]

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_CODE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_TR_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr\s*>", re.IGNORECASE | re.DOTALL)
_TD_RE = re.compile(r"<td\b[^>]*>(.*?)</td\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"\s+")
# Older requests carry a bare 10-digit number instead of the R-prefixed form.
_REQ_NO_RE = re.compile(r"^(?:R\d{2}T[A-Z]\d{6,}|\d{10})$")
_DATE_RE = re.compile(r"^\s*(\d{4})\.(\d{1,2})\.(\d{1,2})\s*$")
_NUMBER = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
_QTY_RE = re.compile(rf"^({_NUMBER}(?:\.[0-9]+)?)\s*([^\d\s.,+-][^\d]*)$")
_WON_RE = re.compile(rf"^({_NUMBER})\s*원$")
_PAGER_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*쪽")


@dataclass(frozen=True)
class PageRow:
    """One delivery row printed on the public listing page; amounts in KRW."""

    agency_name: str
    request_date: date
    quantity: Decimal
    unit: str
    unit_price_krw: int
    amount_krw: int
    request_no: str
    text: str


@dataclass(frozen=True)
class DeliveryPage:
    """Rows of one public page and the number of pages of the row table."""

    rows: tuple[PageRow, ...]
    page_count: int


def _cell_text(inner: str) -> str:
    stripped = _TAG_RE.sub("", inner)
    unescaped = html_module.unescape(stripped)
    return _WS_RE.sub(" ", unescaped).strip()


def _parse_page_date(value: str) -> date:
    match = _DATE_RE.fullmatch(value)
    if match is None:
        raise DataIntegrityError("[DATA] malformed delivery row on page")
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        raise DataIntegrityError("[DATA] malformed delivery row on page") from None


def _parse_quantity(value: str) -> tuple[Decimal, str]:
    match = _QTY_RE.fullmatch(value)
    if match is None:
        raise DataIntegrityError("[DATA] malformed delivery row on page")
    quantity = Decimal(match.group(1).replace(",", ""))
    unit = _WS_RE.sub(" ", match.group(2)).strip()
    return quantity, unit


def _parse_won(value: str) -> int:
    match = _WON_RE.fullmatch(value)
    if match is None:
        raise DataIntegrityError("[DATA] malformed delivery row on page")
    return int(match.group(1).replace(",", ""))


def parse_delivery_rows(html: str) -> DeliveryPage:
    """Parse the delivery-row table of a public listing page; raises DataIntegrityError on a malformed row."""
    cleaned = _CODE_RE.sub("", _COMMENT_RE.sub("", html))
    rows: list[PageRow] = []
    for tr_match in _TR_RE.finditer(cleaned):
        cells = [_cell_text(inner) for inner in _TD_RE.findall(tr_match.group(1))]
        if len(cells) != 6:
            continue
        if not _REQ_NO_RE.fullmatch(cells[5]):
            continue
        request_date = _parse_page_date(cells[1])
        quantity, unit = _parse_quantity(cells[2])
        unit_price = _parse_won(cells[3])
        amount = _parse_won(cells[4])
        rows.append(
            PageRow(
                agency_name=cells[0],
                request_date=request_date,
                quantity=quantity,
                unit=unit,
                unit_price_krw=unit_price,
                amount_krw=amount,
                request_no=cells[5],
                text=" ".join(cells),
            )
        )
    visible = _WS_RE.sub(" ", html_module.unescape(_TAG_RE.sub(" ", cleaned))).strip()
    match = _PAGER_RE.search(visible)
    if match is None:
        page_count = 1
    else:
        page_count = int(match.group(2))
        if page_count < 1:
            raise DataIntegrityError("[DATA] malformed delivery pager on page")
    return DeliveryPage(rows=tuple(rows), page_count=page_count)


def delivery_page_url(template: str, listing_id: str, page: int) -> str:
    """Return the public URL of one page of the delivery-row table."""
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ConfigError("[SYS] invalid delivery page: must be >= 1")
    base = template.format(listing_id=listing_id).split("#", 1)[0]
    if page == 1:
        return base
    return f"{base}?dp={page}"
