"""Danawa catalog search, product, and price-history parsing."""

from __future__ import annotations

import html as html_module
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from src.savetax.errors import DataIntegrityError, DomainValidationError
from src.savetax.publicweb import visible_text

__all__ = [
    "DanawaHit",
    "DanawaProduct",
    "MallOffer",
    "model_pattern",
    "parse_price_history",
    "parse_product",
    "parse_search",
]

_COUNTRY_RE = re.compile(r"^\([A-Z]{2}\)")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SEP = r"[ \-_/\.]"
_LI_RE = re.compile(
    r"<li\b[^>]*\bdata-product-seq=\"(\d+)\"[^>]*>(.*?)</li>",
    re.IGNORECASE | re.DOTALL,
)
_P_RE = re.compile(
    r"<p\b[^>]*\bclass=\"[^\"]*\bmb-2\b[^\"]*\"[^>]*>(.*?)</p>",
    re.IGNORECASE | re.DOTALL,
)
_PRICE_RE = re.compile(r"([\d,]+)\s*원")
_WS_RE = re.compile(r"\s+")
_LD_RE = re.compile(
    r"<script\b[^>]*type=\"application/ld\+json\"[^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)
_MALL_UL_RE = re.compile(
    r"<ul\b[^>]*class=\"[^\"]*list__mall-price[^\"]*\"[^>]*>(.*?)</ul>",
    re.IGNORECASE | re.DOTALL,
)
_LI_ANY_RE = re.compile(r"<li\b[^>]*>(.*?)</li>", re.IGNORECASE | re.DOTALL)
_ARIA_RE = re.compile(r"aria-label=\"([^\"]+)\"", re.IGNORECASE)
_IMG_ALT_RE = re.compile(r"<img\b[^>]*\balt=\"([^\"]+)\"", re.IGNORECASE | re.DOTALL)
_LOGO_BOX_RE = re.compile(r"<div\b[^>]*class=\"[^\"]*box__logo[^\"]*\"[^>]*>(.*?)</div>", re.IGNORECASE | re.DOTALL)
_LOGO_NAME_RE = re.compile(r"(?:aria-label|alt)=(?:\"([^\"]+)\"|'([^']+)')", re.IGNORECASE)
_NUM_SPAN_RE = re.compile(
    r"<span\b[^>]*class=\"[^\"]*text__num[^\"]*\"[^>]*>(.*?)</span>",
    re.IGNORECASE | re.DOTALL,
)
_REG_MONTH_RE = re.compile(r"등록월\s*[:：]\s*([0-9]{4}[-.][0-9]{2})")  # noqa: RUF001
_REG_MONTH_FALLBACK_RE = re.compile(r"등록월\s*[:：]\s*([^<>\s]+)")  # noqa: RUF001


@dataclass(frozen=True)
class DanawaHit:
    """One catalog row of a Danawa search page; price in KRW."""

    pcode: str
    name: str
    category: str | None
    price_krw: int | None


@dataclass(frozen=True)
class MallOffer:
    """One seller row on a catalog page; prices in KRW, shipping None when not stated as a number."""

    mall: str
    price_krw: int
    shipping_krw: int | None


@dataclass(frozen=True)
class DanawaProduct:
    """Structured fields of one catalog page; `low_price_krw` is None when the product is not on sale."""

    pcode: str
    name: str
    brand: str | None
    description: str
    low_price_krw: int | None
    offer_count: int | None
    offers: tuple[MallOffer, ...]
    registered_month: str | None


def _strip_country(model: str) -> str:
    return _COUNTRY_RE.sub("", model, count=1)


def model_pattern(model: str) -> re.Pattern[str]:
    """Return a bounded, separator-tolerant pattern for a model code; raises DomainValidationError for an empty code."""
    stripped = _strip_country(model.strip())
    tokens = _TOKEN_RE.findall(stripped)
    if not tokens:
        raise DomainValidationError("[DATA] invalid model: empty code")
    core = (f"{_SEP}*").join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<![A-Za-z0-9]){core}(?![A-Za-z0-9])(?![\-\+/][A-Za-z0-9])", re.IGNORECASE)


def _clean(text: str) -> str:
    return visible_text(text)


def _parse_int(text: str) -> int | None:
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    return int(digits)


def _category_after(row_html: str) -> str | None:
    pos = row_html.find("상품분류")
    if pos < 0:
        return None
    tail = row_html[pos + len("상품분류") :]
    match = re.search(r">\s*([^<>\s][^<>]*?)\s*<", tail)
    if match:
        value = _WS_RE.sub(" ", html_module.unescape(match.group(1))).strip()
        return value or None
    visible = _clean(tail).lstrip(" :：\t")  # noqa: RUF001
    return visible.split(" ")[0].strip() or None if visible else None


def parse_search(html: str) -> list[DanawaHit]:
    """Return the catalog rows of a search page in page order."""
    hits: list[DanawaHit] = []
    for match in _LI_RE.finditer(html):
        pcode, body = match.group(1), match.group(2)
        name_match = _P_RE.search(body)
        if name_match is None:
            continue
        name = _clean(name_match.group(1))
        if not name:
            continue
        after = body[name_match.end() :]
        price: int | None = None
        price_match = _PRICE_RE.search(_clean(after))
        if price_match is not None:
            price = _parse_int(price_match.group(1))
        hits.append(
            DanawaHit(
                pcode=pcode,
                name=name,
                category=_category_after(body),
                price_krw=price,
            )
        )
    return hits


def _product_block(html: str) -> dict[str, Any]:
    for match in _LD_RE.finditer(html):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            decoded = json.loads(raw)
        except ValueError:
            continue
        candidates = decoded if isinstance(decoded, list) else [decoded]
        for entry in candidates:
            if isinstance(entry, dict) and entry.get("@type") == "Product":
                return entry
    raise DataIntegrityError("[DATA] malformed product page: missing JSON-LD product")


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        amount = Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount <= 0 or amount != amount.to_integral_value():
        return None
    return int(amount)


def _offer_rows(html: str) -> tuple[MallOffer, ...]:
    offers: list[MallOffer] = []
    for ul in _MALL_UL_RE.finditer(html):
        for item in _LI_ANY_RE.finditer(ul.group(1)):
            body = item.group(0)
            # The mall is the logo (image alt for open markets, text label for stores); the row's own aria-label is
            # the generic "상품보기" link and names no seller.
            logo = _LOGO_BOX_RE.search(body)
            logo_name = _LOGO_NAME_RE.search(logo.group(1)) if logo else None
            mall_match = _ARIA_RE.search(body)
            mall = (logo_name.group(1) or logo_name.group(2)) if logo_name else None
            if not mall:
                mall = mall_match.group(1).strip() if mall_match else None
            if not mall:
                alt_match = _IMG_ALT_RE.search(body)
                mall = alt_match.group(1).strip() if alt_match else None
            if not mall:
                continue
            mall = _WS_RE.sub(" ", html_module.unescape(mall)).strip()
            if not mall:
                continue
            price_match = _NUM_SPAN_RE.search(body)
            if price_match is None:
                continue
            price = _parse_int(price_match.group(1))
            if price is None:
                continue
            offers.append(
                MallOffer(
                    mall=mall,
                    price_krw=price,
                    shipping_krw=_shipping_for_row(body[price_match.end() :]),
                )
            )
    return tuple(offers)


def _shipping_for_row(body: str) -> int | None:
    visible = _clean(body)
    if "무료배송" in visible:
        return 0
    for match in _PRICE_RE.finditer(visible):
        return _parse_int(match.group(1))
    return None


def parse_product(html: str, pcode: str) -> DanawaProduct:
    """Parse a catalog page; raises DataIntegrityError when the JSON-LD product block is missing."""
    block = _product_block(html)
    name_raw = block.get("name")
    name = str(name_raw).strip() if isinstance(name_raw, str) else ""
    brand_raw = block.get("brand")
    brand: str | None = None
    if isinstance(brand_raw, dict):
        inner = brand_raw.get("name")
        brand = str(inner).strip() or None if isinstance(inner, str) else None
    elif isinstance(brand_raw, str):
        brand = brand_raw.strip() or None
    desc_raw = block.get("description")
    description = str(desc_raw) if isinstance(desc_raw, str) else ""
    offers_raw = block.get("offers")
    low_price: int | None = None
    offer_count: int | None = None
    if isinstance(offers_raw, dict):
        low_price = _to_int(offers_raw.get("lowPrice"))
        offer_count = _to_int(offers_raw.get("offerCount"))
    offers = _offer_rows(html)
    text = _clean(html)
    month_match = _REG_MONTH_RE.search(text)
    if month_match is not None:
        registered = month_match.group(1).replace(".", "-")
    else:
        fallback = _REG_MONTH_FALLBACK_RE.search(text)
        registered = fallback.group(1).strip() if fallback is not None else None
    return DanawaProduct(
        pcode=pcode,
        name=name,
        brand=brand,
        description=description,
        low_price_krw=low_price,
        offer_count=offer_count,
        offers=offers,
        registered_month=registered,
    )


def parse_price_history(payload: str) -> dict[str, int]:
    """Return monthly lowest prices keyed `YYYY-MM`; raises DataIntegrityError for malformed JSON."""
    try:
        decoded = json.loads(payload)
    except ValueError:
        raise DataIntegrityError("[DATA] malformed price history: not JSON") from None
    if not isinstance(decoded, dict):
        raise DataIntegrityError("[DATA] malformed price history: expected an object")
    if "_error" in decoded:
        raise DataIntegrityError("[DATA] malformed price history: error payload")
    merged: dict[str, int] = {}
    for key in ("24", "12"):
        section = decoded.get(key)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise DataIntegrityError("[DATA] malformed price history: expected range objects")
        result = section.get("result")
        if result is None:
            continue
        if not isinstance(result, list):
            raise DataIntegrityError("[DATA] malformed price history: expected result lists")
        for entry in result:
            if not isinstance(entry, dict):
                continue
            date_raw = entry.get("date")
            value_raw = entry.get("minPrice")
            if not isinstance(date_raw, str):
                continue
            date_match = re.fullmatch(r"(\d{2})-(\d{2})", date_raw.strip())
            if date_match is None:
                continue
            month = int(date_match.group(2))
            if not 1 <= month <= 12:
                continue
            value = _to_int(value_raw)
            if value is None:
                continue
            merged[f"20{date_match.group(1)}-{month:02d}"] = value
    return merged
