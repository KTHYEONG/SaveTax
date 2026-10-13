"""Specialty-shop search and product parsing over Cafe24 and Godomall storefronts."""

from __future__ import annotations

import html as html_module
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import quote, urljoin, urlparse

from src.savetax.config import ShopSpec, _check_url
from src.savetax.errors import ConfigError
from src.savetax.publicweb import visible_text

__all__ = [
    "ShopProduct",
    "parse_shop_product",
    "parse_shop_search",
    "shop_search_url",
]

_COUNTRY_RE = re.compile(r"^\([A-Z]{2}\)")
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_META_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r"""(\w[\w:.-]*)\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""")
_TITLE_RE = re.compile(r"<title\b[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_LD_RE = re.compile(
    r"<script\b[^>]*type=\"application/ld\+json\"[^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ShopProduct:
    """One shop product page: title and sale price in KRW."""

    url: str
    title: str
    price_krw: int


def _bare_model(model: str) -> str:
    return _COUNTRY_RE.sub("", model.strip(), count=1)


def shop_search_url(shop: ShopSpec, model: str) -> str:
    """Return the platform search URL for a model code."""
    query = quote(_bare_model(model), safe="")
    if shop.platform == "godomall":
        return f"https://{shop.host}/goods/goods_search.php?keyword={query}"
    return f"https://{shop.host}/product/search.html?keyword={query}"


def _is_product_href(href: str, platform: str) -> bool:
    if platform == "godomall":
        return re.search(r"goods_view\.php\?[^\"'\s]*goodsNo=(\d+)", href) is not None
    if re.search(r"product_no=(\d+)", href) is not None:
        return True
    return re.search(r"/product/[^\"']*/(\d+)/?", href) is not None


def parse_shop_search(html: str, shop: ShopSpec) -> list[str]:
    """Return absolute https product URLs found on a search page, in page order without duplicates."""
    base = f"https://{shop.host}/"
    wanted = shop.host.lower()
    out: list[str] = []
    seen: set[str] = set()
    for match in _HREF_RE.finditer(html):
        raw = html_module.unescape(match.group(1).strip())
        if not raw or raw.startswith("#") or raw.lower().startswith("javascript:"):
            continue
        if not _is_product_href(raw, shop.platform):
            continue
        absolute = urljoin(base, raw).split("#", 1)[0]
        try:
            _check_url(absolute, "shop product URL")
        except ConfigError:
            continue
        parsed = urlparse(absolute)
        if (parsed.hostname or "").lower() != wanted:
            continue
        if absolute not in seen:
            seen.add(absolute)
            out.append(absolute)
    return out


def _meta_map(html: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for tag in _META_RE.finditer(html):
        attrs: dict[str, str] = {}
        for key, raw in _ATTR_RE.findall(tag.group(0)):
            value = raw.strip("\"'")
            attrs[key.lower()] = html_module.unescape(value)
        name = attrs.get("property") or attrs.get("name") or ""
        content = attrs.get("content", "")
        if name and content and name.lower() not in values:
            values[name.lower()] = content.strip()
    return values


def _amount(text: Any) -> int | None:
    if isinstance(text, bool) or not isinstance(text, (int, float, str)):
        return None
    try:
        amount = Decimal(str(text).replace(",", "").strip())
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount <= 0 or amount != amount.to_integral_value():
        return None
    return int(amount)


def _json_ld_price(html: str) -> tuple[int | None, bool]:
    foreign = False
    prices: list[int] = []
    for match in _LD_RE.finditer(html):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            decoded = json.loads(raw)
        except ValueError:
            continue
        for node in _walk(decoded):
            if not isinstance(node, dict) or "price" not in node:
                continue
            currency = node.get("priceCurrency")
            if currency is None:
                continue
            if not isinstance(currency, str):
                continue
            if currency != "KRW":
                foreign = True
                continue
            amount = _amount(node.get("price"))
            if amount is not None:
                prices.append(amount)
    if foreign or len(set(prices)) != 1:
        return None, foreign
    return prices[0], False


def _walk(node: Any) -> Any:
    yield node
    if isinstance(node, dict):
        if "offers" in node:
            yield from _walk(node["offers"])
        if "@graph" in node:
            yield from _walk(node["@graph"])
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _title(html: str, meta: dict[str, str]) -> str | None:
    og_title = meta.get("og:title", "").strip()
    if og_title:
        return re.sub(r"\s+", " ", html_module.unescape(og_title)).strip() or None
    match = _TITLE_RE.search(html)
    if match is None:
        return None
    text = re.sub(r"\s+", " ", visible_text(match.group(0))).strip()
    return text or None


def parse_shop_product(html: str, url: str) -> ShopProduct | None:
    """Return the product when a KRW price and a title are present, else None."""
    meta = _meta_map(html)
    currency = meta.get("product:price:currency")
    if currency is not None and currency != "KRW":
        return None
    currency_ok = currency is None or currency == "KRW"
    price: int | None = None
    if currency_ok:
        price = _amount(meta.get("product:sale_price:amount", ""))
        if price is None:
            price = _amount(meta.get("product:price:amount", ""))
    if price is None:
        ld_price, _foreign = _json_ld_price(html)
        if ld_price is not None:
            price = ld_price
        elif _foreign:
            return None
        else:
            return None
    title = _title(html, meta)
    if title is None or price is None:
        return None
    return ShopProduct(url=url, title=title, price_krw=price)
