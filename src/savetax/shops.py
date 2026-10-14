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
    "parse_seller_page",
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


_SOLD_OUT_RE = re.compile(r"품절|판매종료|SOLD\s*OUT", re.IGNORECASE)
# Shipping notices mention "일시품절, 제조사 상품수급 문제 … 배송이 늦어질 수" on every page; that is not a status.
_SOLD_OUT_NOTICE_RE = re.compile(r"일시품절\s*,|품절\s*(?:시|될|되는|이\s*발생)")
_SOLD_OUT_WINDOW = 1500
_SALE_PRICE_RE = re.compile(r"판매가\s*[:：]?\s*([\d,]+)\s*원")  # noqa: RUF001
# Product-block price markup per shop platform, in page order of trust. Each captures the product's own price, not
# related-item prices: microdata itemprop (oversystem 「판매가(카드)」), the single-price list `price only` with its
# card-payable sale price (cynexpc, kingdommol, duracom, picknmall, pc4all), compuzone's visible `price_real`
# (a hidden decoy figure precedes it), the quantity-one total (yopc, suzakpc), and cafe24's price text span.
_STRUCTURED_PRICE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"<[^>]*itemprop=[\"']price[\"'][^>]*\bcontent=[\"']([\d.]+)[\"']", re.IGNORECASE),
    re.compile(r"<ul\b[^>]*class=[\"']price only[\"'][^>]*>\s*<li>[^<]*<span\b[^>]*class=[\"']card[\"'][^>]*>\s*([\d,]+)", re.IGNORECASE),
    re.compile(r"<div\b[^>]*class=[\"']price_real[\"'][^>]*>((?:(?!</div>\s*</div>).){0,600})", re.IGNORECASE | re.DOTALL),
    re.compile(r"<[^>]*\bid=[\"']total_price[\"'][^>]*>\s*([\d,]+)", re.IGNORECASE),
    re.compile(r"<[^>]*\bid=[\"']span_product_price_text[\"'][^>]*>\s*([\d,]+)", re.IGNORECASE),
    re.compile(r"<div\b[^>]*class=[\"']view_price[\"'][^>]*>\s*<span\b[^>]*>\s*</span>\s*<span>\s*([\d,]+)", re.IGNORECASE),
)
# Many shop pages title themselves with the shop name; the product name sits in a platform-specific element.
_PRODUCT_NAME_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"<div\b[^>]*class=[\"']data_head[\"'][^>]*>\s*<span\b[^>]*class=[\"']name[\"'][^>]*>([^<]+)</span>", re.IGNORECASE),
    re.compile(r"<td\b[^>]*font-size\s*:\s*28px[^>]*>\s*([^<]+?)\s*</td>", re.IGNORECASE),
    re.compile(r"set_copy\('([^']+)'\)\"[^>]*title=\"상품명 복사", re.IGNORECASE),
)
_HIDDEN_BLOCK_RE = re.compile(r"<div\b[^>]*style=[\"'][^\"']*display\s*:\s*none[^\"']*[\"'][^>]*>.*?</div>", re.IGNORECASE | re.DOTALL)
_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?\s*([A-Za-z0-9._:-]+)", re.IGNORECASE)


def _header_charset(content_type: str | None) -> str | None:
    if not content_type:
        return None
    for part in content_type.split(";"):
        name, separator, value = part.partition("=")
        if separator and name.strip().lower() == "charset" and value.strip():
            return value.strip().strip("\"'")
    return None


def _meta_charset(raw: bytes) -> str | None:
    head = raw[:8192].decode("ascii", errors="ignore")
    match = _CHARSET_RE.search(head)
    if match is None:
        return None
    return match.group(1).strip().strip("\"'")


def _decode_seller_page(raw: bytes, content_type: str | None) -> str:
    charset = _header_charset(content_type) or _meta_charset(raw) or "utf-8"
    if charset.lower().replace("_", "-") in ("euc-kr", "euckr"):
        charset = "cp949"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _visible_sale_price(html: str) -> int | None:
    text = visible_text(html)
    # Member, coupon and card-company promotions are conditional prices the organizer method excludes. The plain
    # card-payable sale price (「판매가(카드)」) is the shop's standard price and stays; cash-only prices are discounts.
    text = re.sub(
        r"(?:회원가?|쿠폰|현금|(?:즉시|청구|결제)\s*할인가?|할인가?)\s*(?:전용\s*)?판매가\s*[:\uff1a]?\s*[\d,]+\s*원", "", text
    )
    text = re.sub(r"판매가\s*\(\s*카드\s*\)", "판매가", text)
    found: list[int] = []
    for match in _SALE_PRICE_RE.finditer(text):
        try:
            amount = int(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if amount > 0:
            found.append(amount)
    if not found or len(set(found)) != 1:
        return None
    return found[0]


def _structured_price(html: str) -> tuple[int, int] | None:
    """Return (price, html offset) of the first product-block price, or None."""
    for pattern in _STRUCTURED_PRICE_RES:
        match = pattern.search(html)
        if match is None:
            continue
        figure = match.group(1)
        if "<" in figure or "&#" in figure:
            visible = html_module.unescape(_HIDDEN_BLOCK_RE.sub("", figure))
            numbers = re.findall(r"\d{1,3}(?:,\d{3})+|\d+", re.sub(r"<[^>]+>", " ", visible))
            figure = numbers[0] if numbers else ""
        amount = _amount(figure.replace(",", ""))
        if amount is not None:
            return amount, match.start()
    return None


def _product_name(html: str) -> str | None:
    for pattern in _PRODUCT_NAME_RES:
        match = pattern.search(html)
        if match is not None:
            name = re.sub(r"\s+", " ", html_module.unescape(match.group(1))).strip()
            if name:
                return name
    return None


def _sold_out_near(html: str, offset: int) -> bool:
    window = visible_text(html[max(0, offset - _SOLD_OUT_WINDOW) : offset + _SOLD_OUT_WINDOW])
    window = _SOLD_OUT_NOTICE_RE.sub(" ", window)
    return _SOLD_OUT_RE.search(window) is not None


def parse_seller_page(raw: bytes, url: str, content_type: str | None) -> ShopProduct | None:
    """Decode by declared charset, then read title and the product's own sale price; None when sold out or unclear."""
    html = _decode_seller_page(raw, content_type)
    meta = _meta_map(html)
    currency = meta.get("product:price:currency")
    if currency is not None and currency != "KRW":
        return None
    price = _amount(meta.get("product:sale_price:amount", "")) or _amount(meta.get("product:price:amount", ""))
    offset = 0
    if price is None:
        structured = _structured_price(html)
        if structured is not None:
            price, offset = structured
    if price is None:
        ld_price, foreign = _json_ld_price(html)
        if foreign:
            return None
        price = ld_price if ld_price is not None else _visible_sale_price(html)
    if price is None or _sold_out_near(html, offset or _first_sale_offset(html)):
        return None
    title = _product_name(html) or _title(html, meta)
    if title is None:
        return None
    return ShopProduct(url=url, title=title, price_krw=price)


def _first_sale_offset(html: str) -> int:
    match = re.search(r"판매가|price", html, re.IGNORECASE)
    return match.start() if match else 0


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
