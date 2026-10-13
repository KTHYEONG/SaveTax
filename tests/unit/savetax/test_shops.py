"""Invariant guards for specialty-shop parsing."""

from __future__ import annotations

import json
import pytest
from src.savetax.config import ShopSpec
from src.savetax.shops import parse_shop_product, parse_shop_search, shop_search_url

_CAFE = ShopSpec("example.cafe24.kr", "cafe24")
_GODO = ShopSpec("example.godo.kr", "godomall")


def test_search_url_per_platform() -> None:
    assert shop_search_url(_CAFE, "(CN)NSH-2228") == "https://example.cafe24.kr/product/search.html?keyword=NSH-2228"
    assert shop_search_url(_GODO, "(CN)NSH-2228") == "https://example.godo.kr/goods/goods_search.php?keyword=NSH-2228"


def test_product_links_are_absolute_and_unique() -> None:
    html = (
        '<a href="/product/detail.html?product_no=11">a</a>'
        '<a href="/product/detail.html?product_no=11">b</a>'
        '<a href="https://other.example/product/detail.html?product_no=11">c</a>'
    )
    assert parse_shop_search(html, _CAFE) == ["https://example.cafe24.kr/product/detail.html?product_no=11"]


def test_godomall_links_resolve() -> None:
    html = '<a href="/goods/goods_view.php?goodsNo=77&x=1">a</a>'
    assert parse_shop_search(html, _GODO) == ["https://example.godo.kr/goods/goods_view.php?goodsNo=77&x=1"]


def test_sale_price_wins_over_list_price() -> None:
    html = (
        '<meta property="product:price:currency" content="KRW">'
        '<meta property="product:sale_price:amount" content="49000">'
        '<meta property="product:price:amount" content="53900">'
        '<meta property="og:title" content="NSH-2228 카메라">'
    )
    product = parse_shop_product(html, "https://example.cafe24.kr/x")
    assert product is not None
    assert product.price_krw == 49000


def test_foreign_currency_yields_nothing() -> None:
    html = (
        '<script type="application/ld+json">{"@type": "Product", "price": "10", "priceCurrency": "USD"}</script>'
        "<title>NSH-2228</title>"
    )
    assert parse_shop_product(html, "https://example.cafe24.kr/x") is None


def test_page_without_structured_price_yields_nothing() -> None:
    assert parse_shop_product("<title>T</title><p>판매가 1,000원</p>", "https://example.cafe24.kr/x") is None


def test_slug_links_and_skipped_hrefs() -> None:
    html = (
        '<a href="#">x</a><a href="javascript:void(0)">y</a><a href="/notice.html">z</a>'
        '<a href="http://example.cafe24.kr/product/detail.html?product_no=11">h</a>'
        '<a href="/product/camera/22/">a</a><a href="/product/camera/22/#frag">b</a>'
    )
    assert parse_shop_search(html, _CAFE) == ["https://example.cafe24.kr/product/camera/22/"]


def test_price_coercions() -> None:
    base = (
        '<script type="application/ld+json">%s</script><meta property="og:title" content="NSH-2228">'
    )
    assert parse_shop_product(base % '{"price": 49000, "priceCurrency": "KRW"}', "https://example.cafe24.kr/x").price_krw == 49000  # type: ignore[union-attr]
    assert parse_shop_product(base % '{"price": true, "priceCurrency": "KRW"}', "https://example.cafe24.kr/x") is None
    assert parse_shop_product(base % '{"price": null, "priceCurrency": "KRW"}', "https://example.cafe24.kr/x") is None
    assert (
        parse_shop_product(base % '{"price": "abc", "priceCurrency": "KRW"}', "https://example.cafe24.kr/x") is None
    )
    assert parse_shop_product(base % '{"price": "10"}', "https://example.cafe24.kr/x") is None
    assert parse_shop_product(base % '{"price": "10", "priceCurrency": 1}', "https://example.cafe24.kr/x") is None
    assert parse_shop_product(base % '[{"price": "52000", "priceCurrency": "KRW"}]', "https://example.cafe24.kr/x").price_krw == 52000  # type: ignore[union-attr]
    assert (
        parse_shop_product(
            '<script type="application/ld+json"></script><script type="application/ld+json">xx</script>' + base % '{"price": "53000", "priceCurrency": "KRW"}',
            "https://example.cafe24.kr/x",
        ).price_krw  # type: ignore[union-attr]
        == 53000
    )


def test_meta_decimal_and_invalid_amounts() -> None:
    html = (
        '<meta property="product:sale_price:amount" content="49000.9">'
        '<meta property="og:title" content="NSH-2228">'
    )
    assert parse_shop_product(html, "https://example.cafe24.kr/x") is None
    bad = (
        '<meta property="product:sale_price:amount" content="abc">'
        '<meta property="product:price:amount" content="">'
        '<meta property="og:title" content="NSH-2228">'
    )
    assert parse_shop_product(bad, "https://example.cafe24.kr/x") is None


def test_missing_title_yields_nothing() -> None:
    html = '<script type="application/ld+json">{"price": "49000", "priceCurrency": "KRW"}</script>'
    assert parse_shop_product(html, "https://example.cafe24.kr/x") is None
    assert parse_shop_product(html + "<title>  </title>", "https://example.cafe24.kr/x") is None


@pytest.mark.parametrize("price", ["NaN", "Infinity", "-1", "0", "1.5"])
def test_invalid_structured_prices_are_excluded(price: str) -> None:
    payload = json.dumps({"offers": {"price": price, "priceCurrency": "KRW"}})
    html = f'<script type="application/ld+json">{payload}</script><title>Model</title>'
    assert parse_shop_product(html, "https://example.cafe24.kr/x") is None


def test_foreign_meta_cannot_fall_back_to_krw_json() -> None:
    html = ('<meta property="product:price:currency" content="USD">'
            '<script type="application/ld+json">{"price": 1000, "priceCurrency": "KRW"}</script><title>N</title>')
    assert parse_shop_product(html, "https://example.cafe24.kr/x") is None


def test_options_are_not_product_evidence() -> None:
    payload = json.dumps({"@type": "Product", "options": [{"price": 1000, "priceCurrency": "KRW"}]})
    assert parse_shop_product(f'<script type="application/ld+json">{payload}</script><title>N</title>',
                              "https://example.cafe24.kr/x") is None


def test_graph_product_price_is_supported() -> None:
    payload = json.dumps({"@graph": [{"@type": "Product", "offers": {"price": 1000, "priceCurrency": "KRW"}}]})
    product = parse_shop_product(f'<script type="application/ld+json">{payload}</script><title>N</title>',
                                "https://example.cafe24.kr/x")
    assert product is not None
    assert product.price_krw == 1000


def test_unsafe_same_host_product_links_are_dropped() -> None:
    html = ('<a href="https://user@example.cafe24.kr/product/x/1/">a</a>'
            '<a href="https://example.cafe24.kr:8443/product/x/1/">b</a>'
            '<a href="/product/x/1/">c</a>')
    assert parse_shop_search(html, _CAFE) == ["https://example.cafe24.kr/product/x/1/"]
