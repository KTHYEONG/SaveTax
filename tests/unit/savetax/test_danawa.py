"""Invariant guards for the Danawa catalog lane."""

from __future__ import annotations

import json

import pytest

from src.savetax.danawa import (
    model_pattern,
    parse_price_history,
    parse_product,
    parse_search,
)
from src.savetax.errors import DataIntegrityError, DomainValidationError

_SEARCH_HTML = (
    '<html><body><ul>'
    '<li data-product-seq="12345"><p class="mb-2 prod">Epson EB-PU2216B (정품)</p>'
    '<span>25,000,000 원</span><span>상품분류</span><span>프로젝터</span></li>'
    '<li><p>plain listing without seq</p></li>'
    "</ul></body></html>"
)

_PRODUCT_HTML = (
    "<html><head>"
    '<script type="application/ld+json">'
    '{"@type": "Product", "name": "Epson EB-PU2216B", "brand": {"name": "Epson"},'
    ' "description": "레이저 프로젝터", "offers": {"lowPrice": 25000000, "offerCount": 27}}'
    "</script></head><body>"
    '<ul class="list__mall-price">'
    '<li><span aria-label="몰A"></span><span class="text__num">2,800,000</span><span>무료배송</span></li>'
    '<li><img src="x" alt="몰B"><span class="text__num">2,900,000</span><span>3,000원</span></li>'
    "</ul>"
    "<span>등록월: 2025-03</span>"
    "</body></html>"
)


def test_country_prefix_is_ignored() -> None:
    assert model_pattern("(CN)EB-PU2216B").search("Epson EB-PU2216B (정품)") is not None


def test_separator_variants_match() -> None:
    assert model_pattern("HL-L6415DW").search("Brother HL L6415DW") is not None


def test_variant_suffix_does_not_match() -> None:
    assert model_pattern("C9300-24T").search("Cisco C9300-24T-A") is None


def test_embedded_token_does_not_match() -> None:
    assert model_pattern("SG2428").search("TL-SG2428P") is None


def test_empty_model_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        model_pattern("(CN)")


def test_catalog_rows_only() -> None:
    hits = parse_search(_SEARCH_HTML)
    assert len(hits) == 1
    assert hits[0].pcode == "12345"
    assert "EB-PU2216B" in hits[0].name
    assert hits[0].price_krw == 25000000


def test_rows_without_name_are_skipped() -> None:
    html = '<li data-product-seq="9"><p class="mb-2 x"></p></li>'
    assert parse_search(html) == []


def test_product_fields_come_from_json_ld() -> None:
    product = parse_product(_PRODUCT_HTML, "12345")
    assert product.name == "Epson EB-PU2216B"
    assert product.brand == "Epson"
    assert product.low_price_krw == 25000000
    assert product.offer_count == 27


def test_seller_rows_carry_shipping() -> None:
    product = parse_product(_PRODUCT_HTML, "12345")
    by_mall = {offer.mall: offer for offer in product.offers}
    assert by_mall["몰A"].shipping_krw == 0
    assert by_mall["몰B"].shipping_krw == 3000


def test_missing_json_ld_is_rejected() -> None:
    with pytest.raises(DataIntegrityError):
        parse_product("<html><body>no block</body></html>", "1")


def test_discontinued_product_has_no_price() -> None:
    html = (
        '<script type="application/ld+json">{"@type": "Product", "name": "N", '
        '"brand": {"name": "B"}, "description": "D"}</script>'
    )
    assert parse_product(html, "1").low_price_krw is None


def test_history_merges_both_ranges() -> None:
    long24 = [{"date": f"{24 + (m // 12):02d}-{m % 12 + 1:02d}", "minPrice": 100 + m} for m in range(11, 35)]
    short12 = [{"date": f"{25 + (m // 12):02d}-{m % 12 + 1:02d}", "minPrice": 200 + m} for m in range(10, 22)]
    payload = json.dumps({"24": {"result": long24}, "12": {"result": short12}})
    merged = parse_price_history(payload)
    assert all(key[:2] == "20" and len(key) == 7 for key in merged)
    assert len(merged) == len(set(merged))
    assert merged["2025-11"] == 200 + 10


def test_malformed_history_is_rejected() -> None:
    with pytest.raises(DataIntegrityError):
        parse_price_history("[]")
    with pytest.raises(DataIntegrityError):
        parse_price_history("not json")
    with pytest.raises(DataIntegrityError):
        parse_price_history('{"_error": {}}')
    with pytest.raises(DataIntegrityError):
        parse_price_history('{"24": []}')
    with pytest.raises(DataIntegrityError):
        parse_price_history('{"24": {"result": {}}}')


def test_search_row_without_name_tag_is_skipped() -> None:
    html = '<li data-product-seq="9"><span>25,000 원</span></li>'
    assert parse_search(html) == []


def test_search_category_plain_text_fallback() -> None:
    html = '<li data-product-seq="9"><p class="mb-2 x">모델 ABC123</p>상품분류 프로젝터</li>'
    hits = parse_search(html)
    assert len(hits) == 1
    assert hits[0].category == "프로젝터"
    assert hits[0].price_krw is None


def test_product_block_skips_empty_and_invalid() -> None:
    html = (
        '<script type="application/ld+json"></script>'
        '<script type="application/ld+json">not json</script>'
        '<script type="application/ld+json">[{"@type": "Other"}]</script>'
        '<script type="application/ld+json">{"@type": "Product", "name": "N", '
        '"brand": "Epson", "description": "D", "offers": {"lowPrice": "25,000,000", "offerCount": 3}}</script>'
        "<span>등록월: 2025.03</span>"
    )
    product = parse_product(html, "9")
    assert product.brand == "Epson"
    assert product.low_price_krw == 25000000
    assert product.registered_month == "2025-03"


def test_product_value_coercions() -> None:
    base = (
        '<script type="application/ld+json">{"@type": "Product", "name": 1, '
        '"brand": {"name": 2}, "description": 3, "offers": {"lowPrice": %s, "offerCount": %s}}</script>'
    )
    product = parse_product(base % ("12.5", "true"), "1")
    assert product.name == ""
    assert product.brand is None
    assert product.description == ""
    assert product.low_price_krw is None
    assert product.offer_count is None
    product2 = parse_product(base % ('"abc"', '"7"'), "1")
    assert product2.low_price_krw is None
    assert product2.offer_count == 7
    product3 = parse_product(base % ('""', "2.0"), "1")
    assert product3.low_price_krw is None
    assert product3.offer_count == 2
    assert parse_product(base % ("null", "null"), "1").registered_month is None


def test_offer_rows_with_missing_fields_are_skipped() -> None:
    html = (
        '<script type="application/ld+json">{"@type": "Product", "name": "N"}</script>'
        '<ul class="list__mall-price">'
        "<li>no mall at all</li>"
        '<li><span aria-label=" "></span></li>'
        '<li><span aria-label="M"></span></li>'
        '<li><span aria-label="M"></span><span class="text__num">가격</span></li>'
        '<li><span aria-label="M"></span><span class="text__num">1,000</span></li>'
        '<li><img alt="M2"><span class="text__num">2,000</span><span>5,000원</span></li>'
        "</ul>"
    )
    product = parse_product(html, "1")
    by_mall = {offer.mall: offer for offer in product.offers}
    assert by_mall["M"].shipping_krw is None
    assert by_mall["M2"].shipping_krw == 5000


def test_offer_mall_unescaped_blank_is_skipped() -> None:
    html = (
        '<script type="application/ld+json">{"@type": "Product", "name": "N"}</script>'
        '<ul class="list__mall-price">'
        '<li><span aria-label="&nbsp;"></span><span class="text__num">1,000</span></li>'
        '<li><span aria-label="M"></span><span class="text__num">2,000</span><span>무료배송</span></li>'
        "</ul>"
    )
    product = parse_product(html, "1")
    assert [offer.mall for offer in product.offers] == ["M"]


def test_history_section_without_result_is_empty() -> None:
    assert parse_price_history('{"24": {}, "12": {"result": []}}') == {}


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-100", "0", "1.5"])
def test_invalid_history_amounts_are_excluded(value: str) -> None:
    payload = json.dumps({"24": {"result": [{"date": "25-10", "minPrice": value}]}})
    assert parse_price_history(payload) == {}


def test_shipping_equal_to_price_is_preserved() -> None:
    html = _PRODUCT_HTML.replace("3,000원", "2,900,000원")
    assert parse_product(html, "1").offers[1].shipping_krw == 2900000


def test_search_price_and_registration_use_visible_text() -> None:
    html = ('<li data-product-seq="9"><p class="mb-2 x">Model ABC123</p>'
            '<script>1원</script><strong>25,000</strong>원</li>')
    assert parse_search(html)[0].price_krw == 25000
    product = parse_product(_PRODUCT_HTML.replace("등록월: 2025-03", "등록월: <b>2025-03</b>"), "1")
    assert product.registered_month == "2025-03"


def test_history_skips_bad_entries() -> None:
    payload = json.dumps(
        {
            "24": {
                "result": [
                    "bad",
                    {"date": 1, "minPrice": 1},
                    {"date": "xx", "minPrice": 1},
                    {"date": "25-13", "minPrice": 1},
                    {"date": "25-10", "minPrice": None},
                    {"date": "25-10", "minPrice": 1500000},
                ]
            },
        }
    )
    assert parse_price_history(payload) == {"2025-10": 1500000}


def test_offer_mall_comes_from_logo_not_generic_link_label() -> None:
    html = (
        '<script type="application/ld+json">{"@type": "Product", "name": "N"}</script>'
        '<ul class="list__mall-price">'
        "<li><div class=\"box__logo\"><img src='x.gif' class='image' alt='G마켓'></div>"
        '<span class="text__num">1,000</span><a aria-label="상품보기"></a></li>'
        '<li><div class="box__logo"><span class="text__logo" role="img" aria-label="판매사">판매사</span></div>'
        '<span class="text__num">1,010</span><span>무료배송</span><a aria-label="상품보기"></a></li>'
        '<li><div class="box__logo"></div><span class="text__num">1,020</span><a aria-label="상품보기"></a></li>'
        "</ul>"
    )
    product = parse_product(html, "1")
    assert [offer.mall for offer in product.offers] == ["G마켓", "판매사", "상품보기"]
