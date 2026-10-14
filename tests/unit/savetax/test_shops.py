"""Invariant guards for specialty-shop parsing."""

from __future__ import annotations

import json
import pytest
from src.savetax.config import ShopSpec
from src.savetax.shops import parse_seller_page, parse_shop_product, parse_shop_search, shop_search_url

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


def test_seller_euckr_page_decoded() -> None:
    html = '<html><head><meta charset="EUC-KR"><title>울트라5 225 정품 팝니다</title></head><body><p>판매가 : 88,000원</p></body></html>'
    raw = html.encode("cp949")
    product = parse_seller_page(raw, "https://compuzone.co.kr/product/x", "text/html")
    assert product is not None
    assert product.price_krw == 88000
    assert "울트라5" in product.title


def test_seller_two_different_sale_prices_ambiguous() -> None:
    html = "<title>CPU</title><p>판매가 88,000원</p><p>판매가 90,000원</p>"
    assert parse_seller_page(html.encode(), "https://oversystem.co.kr/x", None) is None


def test_seller_member_price_ignored() -> None:
    html = "<title>CPU 정품</title><p>회원가 70,000원</p><p>판매가 88,000원</p>"
    product = parse_seller_page(html.encode(), "https://pctopia.co.kr/x", None)
    assert product is not None
    assert product.price_krw == 88000


def test_seller_sold_out_yields_nothing() -> None:
    html = "<title>CPU</title><p>판매가 88,000원</p><p>품절</p>"
    assert parse_seller_page(html.encode(), "https://cs-office.net/x", None) is None


def test_seller_unknown_charset_falls_back_to_utf8() -> None:
    html = "<title>CPU 정품</title><p>판매가 88,000원</p>"
    product = parse_seller_page(html.encode(), "https://pctopia.co.kr/x", "text/html; charset=unknown-charset-xyz")
    assert product is not None
    assert product.price_krw == 88000


def test_seller_header_charset_wins() -> None:
    html = '<html><head><meta charset="utf-8"><title>울트라5 225 정품</title></head><body><p>판매가 88,000원</p></body></html>'
    raw = html.encode("cp949")
    product = parse_seller_page(raw, "https://yopc.co.kr/x", "text/html; charset=EUC-KR")
    assert product is not None
    assert product.price_krw == 88000


def test_seller_broken_sale_figure_ignored() -> None:
    html = "<title>CPU</title><p>판매가 ,,,원</p>"
    assert parse_seller_page(html.encode(), "https://oversystem.co.kr/x", None) is None
    html0 = "<title>CPU</title><p>판매가 0원</p>"
    assert parse_seller_page(html0.encode(), "https://oversystem.co.kr/x", None) is None


def test_seller_foreign_currency_rejected() -> None:
    html = ('<meta property="product:price:currency" content="USD">'
            '<meta property="product:sale_price:amount" content="88">'
            "<title>CPU</title>")
    assert parse_seller_page(html.encode(), "https://compuzone.co.kr/x", None) is None
    html2 = ('<script type="application/ld+json">{"price": 10, "priceCurrency": "USD"}</script>'
             "<title>CPU</title>")
    assert parse_seller_page(html2.encode(), "https://compuzone.co.kr/x", None) is None


def test_seller_json_ld_price_used() -> None:
    html = ('<script type="application/ld+json">{"price": 88000, "priceCurrency": "KRW"}</script>'
            "<title>울트라5 225 정품</title>")
    product = parse_seller_page(html.encode(), "https://duracom.co.kr/x", None)
    assert product is not None
    assert product.price_krw == 88000


def test_seller_price_only_list_reads_card_payable_price() -> None:
    html = ("<title>씨넥스PC</title><div class=\"data_head\"><span class=\"name\">울트라7 265 정품</span></div>"
            "<div class=\"price_wrap clear\"><ul class='price only'><li>판매가<span class='card'>542,500</span>원</li></ul></div>"
            "<span class=\"prd_price\">19,900</span>")
    product = parse_seller_page(html.encode("cp949"), "https://www.cynexpc.com/x", "text/html; charset=euc-kr")
    assert product is not None
    assert (product.title, product.price_krw) == ("울트라7 265 정품", 542500)


def test_seller_compuzone_hidden_decoy_ignored() -> None:
    html = ('<title>[INTEL] 225 : 컴퓨존</title><div class="price_real"><div style="display:none;">263,500</div>'
            "&#50;&#54;&#57;,&#48;&#48;&#48;</div></div>")
    product = parse_seller_page(html.encode(), "https://www.compuzone.co.kr/x", None)
    assert product is not None
    assert product.price_krw == 269000


def test_seller_total_price_and_large_cell_name() -> None:
    html = ('<title>요피씨</title><td style="color:#333; font-size:28px" class="NanumGothic bold"> 울트라5 225 정품 </td>'
            '<span id="total_price" class="x"> 273,220</span>')
    product = parse_seller_page(html.encode(), "https://www.yopc.co.kr/x", None)
    assert product is not None
    assert (product.title, product.price_krw) == ("울트라5 225 정품", 273220)


def test_seller_view_price_and_copy_button_name() -> None:
    html = ("<title>아이코다</title><span onclick=\"set_copy('1771060')\" title=\"번호 복사\"></span>"
            "<span onclick=\"set_copy('ASUS PRIME H810M-K')\" class=\"pointer\" title=\"상품명 복사 : 클릭\"></span>"
            '<div class="view_price"> <span class="no_margin"></span> <span>87,580</span></div>')
    product = parse_seller_page(html.encode(), "https://usr.icoda.co.kr/x", None)
    assert product is not None
    assert (product.title, product.price_krw) == ("ASUS PRIME H810M-K", 87580)


def test_seller_microdata_card_price() -> None:
    html = ('<title>오버시스템-EXCEL II 500W</title><div class="ViewPriceLabel"> 판매가(카드) </div>'
            '<span itemprop="price" content="37000"> 37,000원 </span>')
    product = parse_seller_page(html.encode(), "https://oversystem.co.kr/x", None)
    assert product is not None
    assert product.price_krw == 37000


def test_seller_visible_card_price_kept_and_cash_price_dropped() -> None:
    html = "<title>CPU 정품</title><p>현금판매가 85,000원</p><p>판매가(카드) 88,000원</p><p>즉시할인 판매가 80,000원</p>"
    product = parse_seller_page(html.encode(), "https://pctopia.co.kr/x", None)
    assert product is not None
    assert product.price_krw == 88000


def test_seller_shipping_notice_is_not_sold_out() -> None:
    filler = "가" * 2000
    html = (f'<title>CPU</title><span id="total_price">8,690</span>{filler}'
            "<p>일시품절, 제조사 상품수급 문제 등의 사유로 배송이 늦어질 수 있습니다. 품절 시 연락드립니다.</p>")
    product = parse_seller_page(html.encode(), "https://www.suzakpc.co.kr/x", None)
    assert product is not None
    assert product.price_krw == 8690


def test_seller_sold_out_next_to_price() -> None:
    html = '<title>CPU</title><span id="total_price">8,690</span><button>품절</button>'
    assert parse_seller_page(html.encode(), "https://www.suzakpc.co.kr/x", None) is None


def test_seller_structured_markup_without_number_falls_through() -> None:
    html = '<title>CPU 정품</title><div class="price_real"><div style="display:none;">1</div>문의</div></div><p>판매가 88,000원</p>'
    product = parse_seller_page(html.encode(), "https://www.compuzone.co.kr/x", None)
    assert product is not None
    assert product.price_krw == 88000


def test_seller_page_without_any_name_is_rejected() -> None:
    assert parse_seller_page(b'<span id="total_price">8,690</span>', "https://www.suzakpc.co.kr/x", None) is None
