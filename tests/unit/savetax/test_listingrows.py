"""Invariant guards for the public delivery-row table."""

from datetime import date
from decimal import Decimal

import pytest

from src.savetax.errors import ConfigError, DataIntegrityError
from src.savetax.listingrows import delivery_page_url, parse_delivery_rows


def _table(rows: list[tuple[str, str, str, str, str, str]], pager: str = "") -> str:
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table>{body}</table>{pager}"


def test_rows_are_parsed_in_display_order() -> None:
    html = _table(
        [
            ("경기도 성남시 도서관사업소", "2026.5.29", "7대", "9,176,000원", "64,232,000원", "R26TB01970577"),
            ("서울시청", "2025.11.14", "12대", "20,832,000원", "249,984,000원", "R25TA00000001"),
            ("부산시청", "2026.1.2", "2,448㎡", "26,500원", "64,872,000원", "R26TB00000002"),
        ]
    )
    page = parse_delivery_rows(html)
    assert len(page.rows) == 3
    assert page.rows[0].request_no == "R26TB01970577"
    assert page.rows[0].unit_price_krw == 9176000
    assert page.rows[0].amount_krw == 64232000
    assert page.rows[0].request_date == date(2026, 5, 29)
    assert page.rows[2].quantity == Decimal("2448")


def test_quantity_keeps_decimals_and_unit() -> None:
    html = _table([("기관", "2026.5.29", "1.5톤", "1,000원", "1,500원", "R26TB00000003")])
    page = parse_delivery_rows(html)
    assert page.rows[0].quantity == Decimal("1.5")
    assert page.rows[0].unit == "톤"


def test_citation_text_matches_display() -> None:
    html = _table(
        [("경기도 성남시 도서관사업소", "2026.5.29", "7대", "9,176,000원", "64,232,000원", "R26TB01970577")]
    )
    page = parse_delivery_rows(html)
    assert page.rows[0].text == "경기도 성남시 도서관사업소 2026.5.29 7대 9,176,000원 64,232,000원 R26TB01970577"


def test_pager_with_comments_is_read() -> None:
    html = _table(
        [("기관", "2026.5.29", "7대", "1,000원", "7,000원", "R26TB00000004")],
        "<div>1<!-- --> / <!-- -->3<!-- -->쪽</div>",
    )
    assert parse_delivery_rows(html).page_count == 3


def test_page_without_pager_has_one_page() -> None:
    html = _table([("기관", "2026.5.29", "7대", "1,000원", "7,000원", "R26TB00000004")])
    assert parse_delivery_rows(html).page_count == 1


def test_class_names_are_irrelevant() -> None:
    rows = [("기관", "2026.5.29", "7대", "1,000원", "7,000원", "R26TB00000004")]
    plain = _table(rows)
    fancy = (
        '<table><tr><td class="a" id="x">기관</td><td class="num">2026.5.29</td>'
        '<td class="num">7대</td><td class="ft-amt">1,000원</td>'
        '<td class="ft-amt">7,000원</td><td><span class="ft-code">R26TB00000004</span></td></tr></table>'
    )
    assert parse_delivery_rows(plain).rows == parse_delivery_rows(fancy).rows


def test_malformed_price_cell_fails_closed() -> None:
    html = _table([("기관", "2026.5.29", "7대", "-", "7,000원", "R26TB00000004")])
    with pytest.raises(DataIntegrityError):
        parse_delivery_rows(html)


def test_page_url_adds_page_parameter_only_from_page_two() -> None:
    template = "https://savetax.now/listing/{listing_id}"
    assert delivery_page_url(template, "A_1", 1) == "https://savetax.now/listing/A_1"
    assert delivery_page_url(template, "A_1", 2).endswith("?dp=2")
    with pytest.raises(ConfigError):
        delivery_page_url(template, "A_1", 0)


def test_listing_rows_edge_branches_are_covered() -> None:
    bad_date = _table([("기관", "not-a-date", "7대", "1,000원", "7,000원", "R26TB00000004")])
    with pytest.raises(DataIntegrityError):
        parse_delivery_rows(bad_date)
    bad_day = _table([("기관", "2026.2.30", "7대", "1,000원", "7,000원", "R26TB00000004")])
    with pytest.raises(DataIntegrityError):
        parse_delivery_rows(bad_day)
    bad_qty = _table([("기관", "2026.5.29", "대", "1,000원", "7,000원", "R26TB00000004")])
    with pytest.raises(DataIntegrityError):
        parse_delivery_rows(bad_qty)
    bad_won = _table([("기관", "2026.5.29", "7대", "1,000원", "7000", "R26TB00000004")])
    with pytest.raises(DataIntegrityError):
        parse_delivery_rows(bad_won)
    header_only = "<table><tr><th>항목</th><th>값</th></tr><tr><td>a</td><td>b</td></tr></table>"
    assert parse_delivery_rows(header_only).rows == ()
    other_table = _table([("기관", "2026.5.29", "7대", "1,000원", "7,000원", "NOT-A-ROW")])
    assert parse_delivery_rows(other_table).rows == ()
    zero_pager = _table(
        [("기관", "2026.5.29", "7대", "1,000원", "7,000원", "R26TB00000004")],
        "<div>1 / 0쪽</div>",
    )
    with pytest.raises(DataIntegrityError):
        parse_delivery_rows(zero_pager)
    with pytest.raises(ConfigError):
        delivery_page_url("https://savetax.now/listing/{listing_id}", "A_1", True)
    assert delivery_page_url("https://savetax.now/listing/{listing_id}#frag", "A_1", 1).find("#") < 0


def test_listing_rows_remaining_branches_are_covered() -> None:
    trailing = _table([("기관", "2026.5.29", "7   ", "1,000원", "7,000원", "R26TB00000004")])
    with pytest.raises(DataIntegrityError):
        parse_delivery_rows(trailing)
    won_only = _table([("기관", "2026.5.29", "7대", "원", "7,000원", "R26TB00000004")])
    with pytest.raises(DataIntegrityError):
        parse_delivery_rows(won_only)


@pytest.mark.parametrize("quantity", ["12", "1.대", "1,2대", ",대", "1.5.2톤", "-7대", "1,000", "7 대2"])
def test_malformed_quantity_cannot_be_repaired(quantity: str) -> None:
    with pytest.raises(DataIntegrityError, match="malformed delivery row on page"):
        parse_delivery_rows(_table([("기관", "2026.5.29", quantity, "1,000원", "7,000원", "R26TB00000004")]))


@pytest.mark.parametrize("price", ["-1,000원", "1.5원", "1,00원", "abc1000원", "1 000원"])
def test_price_text_cannot_be_silently_transformed(price: str) -> None:
    with pytest.raises(DataIntegrityError, match="malformed delivery row on page"):
        parse_delivery_rows(_table([("기관", "2026.5.29", "7대", price, "7,000원", "R26TB00000004")]))


def test_hidden_code_cannot_supply_delivery_evidence() -> None:
    table = _table([("기관", "2026.5.29", "7대", "1,000원", "7,000원", "R26TB00000004")], "1 / 9쪽")
    page = parse_delivery_rows(f"<script>{table}</script><style>{table}</style>")
    assert page.rows == ()
    assert page.page_count == 1
