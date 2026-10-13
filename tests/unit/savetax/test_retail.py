"""Invariant guards for the retail same-model lane."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pytest

from src.savetax.competition import Fact, ItemDetail, ItemState
from src.savetax.config import AuditConfig, EvidenceConfig, SaveTaxConfig, ShopSpec
from src.savetax.danawa import DanawaProduct, MallOffer
from src.savetax.dataset import Dataset
from src.savetax.errors import NetworkError
from src.savetax.http import HttpResponse
from src.savetax.publicweb import PublicFetcher
from src.savetax.rawapi import ListingDetail
from src.savetax.retail import brand_matches, build_retail_bundle, current_benchmark, is_accessory_or_pack

_TODAY = date(2026, 10, 13)
_REQUEST = date(2025, 10, 12)
_MONTH = "2025-10"

_COLUMNS = (
    "listing_id",
    "item_id",
    "name",
    "detail_category",
    "category_l_name",
    "manufacturer",
    "model",
    "unit",
    "price",
    "contract_begin",
    "contract_end",
    "vendor_name",
    "dlv_key",
    "dlv_lines",
    "dlv_qty",
    "dlv_unit",
    "dlv_amount",
    "dlv_first_date",
    "dlv_last_date",
)


def _base_row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "listing_id": "111111111_0000000001",
        "item_id": "11111111",
        "name": "프로젝터",
        "detail_category": "프로젝터",
        "category_l_name": "전자/정보/통신/영상",
        "manufacturer": "Epson",
        "model": "EB-PU2216B",
        "unit": "대",
        "price": 2508000,
        "contract_begin": "2025-01-01",
        "contract_end": "2026-12-31",
        "vendor_name": "업체A",
        "dlv_key": 1,
        "dlv_lines": 1,
        "dlv_qty": 2,
        "dlv_unit": "대",
        "dlv_amount": 5472000,
        "dlv_first_date": "2025-10-11",
        "dlv_last_date": "2025-10-12",
    }
    row.update(over)
    return row


def _write_dataset(tmp_path: Path, rows: list[dict[str, Any]]) -> Dataset:
    data_dir = tmp_path / "savetax-products-test"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet = data_dir / "listings.parquet"
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE tmp(listing_id VARCHAR, item_id VARCHAR, name VARCHAR, "
            "detail_category VARCHAR, category_l_name VARCHAR, manufacturer VARCHAR, "
            "model VARCHAR, unit VARCHAR, price DECIMAL(18, 3), contract_begin DATE, "
            "contract_end DATE, vendor_name VARCHAR, dlv_key INTEGER, dlv_lines INTEGER, "
            "dlv_qty BIGINT, dlv_unit VARCHAR, dlv_amount BIGINT, "
            "dlv_first_date DATE, dlv_last_date DATE)"
        )
        for row in rows:
            con.execute(
                "INSERT INTO tmp VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [row.get(c) for c in _COLUMNS],
            )
        proj = ", ".join(_COLUMNS)
        con.execute(f"COPY (SELECT {proj} FROM tmp) TO '{parquet}' (FORMAT PARQUET)")  # noqa: S608
    finally:
        con.close()
    return Dataset(SaveTaxConfig(dataset_dir=data_dir), AuditConfig())


def _evidence(**over: Any) -> EvidenceConfig:
    return EvidenceConfig(**over)  # type: ignore[arg-type]


def _detail_listing(
    listing_id: str,
    price: str = "3500000",
    terms: str | None = "서울",
    region: str | None = "서울",
    contract_no: str = "111111111_0",
) -> ListingDetail:
    return ListingDetail(
        listing_id=listing_id,
        item_id="11111111",
        unit="대",
        price_krw=Decimal(price),
        price_kind="contract",
        contract_no=contract_no,
        contract_type=None,
        contract_begin=date(2025, 1, 1),
        contract_end=date(2026, 12, 31),
        vendor_name="벤더",
        manufacturer=None,
        model=None,
        category=None,
        delivery_terms=terms,
        supply_region=region,
        release="rtest",
    )


class _FakeRaw:
    def __init__(self, mapping: dict[str, ListingDetail | None]) -> None:
        self.mapping = mapping

    def get_listing(self, listing_id: str) -> ListingDetail | None:
        return self.mapping.get(listing_id)


def _product(
    low: int | None = 2800000,
    offers: list[tuple[str, int, int | None]] | None = None,
    brand: str | None = "Epson",
    name: str = "Epson EB-PU2216B 프로젝터",
    description: str = "레이저 프로젝터",
) -> DanawaProduct:
    rows = offers if offers is not None else [("몰A", 2800000, 0), ("몰B", 2900000, 0)]
    return DanawaProduct(
        pcode="1",
        name=name,
        brand=brand,
        description=description,
        low_price_krw=low,
        offer_count=len(rows),
        offers=tuple(MallOffer(mall=mall, price_krw=price, shipping_krw=ship) for mall, price, ship in rows),
        registered_month="2025-03",
    )


def _search_html(hits: list[tuple[str, str, int]]) -> str:
    rows = "".join(
        f'<li data-product-seq="{pcode}"><p class="mb-2 x">{name}</p><span>{price:,} 원</span></li>'
        for pcode, name, price in hits
    )
    return f"<html><body><ul>{rows}</ul></body></html>"


def _product_html(
    name: str = "Epson EB-PU2216B",
    brand: str = "Epson",
    description: str = "레이저 프로젝터",
    low: int | None = 2800000,
    count: int = 2,
    offers: list[tuple[str, int, str]] | None = None,
) -> str:
    rows = offers if offers is not None else [("몰A", 2800000, "무료배송"), ("몰B", 2900000, "무료배송")]
    items = "".join(
        f'<li><span aria-label="{mall}"></span><span class="text__num">{price:,}</span><span>{ship}</span></li>'
        for mall, price, ship in rows
    )
    offers_json = f'"lowPrice": {low}, "offerCount": {count}' if low is not None else ""
    return (
        "<html><head><script type=\"application/ld+json\">"
        f'{{"@type": "Product", "name": "{name}", "brand": {{"name": "{brand}"}}, '
        f'"description": "{description}", "offers": {{{offers_json}}}}}'
        "</script></head><body>"
        f'<ul class="list__mall-price">{items}</ul>'
        "<span>등록월: 2025-03</span></body></html>"
    )


def _history_payload(months: dict[str, int]) -> str:
    result = [{"date": key[2:], "minPrice": value} for key, value in sorted(months.items())]
    return json.dumps({"24": {"result": result}, "12": {"result": []}})


def _fetcher(routes: dict[str, tuple[int, str]], seen: list, evidence: EvidenceConfig | None = None) -> PublicFetcher:
    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        seen.append((url, dict(headers)))  # type: ignore[arg-type]
        for key, (status, body) in routes.items():
            if key in url:
                return HttpResponse(status=status, headers={}, body=body.encode(), elapsed_seconds=0.01)
        return HttpResponse(status=404, headers={}, body=b"", elapsed_seconds=0.01)

    return PublicFetcher(
        SaveTaxConfig(),
        evidence if evidence is not None else EvidenceConfig(),
        transport=transport,  # type: ignore[arg-type]
        sleep=lambda _s: None,
        clock=lambda: 0.0,
    )


def _fact(**over: Any) -> Fact:
    base: dict[str, Any] = {
        "fact_id": "F1",
        "listing_id": "111111111_0000000001",
        "contract_no": "111111111_0",
        "request_no": "R1",
        "line_no": "1",
        "request_date": _REQUEST,
        "unit": "대",
        "tax_basis": "vat_included",
        "basis_id": "basis:대:vat_included",
        "period_kind": "none",
        "unit_price_krw": Decimal("3500000"),
        "quantity": Decimal("2"),
        "comparable_amount_krw": Decimal("7000000"),
        "scorable": True,
        "exclusion_reasons": (),
    }
    base.update(over)
    return Fact(**base)


def _item_detail(
    facts: list[Fact],
    model: str | None = "EB-PU2216B",
    manufacturer: str | None = "Epson",
    state: ItemState = ItemState.AVAILABLE,
) -> ItemDetail:
    return ItemDetail(
        item_id="11111111",
        state=state,
        name="프로젝터",
        model=model,
        manufacturer=manufacturer,
        category="프로젝터",
        source_release="rtest",
        facts_total=len(facts),
        facts_truncated=False,
        facts=tuple(facts),
    )


def _routes_for_product(
    model: str = "EB-PU2216B",
    product_html: str | None = None,
    history: dict[str, int] | None = None,
) -> dict[str, tuple[int, str]]:
    search = _search_html([("1", f"Epson {model} 프로젝터", 2800000)])
    product = product_html if product_html is not None else _product_html(name=f"Epson {model}")
    hist = _history_payload(history if history is not None else {_MONTH: 2800000})
    return {
        "search.danawa.com": (200, search),
        "getProductPriceList": (200, hist),
        "prod.danawa.com/info": (200, product),
    }


def _setup(
    tmp_path: Path,
    routes: dict[str, tuple[int, str]],
    raw_map: dict[str, ListingDetail | None],
    facts: list[Fact] | None = None,
    dataset_rows: list[dict[str, Any]] | None = None,
    detail: ItemDetail | None = None,
) -> tuple[Dataset, _FakeRaw, PublicFetcher, list, ItemDetail]:
    ds = _write_dataset(tmp_path, dataset_rows if dataset_rows is not None else [_base_row()])
    raw = _FakeRaw(raw_map)
    seen: list = []
    fetch = _fetcher(routes, seen)
    item = detail if detail is not None else _item_detail(facts if facts is not None else [_fact()])
    return ds, raw, fetch, seen, item


def test_alias_brand_matches() -> None:
    assert brand_matches("엘지전자 주식회사", "LG전자 TV", "LG전자", _evidence()) is True


def test_other_maker_does_not_match() -> None:
    assert brand_matches("Alcatel-Lucent Enterprise", "스위치", "이지넷유비쿼터스", _evidence()) is False


def test_empty_manufacturer_or_brand_is_false() -> None:
    assert brand_matches("", "이름", "Epson", _evidence()) is False
    assert brand_matches("Epson", "이름", None, _evidence()) is False


def test_accessory_is_excluded() -> None:
    assert (
        is_accessory_or_pack("프로젝터", "액세서리 / 회전유리판 / 적용모델: CMW-G4010ES", None, _evidence()) is True
    )


def test_multi_pack_is_excluded() -> None:
    assert is_accessory_or_pack("토너(2x16TB)", "설명", None, _evidence()) is True
    assert is_accessory_or_pack("일반 프로젝터", "일반 설명", "프로젝터", _evidence()) is False


def test_supported_lowest_price_is_used() -> None:
    bench = current_benchmark(_product(), {}, ("2024-01", "2026-12"), _evidence())
    assert bench == 2800000


def test_outlier_lowest_falls_back_to_second() -> None:
    product = _product(low=25000000, offers=[("몰A", 25000000, 0), ("몰B", 78474000, 0)])
    assert current_benchmark(product, {}, ("2024-01", "2026-12"), _evidence()) == 78474000


def test_single_unstable_seller_is_unsupported() -> None:
    product = _product(low=94800, offers=[("몰A", 94800, 0)])
    history = {"2025-01": 94800, "2025-02": 132000}
    assert current_benchmark(product, history, ("2025-01", "2025-02"), _evidence()) is None


def test_single_stable_seller_is_supported() -> None:
    product = _product(low=100000, offers=[("몰A", 100000, 0)])
    history = {"2025-01": 100000, "2025-02": 110000}
    assert current_benchmark(product, history, ("2025-01", "2025-02"), _evidence()) == 100000


def test_no_low_price_is_unsupported() -> None:
    assert current_benchmark(_product(low=None), {}, ("2024-01", "2026-12"), _evidence()) is None


def test_order_month_price_raises_benchmark(tmp_path: Path) -> None:
    routes = _routes_for_product(history={_MONTH: 3150000})
    order = _detail_listing("111111111_0000000001", price="3500000")
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order})
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("3150000")


def test_missing_month_rejects_fact(tmp_path: Path) -> None:
    routes = _routes_for_product(history={"2025-09": 2800000})
    order = _detail_listing("111111111_0000000001", price="3500000")
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order})
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is None


def test_sibling_delta_is_installation(tmp_path: Path) -> None:
    product_html = _product_html(
        offers=[("몰A", 2000000, "무료배송"), ("몰B", 2100000, "무료배송")],
        low=2000000,
    )
    routes = _routes_for_product(product_html=product_html, history={_MONTH: 2000000})
    rows = [
        _base_row(listing_id="111111111_0000000001", price=2420000),
        _base_row(
            listing_id="111111111_0000000002",
            price=2200000,
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
        ),
    ]
    order_install = _detail_listing("111111111_0000000001", price="2500000", terms="현장설치도")
    order_plain = _detail_listing("111111111_0000000002", price="2280000", terms="납품장소도")
    fact = _fact(unit_price_krw=Decimal("2500000"), comparable_amount_krw=Decimal("5000000"))
    ds, raw, fetch, _seen, item = _setup(
        tmp_path,
        routes,
        {order_install.listing_id: order_install, order_plain.listing_id: order_plain},
        facts=[fact],
        dataset_rows=rows,
    )
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2220000")


def test_allowance_applies_without_sibling(tmp_path: Path) -> None:
    product_html = _product_html(
        offers=[("몰A", 1000000, "무료배송"), ("몰B", 1050000, "무료배송")],
        low=1000000,
    )
    routes = _routes_for_product(product_html=product_html, history={_MONTH: 1000000})
    order = _detail_listing("111111111_0000000001", price="1400000", terms="현장설치도")
    fact = _fact(unit_price_krw=Decimal("1400000"), comparable_amount_krw=Decimal("2800000"))
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order}, facts=[fact])
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("1150000")


def test_margin_gate(tmp_path: Path) -> None:
    product_html = _product_html(
        offers=[("몰A", 1000000, "무료배송"), ("몰B", 1050000, "무료배송")],
        low=1000000,
    )
    routes = _routes_for_product(product_html=product_html, history={_MONTH: 1000000})

    def run(price: str) -> Any:
        order = _detail_listing("111111111_0000000001", price=price, terms="현장설치도")
        fact = _fact(unit_price_krw=Decimal(price), quantity=Decimal("1"), comparable_amount_krw=Decimal(price))
        ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order}, facts=[fact])
        try:
            return build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
        finally:
            ds.close()

    assert run("1264999") is None
    kept = run("1265000")
    assert kept is not None
    assert kept.lines[0].benchmark_unit_price_krw == Decimal("1150000")


def test_brand_unconfirmed_bundle_needs_review(tmp_path: Path) -> None:
    product_html = _product_html(
        name="Sony EB-PU2216B 프로젝터",
        brand="Sony",
        offers=[("몰A", 1000000, "무료배송"), ("몰B", 1050000, "무료배송")],
        low=1000000,
    )
    routes = _routes_for_product(product_html=product_html, history={_MONTH: 1000000})
    order = _detail_listing("111111111_0000000001", price="1900000", terms="서울")
    fact = _fact(unit_price_krw=Decimal("1900000"), comparable_amount_krw=Decimal("3800000"))
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order}, facts=[fact])
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.review_required is True


def test_implausible_gap_is_rejected(tmp_path: Path) -> None:
    product_html = _product_html(
        name="Sony EB-PU2216B 프로젝터",
        brand="Sony",
        offers=[("몰A", 1000000, "무료배송"), ("몰B", 1050000, "무료배송")],
        low=1000000,
    )
    routes = _routes_for_product(product_html=product_html, history={_MONTH: 1000000})
    order = _detail_listing("111111111_0000000001", price="12000000", terms="서울")
    fact = _fact(unit_price_krw=Decimal("12000000"), comparable_amount_krw=Decimal("24000000"))
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order}, facts=[fact])
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is None


def test_observed_price_equals_headline_with_fallback(tmp_path: Path) -> None:
    product_html = _product_html(
        offers=[("몰A", 25000000, "무료배송"), ("몰B", 78474000, "무료배송")],
        low=25000000,
    )
    routes = _routes_for_product(product_html=product_html, history={_MONTH: 78474000})
    order = _detail_listing("111111111_0000000001", price="90000000", terms="서울")
    fact = _fact(unit_price_krw=Decimal("90000000"), comparable_amount_krw=Decimal("180000000"))
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order}, facts=[fact])
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    proof = bundle.lines[0].price_evidence[0]
    assert proof.observed_price_krw == 25000000
    assert bundle.lines[0].benchmark_unit_price_krw > Decimal(proof.observed_price_krw)


def test_shop_page_is_review_only(tmp_path: Path) -> None:
    shop = ShopSpec("shopexample.kr", "cafe24")
    evidence = _evidence(shops=(shop,))
    search_empty = "<html><body><ul></ul></body></html>"
    shop_search = '<a href="/product/detail.html?product_no=55">p</a>'
    shop_product = (
        '<meta property="product:price:currency" content="KRW">'
        '<meta property="product:price:amount" content="1000000">'
        '<meta property="og:title" content="Epson EB-PU2216B 프로젝터">'
    )
    routes = {
        "search.danawa.com": (200, search_empty),
        "shopexample.kr/product/search": (200, shop_search),
        "shopexample.kr/product/detail": (200, shop_product),
    }
    order = _detail_listing("111111111_0000000001", price="1400000", terms="서울")
    fact = _fact(unit_price_krw=Decimal("1400000"), comparable_amount_krw=Decimal("2800000"))
    ds = _write_dataset(tmp_path, [_base_row()])
    raw = _FakeRaw({order.listing_id: order})
    seen: list = []
    fetch = _fetcher(routes, seen, evidence)
    item = _item_detail([fact])
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, evidence, AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.review_required is True


def test_unavailable_or_model_missing_yields_nothing(tmp_path: Path) -> None:
    ds = _write_dataset(tmp_path, [_base_row()])
    raw = _FakeRaw({})
    seen: list = []
    fetch = _fetcher({}, seen)
    try:
        assert (
            build_retail_bundle(_item_detail([_fact()], state=ItemState.CLAIMED), ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
            is None
        )
        assert (
            build_retail_bundle(_item_detail([_fact()], model=""), ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
            is None
        )
        assert (
            build_retail_bundle(_item_detail([_fact()], model=None), ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
            is None
        )
    finally:
        ds.close()
    assert seen == []


def test_unit_not_single_piece_rejects(tmp_path: Path) -> None:
    routes = _routes_for_product(history={_MONTH: 2800000})
    order = _detail_listing("111111111_0000000001", price="3500000")
    fact = _fact(unit="조")
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order}, facts=[fact])
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is None


def test_placeholder_unit_is_not_single_piece() -> None:
    from src.savetax.config import EvidenceConfig
    from src.savetax.retail import _is_single_piece

    assert _is_single_piece("미상", EvidenceConfig()) is False
    assert _is_single_piece("EA", EvidenceConfig()) is True
    assert _is_single_piece("ea", EvidenceConfig()) is True


def test_nonscorable_and_dateless_facts_reject(tmp_path: Path) -> None:
    routes = _routes_for_product(history={_MONTH: 2800000})
    order = _detail_listing("111111111_0000000001", price="3500000")
    facts = [_fact(fact_id="BAD1", scorable=False), _fact(fact_id="BAD2", request_date=None), _fact(fact_id="F1")]
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order}, facts=facts)
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    reasons = {entry.fact_id: entry.reason for entry in bundle.rejected}
    assert reasons["BAD1"] == "not_scorable"
    assert reasons["BAD2"] == "no_request_date"


def test_empty_normalized_brand_is_false() -> None:
    assert brand_matches("전자", "이름", "Epson", _evidence()) is False
    assert brand_matches("Epson", "이름", "주식회사", _evidence()) is False
    assert brand_matches("(주문자상표부착) Epson", "Epson 물건", "Epson", _evidence()) is True


def test_benchmark_without_offers_is_none() -> None:
    assert current_benchmark(_product(low=100, offers=[]), {}, ("2024-01", "2026-12"), _evidence()) is None


def test_single_offer_without_history_is_none() -> None:
    product = _product(low=100000, offers=[("몰A", 100000, 0)])
    assert current_benchmark(product, {}, ("2025-01", "2025-02"), _evidence()) is None


def test_bad_model_pattern_yields_nothing(tmp_path: Path) -> None:
    ds = _write_dataset(tmp_path, [_base_row()])
    raw = _FakeRaw({})
    seen: list = []
    fetch = _fetcher({}, seen)
    try:
        assert (
            build_retail_bundle(_item_detail([_fact()], model="(CN)"), ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
            is None
        )
    finally:
        ds.close()
    assert seen == []


def test_unknown_contract_has_no_order_rows(tmp_path: Path) -> None:
    routes = _routes_for_product(history={_MONTH: 2800000})
    order = _detail_listing("111111111_0000000001", price="3500000")
    fact = _fact(contract_no="999999999_0")
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order}, facts=[fact])
    try:
        assert build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY) is None
    finally:
        ds.close()


def test_too_many_order_rows_rejects(tmp_path: Path) -> None:
    routes = _routes_for_product(history={_MONTH: 2800000})
    rows = [
        _base_row(listing_id="111111111_0000000001", price=3500000),
        _base_row(
            listing_id="111111111_0000000002",
            price=3500000,
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
        ),
    ]
    order_a = _detail_listing("111111111_0000000001", price="3500000")
    order_b = _detail_listing("111111111_0000000002", price="3500000")
    ds = _write_dataset(tmp_path, rows)
    raw = _FakeRaw({order_a.listing_id: order_a, order_b.listing_id: order_b})
    seen: list = []
    fetch = _fetcher(routes, seen)
    item = _item_detail([_fact()])
    try:
        assert (
            build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(max_condition_listings=1), today=_TODAY)  # type: ignore[arg-type]
            is None
        )
    finally:
        ds.close()


def test_wrong_raw_identity_rejects(tmp_path: Path) -> None:
    routes = _routes_for_product(history={_MONTH: 2800000})
    bad = _detail_listing("111111111_0000000001", price="3500000")
    bad = ListingDetail(
        listing_id="wrong",
        item_id=bad.item_id,
        unit=bad.unit,
        price_krw=bad.price_krw,
        price_kind=bad.price_kind,
        contract_no=bad.contract_no,
        contract_type=bad.contract_type,
        contract_begin=bad.contract_begin,
        contract_end=bad.contract_end,
        vendor_name=bad.vendor_name,
        manufacturer=bad.manufacturer,
        model=bad.model,
        category=bad.category,
        delivery_terms=bad.delivery_terms,
        supply_region=bad.supply_region,
        release=bad.release,
    )
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {"111111111_0000000001": bad})
    try:
        assert build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY) is None
    finally:
        ds.close()


def test_blank_order_terms_rejects(tmp_path: Path) -> None:
    routes = _routes_for_product(history={_MONTH: 2800000})
    order = _detail_listing("111111111_0000000001", price="3500000", terms="   ", region=None)
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order})
    try:
        assert build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY) is None
    finally:
        ds.close()


def test_missing_region_rejects(tmp_path: Path) -> None:
    routes = _routes_for_product(history={_MONTH: 2800000})
    order = _detail_listing("111111111_0000000001", price="3500000", terms="서울", region=None)
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order})
    try:
        assert build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY) is None
    finally:
        ds.close()


def test_ambiguous_order_conditions_reject(tmp_path: Path) -> None:
    routes = _routes_for_product(history={_MONTH: 2800000})
    rows = [
        _base_row(listing_id="111111111_0000000001", price=3500000),
        _base_row(
            listing_id="111111111_0000000002",
            price=3500000,
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
        ),
    ]
    order_a = _detail_listing("111111111_0000000001", price="3500000", terms="A")
    order_b = _detail_listing("111111111_0000000002", price="3500000", terms="B")
    ds = _write_dataset(tmp_path, rows)
    raw = _FakeRaw({order_a.listing_id: order_a, order_b.listing_id: order_b})
    seen: list = []
    fetch = _fetcher(routes, seen)
    item = _item_detail([_fact(unit_price_krw=Decimal("9999999"), comparable_amount_krw=Decimal("19999998"))])
    try:
        assert build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY) is None
    finally:
        ds.close()


def test_mixed_shipping_quote_and_caveat(tmp_path: Path) -> None:
    product_html = _product_html(
        offers=[("몰A", 2800000, "배송비 별도"), ("몰B", 2900000, "2,500원")],
        low=2800000,
    )
    routes = _routes_for_product(product_html=product_html, history={_MONTH: 2800000})
    order = _detail_listing("111111111_0000000001", price="3500000")
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order})
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    assert any("배송비 미표기" in c for c in bundle.caveats)
    assert "배송비 2,500원" in bundle.lines[0].price_evidence[0].quote


def test_empty_offers_bundle_is_none(tmp_path: Path) -> None:
    product_html = (
        '<script type="application/ld+json">{"@type": "Product", "name": "Epson EB-PU2216B", '
        '"brand": {"name": "Epson"}, "description": "레이저 프로젝터", '
        '"offers": {"lowPrice": 2800000, "offerCount": 0}}</script>'
    )
    routes = _routes_for_product(product_html=product_html, history={_MONTH: 2800000})
    order = _detail_listing("111111111_0000000001", price="3500000")
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order})
    try:
        assert build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY) is None
    finally:
        ds.close()


def test_single_offer_bundle_uses_stable_history(tmp_path: Path) -> None:
    product_html = _product_html(
        offers=[("몰A", 100000, "무료배송")],
        low=100000,
        count=1,
    )
    routes = _routes_for_product(product_html=product_html, history={_MONTH: 100000, "2025-09": 110000})
    order = _detail_listing("111111111_0000000001", price="140000", terms="서울")
    fact = _fact(unit_price_krw=Decimal("140000"), comparable_amount_krw=Decimal("280000"))
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order}, facts=[fact])
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("100000")
    assert bundle.review_required is True


def test_catalog_gates_reject(tmp_path: Path) -> None:
    order = _detail_listing("111111111_0000000001", price="3500000")
    base_routes = _routes_for_product(history={_MONTH: 2800000})

    def run_with(product_html: str | None, status: int = 200) -> Any:
        routes = dict(base_routes)
        routes["prod.danawa.com/info"] = (status, product_html or "")
        ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order})
        try:
            return build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
        finally:
            ds.close()

    assert run_with(None, status=404) is None
    assert run_with("<html>no json-ld</html>") is None
    assert run_with(_product_html(name="다른 제품명")) is None
    assert run_with(_product_html(description="액세서리 전용 거치대 모델")) is None
    bad_history = dict(base_routes)
    bad_history["getProductPriceList"] = (200, "not json")
    ds, raw, fetch, _seen, item = _setup(tmp_path, bad_history, {order.listing_id: order})
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is None
    discontinued = _product_html(low=None)
    assert run_with(discontinued) is None


def test_zero_expected_maps_to_no_margin(tmp_path: Path) -> None:
    product_html = _product_html(
        offers=[("몰A", 1000000, "무료배송"), ("몰B", 1050000, "무료배송")],
        low=1000000,
    )
    routes = _routes_for_product(product_html=product_html, history={_MONTH: 1000000})
    order = _detail_listing("111111111_0000000001", price="1000000", terms="서울")
    fact = _fact(unit_price_krw=Decimal("1000000"), quantity=Decimal("2"), comparable_amount_krw=Decimal("2000000"))
    ds, raw, fetch, _seen, item = _setup(
        tmp_path, routes, {order.listing_id: order}, facts=[fact],
    )
    try:
        bundle = build_retail_bundle(
            item, ds, raw, fetch, _evidence(retail_markup_threshold=Decimal("1.0")), AuditConfig(), today=_TODAY  # type: ignore[arg-type]
        )
    finally:
        ds.close()
    assert bundle is None


def test_shop_search_failure_pauses(tmp_path: Path) -> None:
    import socket

    shop = ShopSpec("fails.example", "cafe24")
    evidence = _evidence(shops=(shop,))
    routes = {"search.danawa.com": (200, "<html><body><ul></ul></body></html>")}

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        if "fails.example" in url:
            raise socket.gaierror("dns fail")
        for key, (status, body) in routes.items():
            if key in url:
                return HttpResponse(status=status, headers={}, body=body.encode(), elapsed_seconds=0.01)
        return HttpResponse(status=404, headers={}, body=b"", elapsed_seconds=0.01)

    fetch = PublicFetcher(
        SaveTaxConfig(), evidence, transport=transport, sleep=lambda _s: None, clock=lambda: 0.0  # type: ignore[arg-type]
    )
    ds = _write_dataset(tmp_path, [_base_row()])
    raw = _FakeRaw({})
    item = _item_detail([_fact()])
    try:
        with pytest.raises(NetworkError):
            build_retail_bundle(item, ds, raw, fetch, evidence, AuditConfig(), today=_TODAY)
    finally:
        ds.close()


def test_shop_links_with_bad_pages_are_skipped(tmp_path: Path) -> None:
    shop = ShopSpec("shopexample.kr", "cafe24")
    evidence = _evidence(shops=(shop,))
    search_empty = "<html><body><ul></ul></body></html>"
    shop_search = (
        '<a href="/product/detail.html?product_no=1">a</a>'
        '<a href="/product/detail.html?product_no=2">b</a>'
        '<a href="/product/detail.html?product_no=3">c</a>'
        '<a href="/product/detail.html?product_no=4">d</a>'
        '<a href="/product/detail.html?product_no=5">e</a>'
    )
    good_product = (
        '<meta property="product:price:currency" content="KRW">'
        '<meta property="product:price:amount" content="1000000">'
        '<meta property="og:title" content="Epson EB-PU2216B 프로젝터">'
    )
    routes = {
        "search.danawa.com": (200, search_empty),
        "shopexample.kr/product/search": (200, shop_search),
        "product_no=1": (404, ""),
        "product_no=2": (200, "<title>no price</title>"),
        "product_no=3": (200, good_product.replace("EB-PU2216B", "OTHER-MODEL")),
        "product_no=4": (200, good_product.replace("프로젝터", "액세서리 거치대")),
        "product_no=5": (200, good_product),
    }
    order = _detail_listing("111111111_0000000001", price="1400000", terms="서울")
    fact = _fact(unit_price_krw=Decimal("1400000"), comparable_amount_krw=Decimal("2800000"))
    ds = _write_dataset(tmp_path, [_base_row()])
    raw = _FakeRaw({order.listing_id: order})
    seen: list = []
    fetch = _fetcher(routes, seen, evidence)
    item = _item_detail([fact])
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, evidence, AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.review_required is True


def test_shop_mixed_facts_and_margin(tmp_path: Path) -> None:
    shop = ShopSpec("shopexample.kr", "cafe24")
    evidence = _evidence(shops=(shop,))
    search_empty = "<html><body><ul></ul></body></html>"
    shop_search = '<a href="/product/detail.html?product_no=55">p</a>'
    shop_product = (
        '<meta property="product:price:currency" content="KRW">'
        '<meta property="product:price:amount" content="1000000">'
        '<meta property="og:title" content="Epson EB-PU2216B 프로젝터">'
    )
    routes = {
        "search.danawa.com": (200, search_empty),
        "shopexample.kr/product/search": (200, shop_search),
        "shopexample.kr/product/detail": (200, shop_product),
    }
    order = _detail_listing("111111111_0000000001", price="1400000", terms="현장설치도")
    facts = [
        _fact(fact_id="BAD1", scorable=False),
        _fact(fact_id="BAD2", request_date=None),
        _fact(fact_id="BAD3", unit="조"),
        _fact(fact_id="THIN", unit_price_krw=Decimal("1000000"), quantity=Decimal("1"), comparable_amount_krw=Decimal("1000000")),
        _fact(fact_id="F1", unit_price_krw=Decimal("1400000"), comparable_amount_krw=Decimal("2800000")),
    ]
    ds = _write_dataset(tmp_path, [_base_row()])
    raw = _FakeRaw({order.listing_id: order})
    seen: list = []
    fetch = _fetcher(routes, seen, evidence)
    item = _item_detail(facts)
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, evidence, AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    reasons = {entry.fact_id: entry.reason for entry in bundle.rejected}
    assert reasons["BAD1"] == "not_scorable"
    assert reasons["BAD2"] == "no_request_date"
    assert reasons["BAD3"] == "unit_not_single_piece"
    assert reasons["THIN"] == "no_margin"
    assert any("허용률" in c for c in bundle.caveats)


def test_shop_product_page_error_pauses(tmp_path: Path) -> None:
    import socket

    shop = ShopSpec("shopexample.kr", "cafe24")
    evidence = _evidence(shops=(shop,))
    search_empty = "<html><body><ul></ul></body></html>"
    shop_search = '<a href="/product/detail.html?product_no=55">p</a>'
    good_product = (
        '<meta property="product:price:currency" content="KRW">'
        '<meta property="product:price:amount" content="1000000">'
        '<meta property="og:title" content="Epson EB-PU2216B 프로젝터">'
    )
    routes = {
        "search.danawa.com": (200, search_empty),
        "shopexample.kr/product/search": (200, shop_search),
    }

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        if "product/detail" in url:
            raise socket.gaierror("dns fail")
        for key, (status, body) in routes.items():
            if key in url:
                return HttpResponse(status=status, headers={}, body=body.encode(), elapsed_seconds=0.01)
        return HttpResponse(status=404, headers={}, body=b"", elapsed_seconds=0.01)

    fetch = PublicFetcher(
        SaveTaxConfig(), evidence, transport=transport, sleep=lambda _s: None, clock=lambda: 0.0  # type: ignore[arg-type]
    )
    order = _detail_listing("111111111_0000000001", price="1400000", terms="서울")
    fact = _fact(unit_price_krw=Decimal("1400000"), comparable_amount_krw=Decimal("2800000"))
    ds = _write_dataset(tmp_path, [_base_row()])
    raw = _FakeRaw({order.listing_id: order})
    item = _item_detail([fact])
    try:
        with pytest.raises(NetworkError):
            build_retail_bundle(item, ds, raw, fetch, evidence, AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert good_product is not None


def test_shop_unresolvable_order_rejects(tmp_path: Path) -> None:
    shop = ShopSpec("shopexample.kr", "cafe24")
    evidence = _evidence(shops=(shop,))
    routes = {
        "search.danawa.com": (200, "<html><body><ul></ul></body></html>"),
        "shopexample.kr/product/search": (200, '<a href="/product/detail.html?product_no=55">p</a>'),
        "shopexample.kr/product/detail": (
            200,
            '<meta property="product:price:currency" content="KRW">'
            '<meta property="product:price:amount" content="1000000">'
            '<meta property="og:title" content="Epson EB-PU2216B 프로젝터">',
        ),
    }
    order = _detail_listing("111111111_0000000001", price="1400000", terms="서울")
    fact = _fact(contract_no="999999999_0")
    ds = _write_dataset(tmp_path, [_base_row()])
    raw = _FakeRaw({order.listing_id: order})
    seen: list = []
    fetch = _fetcher(routes, seen, evidence)
    item = _item_detail([fact])
    try:
        assert build_retail_bundle(item, ds, raw, fetch, evidence, AuditConfig(), today=_TODAY) is None
    finally:
        ds.close()


def test_shop_zero_expected_maps_to_no_margin(tmp_path: Path) -> None:
    shop = ShopSpec("shopexample.kr", "cafe24")
    evidence = _evidence(shops=(shop,), retail_markup_threshold=Decimal("1.0"))
    routes = {
        "search.danawa.com": (200, "<html><body><ul></ul></body></html>"),
        "shopexample.kr/product/search": (200, '<a href="/product/detail.html?product_no=55">p</a>'),
        "shopexample.kr/product/detail": (
            200,
            '<meta property="product:price:currency" content="KRW">'
            '<meta property="product:price:amount" content="1000000">'
            '<meta property="og:title" content="Epson EB-PU2216B 프로젝터">',
        ),
    }
    order = _detail_listing("111111111_0000000001", price="1000000", terms="서울")
    fact = _fact(unit_price_krw=Decimal("1000000"), quantity=Decimal("2"), comparable_amount_krw=Decimal("2000000"))
    ds = _write_dataset(tmp_path, [_base_row()])
    raw = _FakeRaw({order.listing_id: order})
    seen: list = []
    fetch = _fetcher(routes, seen, evidence)
    item = _item_detail([fact])
    try:
        assert build_retail_bundle(item, ds, raw, fetch, evidence, AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_catalog_rejection_does_not_search_other_sources(tmp_path: Path) -> None:
    routes = _routes_for_product(product_html=_product_html(description="액세서리"))
    ds, raw, fetch, seen, item = _setup(tmp_path, routes, {})
    try:
        assert build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY) is None
    finally:
        ds.close()
    assert len(seen) == 2


def test_credential_shaped_model_never_reaches_public_url(tmp_path: Path) -> None:
    ds, raw, fetch, seen, item = _setup(tmp_path, {}, {}, detail=_item_detail([], model="savetax_test-token"))
    try:
        assert build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY) is None
    finally:
        ds.close()
    assert seen == []


def test_install_delta_uses_matching_listing_and_rounds_up() -> None:
    from src.savetax.retail import _installation

    rows = [_detail_listing("a", price="2420000.1", terms="현장설치도"),
            _detail_listing("b", price="9000000", terms="현장설치도"),
            _detail_listing("c", price="2200000", terms="납품장소도")]
    assert _installation(Decimal("2420000.1"), "현장설치도", 1000000, rows, _evidence()) == (220001, False)
    assert _installation(Decimal("2000000"), "현장설치도", 1000000,
                         [_detail_listing("a", price="2000000", terms="현장설치도"), rows[2]], _evidence()) == (150000, True)


def test_fallback_shipping_flag_comes_from_lowest_row(tmp_path: Path) -> None:
    html = _product_html(low=1000000, offers=[("A", 1000000, "배송비 별도"), ("B", 2000000, "무료배송")])
    routes = _routes_for_product(product_html=html, history={_MONTH: 2000000})
    order = _detail_listing("111111111_0000000001", price="3500000")
    ds, raw, fetch, _seen, item = _setup(tmp_path, routes, {order.listing_id: order})
    try:
        bundle = build_retail_bundle(item, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].price_evidence[0].shipping_included is False
