"""Invariant guards for the cross-contract evidence lane."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pytest

from src.savetax.competition import Fact, ItemDetail, ItemState
from src.savetax.config import AuditConfig, EvidenceConfig, ListingSeries, SaveTaxConfig, SeriesBlock, VariantPolicy
from src.savetax.crosscontract import (
    CrossCandidate,
    build_cross_bundle,
    find_cross_candidates,
    parse_listing_page,
    region_covers,
)
from src.savetax.dataset import Dataset
from src.savetax.errors import DataIntegrityError
from src.savetax.http import HttpResponse
from src.savetax.publicweb import PublicFetcher
from src.savetax.rawapi import ListingDetail

_TODAY = date(2026, 10, 13)
_REQUEST = date(2025, 10, 12)

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
        "name": "물탱크",
        "detail_category": "물탱크",
        "category_l_name": "기계",
        "manufacturer": "M",
        "model": "T-100",
        "unit": "대",
        "price": 1200,
        "contract_begin": "2025-01-01",
        "contract_end": "2026-12-31",
        "vendor_name": "업체A",
        "dlv_key": 1,
        "dlv_lines": 1,
        "dlv_qty": 10,
        "dlv_unit": "대",
        "dlv_amount": 12000,
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


def _detail(
    listing_id: str,
    item_id: str = "11111111",
    unit: str | None = "대",
    price: str = "2508000",
    contract_no: str | None = None,
    terms: str | None = "현장설치도",
    region: str | None = "서울",
    begin: str | None = "20250101",
    end: str | None = "20261231",
    kind: str = "contract",
    contract_date: date | None = _REQUEST,
    change_order: str | None = None,
) -> ListingDetail:
    return ListingDetail(
        listing_id=listing_id,
        item_id=item_id,
        unit=unit,
        price_krw=Decimal(price),
        price_kind=kind,
        contract_no=contract_no if contract_no is not None else listing_id[:11],
        contract_type=None,
        contract_begin=date(int(begin[0:4]), int(begin[4:6]), int(begin[6:8])) if begin else None,
        contract_end=date(int(end[0:4]), int(end[4:6]), int(end[6:8])) if end else None,
        vendor_name="벤더",
        manufacturer=None,
        model=None,
        category=None,
        delivery_terms=terms,
        supply_region=region,
        release="rtest",
        contract_date=contract_date,
        change_order=change_order,
    )


class _FakeRaw:
    def __init__(
        self,
        mapping: dict[str, ListingDetail | None],
        deliveries: dict[str, tuple | None] | None = None,
    ) -> None:
        self.mapping = mapping
        self.calls: list[str] = []
        self.delivery_calls: list[str] = []
        self.deliveries = deliveries or {}

    def get_listing(self, listing_id: str) -> ListingDetail | None:
        self.calls.append(listing_id)
        return self.mapping.get(listing_id)

    def list_deliveries(self, listing_id: str):  # type: ignore[no-untyped-def]
        self.delivery_calls.append(listing_id)
        if listing_id in self.deliveries:
            return self.deliveries[listing_id]
        return None


def _fetcher(pages: dict[str, tuple[int, str]], seen: list) -> PublicFetcher:
    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        seen.append((url, dict(headers)))  # type: ignore[arg-type]
        listing_id = url.rsplit("/", 1)[-1]
        status, text = pages.get(listing_id, (404, ""))
        return HttpResponse(status=status, headers={}, body=text.encode(), elapsed_seconds=0.01)

    return PublicFetcher(
        SaveTaxConfig(),
        EvidenceConfig(),
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
        "unit_price_krw": Decimal("2736000"),
        "quantity": Decimal("2"),
        "comparable_amount_krw": Decimal("5472000"),
        "scorable": True,
        "exclusion_reasons": (),
    }
    base.update(over)
    return Fact(**base)


def _item_detail(facts: list[Fact], state: ItemState = ItemState.AVAILABLE) -> ItemDetail:
    return ItemDetail(
        item_id="11111111",
        state=state,
        name=None,
        model=None,
        manufacturer=None,
        category=None,
        source_release="rtest",
        facts_total=len(facts),
        facts_truncated=False,
        facts=tuple(facts),
    )


_PAGE = (
    "멀티스크린컴퓨터 사진 불러오는 중 멀티스크린컴퓨터 단위 대 물품식별번호 25051267 "
    "계약품목관리번호 002170355_1140000338 계약 단가 2,508,000원 / 대 공급 업체 주식회사 그린아이티코리아 "
    "납품요구 아직 납품요구 없음 계약 기간 2021.11.8 ~ 2027.6.6 납품 조건 지정 장소 납품·설치 포함 "
    "납품요구 기록 현재 조회 자료에서는 이 계약의 납품요구를 찾지 못했습니다. "
    "공급 지역 전지역(제주 운임비 별도) 인증 사회적가치실현기업"
)


def test_cheaper_listing_of_another_contract_is_found(tmp_path: Path) -> None:
    rows = [
        _base_row(price=1200000, dlv_amount=12000000, dlv_qty=10),
        _base_row(
            listing_id="222222222_0000000001",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            price=1000000,
            vendor_name="업체B",
        ),
    ]
    ds = _write_dataset(tmp_path, rows)
    try:
        found = find_cross_candidates(ds, _evidence(), AuditConfig())
    finally:
        ds.close()
    assert len(found) == 1
    assert found[0].benchmark_listing_ids == ("222222222_0000000001",)


def test_same_contract_is_never_a_benchmark(tmp_path: Path) -> None:
    rows = [
        _base_row(),
        _base_row(
            listing_id="111111111_0000000002",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            price=100,
            vendor_name="업체B",
        ),
    ]
    ds = _write_dataset(tmp_path, rows)
    try:
        assert find_cross_candidates(ds, _evidence(), AuditConfig()) == []
    finally:
        ds.close()


def test_threshold_boundary_is_inclusive(tmp_path: Path) -> None:
    def run(amount: int) -> int:
        rows = [
            _base_row(dlv_amount=amount, dlv_qty=10, price=1200000),
            _base_row(
                listing_id="222222222_0000000001",
                dlv_key=None,
                dlv_lines=None,
                dlv_qty=None,
                dlv_unit=None,
                dlv_amount=None,
                dlv_first_date=None,
                dlv_last_date=None,
                price=1000000,
                vendor_name="업체B",
            ),
        ]
        ds = _write_dataset(tmp_path, rows)
        try:
            return len(find_cross_candidates(ds, _evidence(), AuditConfig()))
        finally:
            ds.close()

    assert run(12_000_000) == 1
    assert run(11_999_990) == 0


def test_different_unit_is_excluded(tmp_path: Path) -> None:
    rows = [
        _base_row(price=1200000, dlv_amount=12000000, dlv_qty=10),
        _base_row(
            listing_id="222222222_0000000001",
            unit="개",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            price=1000000,
            vendor_name="업체B",
        ),
    ]
    ds = _write_dataset(tmp_path, rows)
    try:
        assert find_cross_candidates(ds, _evidence(), AuditConfig()) == []
    finally:
        ds.close()


def test_non_overlapping_window_is_excluded(tmp_path: Path) -> None:
    rows = [
        _base_row(price=1200000, dlv_amount=12000000, dlv_qty=10),
        _base_row(
            listing_id="222222222_0000000001",
            contract_end="2025-10-10",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            price=1000000,
            vendor_name="업체B",
        ),
    ]
    ds = _write_dataset(tmp_path, rows)
    try:
        assert find_cross_candidates(ds, _evidence(), AuditConfig()) == []
    finally:
        ds.close()


def test_gap_below_minimum_is_excluded(tmp_path: Path) -> None:
    rows = [
        _base_row(dlv_amount=1299999, dlv_qty=1, price=1299999),
        _base_row(
            listing_id="222222222_0000000001",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            price=1000000,
            vendor_name="업체B",
        ),
    ]
    ds = _write_dataset(tmp_path, rows)
    try:
        assert find_cross_candidates(ds, _evidence(min_item_expected_krw=300000), AuditConfig()) == []
    finally:
        ds.close()


def test_benchmarks_are_capped_and_ordered(tmp_path: Path) -> None:
    rows = [_base_row(dlv_amount=10_000_000, dlv_qty=1, price=10_000_000)]
    for i, price in enumerate([500, 400, 300, 200, 100, 600, 700]):
        rows.append(
            _base_row(
                listing_id=f"22222222{i}_0000000001",
                dlv_key=None,
                dlv_lines=None,
                dlv_qty=None,
                dlv_unit=None,
                dlv_amount=None,
                dlv_first_date=None,
                dlv_last_date=None,
                price=price,
                vendor_name="업체B",
            )
        )
    ds = _write_dataset(tmp_path, rows)
    try:
        found = find_cross_candidates(ds, _evidence(max_benchmarks_per_item=5), AuditConfig())
    finally:
        ds.close()
    assert len(found) == 1
    assert found[0].benchmark_listing_ids == tuple(f"22222222{i}_0000000001" for i in (4, 3, 2, 1, 0))


def test_all_region_covers_province() -> None:
    assert region_covers("전지역(제주 운임비 별도)", "경상북도", _evidence()) is True


def test_province_does_not_cover_another() -> None:
    assert region_covers("경상북도", "전라남도", _evidence()) is False
    assert region_covers(None, "경상북도", _evidence()) is False
    assert region_covers("전지역", None, _evidence()) is False


def test_listing_page_fields_are_parsed() -> None:
    page = parse_listing_page(_PAGE)
    assert page.listing_id == "002170355_1140000338"
    assert page.item_id == "25051267"
    assert page.unit_price_krw == 2508000
    assert page.unit == "대"
    assert page.vendor_name == "주식회사 그린아이티코리아"
    assert page.contract_period == "2021.11.8 ~ 2027.6.6"
    assert page.delivery_terms == "지정 장소 납품·설치 포함"
    assert page.supply_region == "전지역(제주 운임비 별도)"


def test_sibling_prices_are_not_read() -> None:
    text = _PAGE + " 같은 물건의 다른 등재 002270042_1280001418 납품 단가 2,736,000원 계약 단가 9,999,999원 / 대"
    assert parse_listing_page(text).unit_price_krw == 2508000


def test_page_without_price_is_rejected() -> None:
    with pytest.raises(DataIntegrityError):
        parse_listing_page("단위 대 물품식별번호 1 계약품목관리번호 2 공급 업체 V")


def _bundle_dataset(
    tmp_path: Path, order_price: str = "2736000",
    benchmarks: list[ListingDetail] | None = None,
) -> Dataset:
    rows = [
        _base_row(listing_id="111111111_0000000001", price=2736000, dlv_amount=5472000, dlv_qty=2),
    ]
    if benchmarks is None:
        benchmarks = [_detail("222222222_0000000001"), _detail("333333333_0000000002")]
    rows.extend(
        _base_row(
            listing_id=b.listing_id, price=b.price_krw, dlv_key=None,
            dlv_lines=None, dlv_qty=None, dlv_unit=None, dlv_amount=None,
            dlv_first_date=None, dlv_last_date=None,
        )
        for b in benchmarks
    )
    return _write_dataset(tmp_path, rows)


def _bundle_setup(
    tmp_path: Path,
    raw_map: dict[str, ListingDetail | None],
    pages: dict[str, tuple[int, str]],
    facts: list[Fact] | None = None,
    candidate: CrossCandidate | None = None,
) -> tuple[Dataset, _FakeRaw, PublicFetcher, list, ItemDetail, CrossCandidate]:
    ds = _bundle_dataset(tmp_path, benchmarks=[
        b for lid, b in raw_map.items() if b is not None and lid != "111111111_0000000001"
    ])
    raw = _FakeRaw(raw_map)
    seen: list = []
    fetch = _fetcher(pages, seen)
    detail = _item_detail(facts if facts is not None else [_fact()])
    cand = candidate or CrossCandidate("11111111", 5472000, 456000, ("222222222_0000000001",))
    return ds, raw, fetch, seen, detail, cand


def test_admissible_fact_becomes_line_with_exact_b(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    pages = {"222222222_0000000001": (200, _PAGE.replace("002170355_1140000338", "222222222_0000000001").replace("25051267", "11111111").replace("2,508,000원 / 대", "2,508,000원 / 대").replace("전지역(제주 운임비 별도)", "서울"))}
    pages["222222222_0000000001"] = (
        200,
        "사진 불러오는 중 제품명 단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
        "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 2025.1.1 ~ 2026.12.31 "
        "납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음",
    )
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, pages)
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert len(bundle.lines) == 1
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2508000")
    assert bundle.lines[0].expected_krw == 456000
    assert bundle.lines[0].price_evidence[0].observed_price_krw == 2508000


def test_delivery_terms_mismatch_rejects(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="현장설치도", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="납품장소 하차도", region="서울")
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, {})
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is None


def test_request_date_outside_window_rejects(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울", end="20250101")
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, {})
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is None


def test_catalog_reference_is_dropped(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울", kind="catalog_reference")
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, {})
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is None


def test_ambiguous_ordering_conditions_reject(tmp_path: Path) -> None:
    ds = _write_dataset(
        tmp_path,
        [
            _base_row(listing_id="111111111_0000000001", price=2736000, dlv_amount=5472000, dlv_qty=2),
            _base_row(
                listing_id="111111111_0000000002",
                price=2736000,
                dlv_key=None,
                dlv_lines=None,
                dlv_qty=None,
                dlv_unit=None,
                dlv_amount=None,
                dlv_first_date=None,
                dlv_last_date=None,
            ),
        ],
    )
    order_a = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="A", region="서울")
    order_b = _detail("111111111_0000000002", contract_no="111111111_0", price="2736000", terms="B", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="A", region="서울")
    raw = _FakeRaw({order_a.listing_id: order_a, order_b.listing_id: order_b, bench.listing_id: bench})
    seen: list = []
    fetch = _fetcher({}, seen)
    detail = _item_detail([_fact()])
    cand = CrossCandidate("11111111", 5472000, 456000, ("222222222_0000000001",))
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is None


def test_discounted_order_falls_back_with_caveat(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="9999999", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    pages = {
        "222222222_0000000001": (
            200,
            "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
            "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음",
        )
    }
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, pages)
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert len(bundle.lines) == 1
    assert bundle.caveats != ()


def test_page_price_mismatch_falls_back(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    b1 = _detail("222222222_0000000001", price="2400000", terms="서울", region="서울")
    b2 = _detail("333333333_0000000002", price="2508000", terms="서울", region="서울")
    pages = {
        "222222222_0000000001": (
            200,
            "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
            "계약 단가 9,999,999원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음",
        ),
        "333333333_0000000002": (
            200,
            "단위 대 물품식별번호 11111111 계약품목관리번호 333333333_0000000002 "
            "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음",
        ),
    }
    ds = _bundle_dataset(tmp_path)
    raw = _FakeRaw({order.listing_id: order, b1.listing_id: b1, b2.listing_id: b2})
    seen: list = []
    fetch = _fetcher(pages, seen)
    detail = _item_detail([_fact()])
    cand = CrossCandidate("11111111", 5472000, 800000, ("222222222_0000000001", "333333333_0000000002"))
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2508000")


def test_unreadable_page_rejects(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(
        tmp_path, {order.listing_id: order, bench.listing_id: bench}, {"222222222_0000000001": (403, "")}
    )
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is None


def test_unavailable_item_yields_nothing(tmp_path: Path) -> None:
    ds = _bundle_dataset(tmp_path)
    raw = _FakeRaw({})
    seen: list = []
    fetch = _fetcher({}, seen)
    detail = _item_detail([_fact()], state=ItemState.CLAIMED)
    cand = CrossCandidate("11111111", 1, 1, ("222222222_0000000001",))
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()
    assert raw.calls == []
    assert seen == []


def test_total_below_minimum_yields_nothing(tmp_path: Path) -> None:
    small_fact = _fact(comparable_amount_krw=Decimal("2807999"), quantity=Decimal("1"), unit_price_krw=Decimal("2807999"))
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2807999", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    pages = {
        "222222222_0000000001": (
            200,
            "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
            "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음",
        )
    }
    ds = _write_dataset(
        tmp_path,
        [_base_row(listing_id="111111111_0000000001", price=2807999, dlv_amount=2807999, dlv_qty=1)],
    )
    raw = _FakeRaw({order.listing_id: order, bench.listing_id: bench})
    seen: list = []
    fetch = _fetcher(pages, seen)
    detail = _item_detail([small_fact])
    cand = CrossCandidate("11111111", 2807999, 299999, ("222222222_0000000001",))
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_quote_contains_only_page_text(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    pages = {
        "222222222_0000000001": (
            200,
            "사진 불러오는 중 제품명 단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
            "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음",
        )
    }
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, pages)
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    quote = bundle.lines[0].price_evidence[0].quote
    segments = quote.split(" / ")
    assert segments[-1].endswith("확인")
    for segment in segments[:-1]:
        assert any(
            segment.startswith(prefix)
            for prefix in ("제품명", "물품식별번호", "계약품목관리번호", "계약 단가", "공급 업체", "계약 기간", "납품 조건", "공급 지역", "비교 단위", "savetax.now")
        ) or segment in ("대", "개")


def test_no_credential_reaches_savetax_now(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    pages = {
        "222222222_0000000001": (
            200,
            "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
            "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음",
        )
    }
    ds, raw, fetch, seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, pages)
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    for _url, headers in seen:
        assert "Authorization" not in headers


def test_record_label_without_standalone_request_is_absent() -> None:
    page = parse_listing_page(
        "단위 대 물품식별번호 1 계약품목관리번호 2 계약 단가 1,000원 / 대 "
        "공급 업체 V 계약 기간 P 납품 조건 T 납품요구 기록 없음 공급 지역 S 인증 C"
    )
    assert page.delivery_terms == "T"


def test_quote_without_unit_uses_price_only(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    pages = {
        "222222222_0000000001": (
            200,
            "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
            "계약 단가 2,508,000원 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음",
        )
    }
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, pages)
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert "계약 단가 2,508,000원" in bundle.lines[0].price_evidence[0].quote


def test_mixed_scorable_facts_keep_line_with_rejections(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    pages = {
        "222222222_0000000001": (
            200,
            "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
            "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음",
        )
    }
    facts = [_fact(fact_id="BAD1", scorable=False), _fact(fact_id="BAD2", request_date=None), _fact(fact_id="F1")]
    ds, raw, fetch, _seen, _d, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, pages)
    detail = _item_detail(facts)
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    reasons = {r.fact_id: r.reason for r in bundle.rejected}
    assert reasons["BAD1"] == "not_scorable"
    assert reasons["BAD2"] == "no_request_date"


def test_all_facts_unscorable_returns_none(tmp_path: Path) -> None:
    ds = _bundle_dataset(tmp_path)
    raw = _FakeRaw({})
    seen: list = []
    fetch = _fetcher({}, seen)
    detail = _item_detail([_fact(fact_id="BAD", scorable=False)])
    cand = CrossCandidate("11111111", 1, 1, ())
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_missing_benchmark_is_dropped(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order}, {})
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_blank_terms_benchmark_is_dropped(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="   ", region="서울")
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, {})
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_benchmark_without_dates_is_dropped(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울", begin=None, end=None)
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, {})
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_unknown_contract_has_no_order_listings(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    ds, raw, fetch, _seen, _d, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, {})
    detail = _item_detail([_fact(contract_no="999999999_0")])
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_too_many_order_listings_rejects(tmp_path: Path) -> None:
    ds = _write_dataset(
        tmp_path,
        [
            _base_row(listing_id="111111111_0000000001", price=2736000, dlv_amount=5472000, dlv_qty=2),
            _base_row(listing_id="111111111_0000000002", price=2736000, dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None, dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
        ],
    )
    order_a = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    order_b = _detail("111111111_0000000002", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    raw = _FakeRaw({order_a.listing_id: order_a, order_b.listing_id: order_b, bench.listing_id: bench})
    seen: list = []
    fetch = _fetcher({}, seen)
    detail = _item_detail([_fact()])
    cand = CrossCandidate("11111111", 5472000, 456000, ("222222222_0000000001",))
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(max_condition_listings=1), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is None


def test_unserved_order_listing_rejects(tmp_path: Path) -> None:
    ds = _bundle_dataset(tmp_path)
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    raw = _FakeRaw({bench.listing_id: bench})
    seen: list = []
    fetch = _fetcher({}, seen)
    detail = _item_detail([_fact()])
    cand = CrossCandidate("11111111", 5472000, 456000, ("222222222_0000000001",))
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_order_listing_without_conditions_rejects(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms=None, region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, {})
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_same_contract_benchmark_is_inadmissible(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", contract_no="111111111_0", price="2508000", terms="서울", region="서울")
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, {})
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_unit_mismatch_benchmark_is_inadmissible(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", unit="개", price="2508000", terms="서울", region="서울")
    ds, raw, fetch, _seen, _d, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, {})
    detail = _item_detail([_fact(unit="대")])
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_region_mismatch_rejects(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="부산")
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, {})
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_margin_below_threshold_rejects(tmp_path: Path) -> None:
    thin = _fact(unit_price_krw=Decimal("2600000"), comparable_amount_krw=Decimal("5200000"))
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2600000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    ds = _write_dataset(tmp_path, [_base_row(listing_id="111111111_0000000001", price=2600000, dlv_amount=5200000, dlv_qty=2)])
    raw = _FakeRaw({order.listing_id: order, bench.listing_id: bench})
    seen: list = []
    fetch = _fetcher({}, seen)
    detail = _item_detail([thin])
    cand = CrossCandidate("11111111", 5200000, 200000, ("222222222_0000000001",))
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_unparseable_page_rejects(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(
        tmp_path, {order.listing_id: order, bench.listing_id: bench}, {"222222222_0000000001": (200, "단위 대")}
    )
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_identity_mismatch_rejects(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    bad_page = (
        "단위 대 물품식별번호 99999999 계약품목관리번호 222222222_0000000001 "
        "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음"
    )
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(
        tmp_path, {order.listing_id: order, bench.listing_id: bench}, {"222222222_0000000001": (200, bad_page)}
    )
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_price_mismatch_without_fallback_rejects(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    bad_page = (
        "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
        "계약 단가 9,999,999원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음"
    )
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(
        tmp_path, {order.listing_id: order, bench.listing_id: bench}, {"222222222_0000000001": (200, bad_page)}
    )
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_zero_expected_line_is_rejected(tmp_path: Path) -> None:
    flat = _fact(unit_price_krw=Decimal("2760000"), comparable_amount_krw=Decimal("5000000"), quantity=Decimal("2"))
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2760000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    pages = {
        "222222222_0000000001": (
            200,
            "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
            "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음",
        )
    }
    ds = _write_dataset(tmp_path, [_base_row(listing_id="111111111_0000000001", price=2760000, dlv_amount=5520000, dlv_qty=2)])
    raw = _FakeRaw({order.listing_id: order, bench.listing_id: bench})
    seen: list = []
    fetch = _fetcher(pages, seen)
    detail = _item_detail([flat])
    cand = CrossCandidate("11111111", 5520000, 500000, ("222222222_0000000001",))
    try:
        assert build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.04")), AuditConfig(), today=_TODAY) is None  # type: ignore[arg-type]
    finally:
        ds.close()


def test_unreadable_page_falls_back(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    b1 = _detail("222222222_0000000001", price="2400000", terms="서울", region="서울")
    b2 = _detail("333333333_0000000002", price="2508000", terms="서울", region="서울")
    good = (
        "단위 대 물품식별번호 11111111 계약품목관리번호 333333333_0000000002 "
        "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음"
    )
    ds = _bundle_dataset(tmp_path)
    raw = _FakeRaw({order.listing_id: order, b1.listing_id: b1, b2.listing_id: b2})
    seen: list = []
    pages = {"333333333_0000000002": (200, good)}
    fetch = _fetcher(pages, seen)
    seen_pages = pages

    def _transport(url: str, headers: object, timeout: float) -> HttpResponse:
        from src.savetax.http import HttpResponse as _Resp

        seen.append((url, dict(headers)))  # type: ignore[arg-type]
        lid = url.rsplit("/", 1)[-1]
        if lid == "222222222_0000000001":
            return _Resp(status=403, headers={}, body=b"", elapsed_seconds=0.01)
        status, text = seen_pages.get(lid, (404, ""))
        return _Resp(status=status, headers={}, body=text.encode(), elapsed_seconds=0.01)

    from src.savetax.config import SaveTaxConfig as _Cfg
    from src.savetax.publicweb import PublicFetcher as _Fetch

    fetch = _Fetch(_Cfg(), _evidence(), transport=_transport, sleep=lambda _s: None, clock=lambda: 0.0)  # type: ignore[arg-type]
    detail = _item_detail([_fact()])
    cand = CrossCandidate("11111111", 5472000, 800000, ("222222222_0000000001", "333333333_0000000002"))
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2508000")


def test_region_notation_difference_records_caveat(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울(강남 포함)")
    pages = {
        "222222222_0000000001": (
            200,
            "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
            "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울(강남 포함) 인증 없음",
        )
    }
    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, pages)
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert any("공급 지역 표기 차이" in c for c in bundle.caveats)


@pytest.mark.parametrize("price", ["2508000", "2508000.5"])
def test_html_confirmation_is_cached_and_price_is_exact(tmp_path: Path, price: str) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2760000")
    bench = _detail("222222222_0000000001", price=price)
    html = (
        "<script>물품식별번호 fake</script><p>단위 대</p><p>물품식별번호 11111111</p>"
        "<p>계약품목관리번호 222222222_0000000001</p><p>계약 단가 <b>2,508,000</b>원 / 대</p>"
        "<p>공급 업체 V</p><p>납품요구 없음</p><p>계약 기간 P</p><p>납품 조건 현장설치도</p>"
        "<p>납품요구 기록 없음</p><p>공급 지역 서울</p><p>인증 없음</p>"
    )
    facts = [_fact(fact_id=f"F{i}", unit_price_krw=Decimal("2760000"), comparable_amount_krw=Decimal("5520000")) for i in range(2)]
    ds, raw, fetch, seen, detail, cand = _bundle_setup(
        tmp_path, {order.listing_id: order, bench.listing_id: bench}, {bench.listing_id: (200, html)}, facts=facts,
    )
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    if price == "2508000":
        assert bundle is not None
        assert len(bundle.lines) == 2
        assert bundle.expected_total_krw == 1008000
        assert bundle.rejected == ()
    else:
        assert bundle is None
    assert len(seen) == 1


def test_blank_order_conditions_never_produce_evidence(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2760000", terms=" ")
    bench = _detail("222222222_0000000001", terms="현장설치도")
    ds, raw, fetch, seen, detail, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, {})
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is None
    assert seen == []


def test_wrong_item_snapshot_is_rejected_before_requests(tmp_path: Path) -> None:
    from dataclasses import replace

    ds, raw, fetch, seen, detail, cand = _bundle_setup(tmp_path, {}, {})
    try:
        with pytest.raises(DataIntegrityError, match="identity mismatch"):
            build_cross_bundle(cand, replace(detail, item_id="wrong"), ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert raw.calls == []
    assert seen == []


@pytest.mark.parametrize("field", ["listing_id", "item_id", "contract_no"])
def test_wrong_raw_order_identity_cannot_supply_conditions(tmp_path: Path, field: str) -> None:
    from dataclasses import replace

    lid = "111111111_0000000001"
    order = replace(_detail(lid, contract_no="111111111_0", price="2760000"), **{field: "wrong"})
    bench = _detail("222222222_0000000001")
    ds, raw, fetch, seen, detail, cand = _bundle_setup(tmp_path, {lid: order, bench.listing_id: bench}, {})
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is None
    assert seen == []


def _conflict_setup(tmp_path: Path, fact_unit: str, raw_unit: str, page_text: str) -> tuple[Dataset, _FakeRaw, PublicFetcher, list, ItemDetail, CrossCandidate]:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울", unit=fact_unit)
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울", unit=raw_unit)
    pages = {"222222222_0000000001": (200, page_text)}
    ds, raw, fetch, seen, _d, cand = _bundle_setup(tmp_path, {order.listing_id: order, bench.listing_id: bench}, pages)  # type: ignore[arg-type]
    detail = _item_detail([_fact(unit=fact_unit)])
    return ds, raw, fetch, seen, detail, cand


_CONFLICT_PAGE = (
    "단위 개 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
    "계약 단가 2,508,000원 / 개 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음"
)


def test_display_conflict_is_annotated_not_dropped(tmp_path: Path) -> None:
    ds, raw, fetch, _seen, detail, cand = _conflict_setup(tmp_path, "조", "조", _CONFLICT_PAGE)  # type: ignore[misc]
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    quote = bundle.lines[0].price_evidence[0].quote
    assert "원 / 개(" in quote
    assert "원시 데이터·납품 사실 단위 조" in quote
    assert "비교 단위 조" in quote
    assert len(bundle.unit_notes) == 1


def test_matching_labels_add_no_note(tmp_path: Path) -> None:
    page = _CONFLICT_PAGE.replace("단위 개", "단위 조").replace("/ 개", "/ 조")
    ds, raw, fetch, _seen, detail, cand = _conflict_setup(tmp_path, "조", "조", page)  # type: ignore[misc]
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.unit_notes == ()
    assert "비교 단위 조" in bundle.lines[0].price_evidence[0].quote


def test_reject_policy_drops_conflicting_benchmark(tmp_path: Path) -> None:
    ds, raw, fetch, _seen, detail, cand = _conflict_setup(tmp_path, "조", "조", _CONFLICT_PAGE)  # type: ignore[misc]
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05"), unit_conflict_policy="reject"), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines == ()
    assert bundle.rejected[0].reason == "unit_display_conflict"


def test_reject_policy_falls_back_to_next_benchmark(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울", unit="조")
    cheap = _detail("222222222_0000000001", price="2400000", terms="서울", region="서울", unit="조")
    nxt = _detail("333333333_0000000002", price="2508000", terms="서울", region="서울", unit="조")
    bad = (
        "단위 개 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
        "계약 단가 2,400,000원 / 개 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음"
    )
    good = (
        "단위 조 물품식별번호 11111111 계약품목관리번호 333333333_0000000002 "
        "계약 단가 2,508,000원 / 조 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음"
    )
    ds = _bundle_dataset(tmp_path)
    raw = _FakeRaw({order.listing_id: order, cheap.listing_id: cheap, nxt.listing_id: nxt})
    seen: list = []
    fetch = _fetcher({"222222222_0000000001": (200, bad), "333333333_0000000002": (200, good)}, seen)
    detail = _item_detail([_fact(unit="조")])
    cand = CrossCandidate("11111111", 5472000, 800000, ("222222222_0000000001", "333333333_0000000002"))
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05"), unit_conflict_policy="reject"), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2508000")


def test_data_layer_mismatch_rejected_before_page_read(tmp_path: Path) -> None:
    page = _CONFLICT_PAGE.replace("단위 개", "단위 조").replace("/ 개", "/ 조")
    ds, raw, fetch, seen, detail, cand = _conflict_setup(tmp_path, "조", "개", page)  # type: ignore[misc]
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is None
    assert seen == []


def test_case_variant_is_comparable(tmp_path: Path) -> None:
    page = (
        "단위 ea 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
        "계약 단가 2,508,000원 / ea 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음"
    )
    ds, raw, fetch, _seen, _d, cand = _bundle_setup(
        tmp_path,
        {
            "111111111_0000000001": _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울", unit="EA"),
            "222222222_0000000001": _detail("222222222_0000000001", price="2508000", terms="서울", region="서울", unit="ea"),
        },
        {"222222222_0000000001": (200, page)},
    )
    detail = _item_detail([_fact(unit="EA")])
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.unit_notes == ()


def test_self_contradicting_page_is_unparseable() -> None:
    with pytest.raises(DataIntegrityError):
        parse_listing_page(
            "단위 개 물품식별번호 1 계약품목관리번호 2 계약 단가 1,000원 / 대 "
            "공급 업체 V 계약 기간 P 납품 조건 T 납품요구 기록 없음 공급 지역 S 인증 C"
        )


def test_quote_respects_length_cap_with_unit_segments_kept(tmp_path: Path) -> None:
    long_text = "V" * 5000
    page = (
        "단위 조 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
        f"계약 단가 2,508,000원 / 조 공급 업체 {long_text} 납품요구 없음 계약 기간 {long_text} 납품 조건 서울 납품요구 기록 없음 공급 지역 {long_text} 인증 없음"
    )
    ds, raw, fetch, _seen, detail, cand = _conflict_setup(tmp_path, "조", "조", page)  # type: ignore[misc]
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    quote = bundle.lines[0].price_evidence[0].quote
    assert len(quote) <= 2000
    assert "2,508,000" in quote
    assert "비교 단위" in quote
    assert _TODAY.isoformat() in quote


def test_price_fields_ignore_unit_handling(tmp_path: Path) -> None:
    ds, raw, fetch, _seen, detail, cand = _conflict_setup(tmp_path, "조", "조", _CONFLICT_PAGE)  # type: ignore[misc]
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2508000")
    assert bundle.lines[0].price_evidence[0].observed_price_krw == 2508000


def test_quote_length_fallbacks_keep_required_segments() -> None:
    from src.savetax.crosscontract import ListingPage, _quote

    base = {
        "listing_id": "1",
        "item_id": "2",
        "name": "N",
        "unit_price_krw": 1000,
        "unit": "조",
        "unit_label": "조",
        "vendor_name": "V",
        "contract_period": "P",
        "delivery_terms": "T",
        "supply_region": "S",
    }
    short = _quote(ListingPage(**base), _TODAY, "조")
    assert "비교 단위 조" in short
    drop_period = _quote(
        ListingPage(**{**base, "vendor_name": "V" * 100, "contract_period": "P" * 1900}),
        _TODAY,
        "조",
    )
    assert len(drop_period) <= 2000
    assert "비교 단위 조" in drop_period
    drop_region = _quote(
        ListingPage(
            **{
                **base,
                "vendor_name": "V" * 500,
                "contract_period": "P" * 500,
                "supply_region": "S" * 1500,
            }
        ),
        _TODAY,
        "조",
    )
    assert len(drop_region) <= 2000
    assert "비교 단위 조" in drop_region
    vendor_short = _quote(
        ListingPage(
            **{
                **base,
                "vendor_name": "V" * 1950,
                "contract_period": "P" * 100,
                "supply_region": "S" * 100,
            }
        ),
        _TODAY,
        "조",
    )
    assert len(vendor_short) <= 2000
    assert "비교 단위 조" in vendor_short
    extreme = _quote(
        ListingPage(
            **{
                **base,
                "name": "N" * 3000,
                "vendor_name": "V" * 3000,
                "contract_period": "P" * 3000,
                "supply_region": "S" * 3000,
            }
        ),
        _TODAY,
        "조",
    )
    assert len(extreme) <= 2000
    assert "비교 단위 조" in extreme
    assert _TODAY.isoformat() in extreme


def test_cached_conflict_rejects_each_fact_without_duplicate_fetches(tmp_path: Path) -> None:
    ds, raw, fetch, seen, detail, cand = _conflict_setup(tmp_path, "조", "조", _CONFLICT_PAGE)
    from dataclasses import replace

    detail = replace(detail, facts=(detail.facts[0], replace(detail.facts[0], fact_id="F2")))
    records: list[dict[str, str]] = []
    try:
        bundle = build_cross_bundle(
            cand, detail, ds, raw, fetch,
            _evidence(cross_markup_threshold=Decimal("1.05"), unit_conflict_policy="reject"),
            AuditConfig(), today=_TODAY, on_unit_conflict=records.append,
        )  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines == ()
    assert [(r.fact_id, r.reason) for r in bundle.rejected] == [
        ("F1", "unit_display_conflict"), ("F2", "unit_display_conflict"),
    ]
    assert len(records) == 1
    assert len(seen) == 1


def test_required_unit_segments_are_never_truncated(tmp_path: Path) -> None:
    unit = "U" * 2100
    page = _CONFLICT_PAGE.replace("단위 개", f"단위 {unit}").replace("/ 개", f"/ {unit}")
    ds, raw, fetch, seen, detail, cand = _conflict_setup(tmp_path, unit, unit, page)
    try:
        bundle = build_cross_bundle(
            cand, detail, ds, raw, fetch,
            _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY,
        )  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is None
    assert len(seen) == 1


def test_own_item_rebuilt_only_when_allowed(tmp_path: Path) -> None:
    from dataclasses import replace

    ds, raw, fetch, _seen, detail, cand = _bundle_setup(
        tmp_path,
        {_detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울").listing_id:
         _detail("111111111_0000000001", contract_no="111111111_0", price="2736000", terms="서울", region="서울"),
         _detail("222222222_0000000001", price="2508000", terms="서울", region="서울").listing_id:
         _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")},
        {"222222222_0000000001": (200,
         "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
         "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음")},
    )
    mine = replace(detail, state=ItemState.CLAIMED_BY_ME)
    try:
        assert build_cross_bundle(cand, mine, ds, raw, fetch,  # type: ignore[arg-type]
                                  _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(),
                                  today=_TODAY) is None
        got = build_cross_bundle(cand, mine, ds, raw, fetch,  # type: ignore[arg-type]
                                 _evidence(cross_markup_threshold=Decimal("1.05")), AuditConfig(),
                                 today=_TODAY, allowed_states=frozenset({ItemState.CLAIMED_BY_ME}))
        assert got is not None
    finally:
        ds.close()


def test_foreign_claim_never_rebuilt(tmp_path: Path) -> None:
    from dataclasses import replace

    ds, raw, fetch, _seen, detail, cand = _bundle_setup(tmp_path, {}, {})
    foreign = replace(detail, state=ItemState.CLAIMED)
    try:
        assert build_cross_bundle(cand, foreign, ds, raw, fetch,  # type: ignore[arg-type]
                                  _evidence(), AuditConfig(), today=_TODAY,
                                  allowed_states=frozenset({ItemState.CLAIMED_BY_ME})) is None
    finally:
        ds.close()


_L1G = "111111111_0000000001"
_L1S = "111111111_0000000002"
_L2G = "222222222_0000000001"
_L2S = "222222222_0000000002"

_SERIES_BLOCKS = (
    SeriesBlock("111111111_0", 1, 1, ListingSeries.GENERAL),
    SeriesBlock("111111111_0", 2, 2, ListingSeries.SEISMIC),
    SeriesBlock("222222222_0", 1, 1, ListingSeries.GENERAL),
    SeriesBlock("222222222_0", 2, 2, ListingSeries.SEISMIC),
)

_SERIES_SCOPE = "같은 물품식별번호가 한 계약 안에 제품 계열별로 따로 등재된 경우 등재 순번으로 계열을 구분하여 같은 계열의 등재끼리만 비교하였다."


def _sev(**over: object) -> EvidenceConfig:
    base: dict[str, object] = {
        "variant_policy": VariantPolicy.SERIES,
        "series_blocks": _SERIES_BLOCKS,
        "cross_markup_threshold": Decimal("1.05"),
    }
    base.update(over)
    return EvidenceConfig(**base)  # type: ignore[arg-type]


def _spage(lid: str, price: str) -> tuple[int, str]:
    return (
        200,
        f"단위 대 물품식별번호 11111111 계약품목관리번호 {lid} "
        f"계약 단가 {price}원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P "
        f"납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음",
    )


def _sdetail(lid: str, price: str, contract: str) -> ListingDetail:
    return _detail(lid, price=price, contract_no=contract, terms="서울", region="서울", contract_date=_REQUEST)


def _series_setup(
    tmp_path: Path,
    order: tuple[str, str] = ("2508000", "2736000"),
    bench: tuple[str, str] = ("2000000", "2400000"),
    bench_contract: str = "222222222_0",
    bench_ids: tuple[str, str] = (_L2G, _L2S),
) -> tuple[Dataset, _FakeRaw]:
    rows = [
        _base_row(listing_id=_L1G, price=int(order[0]),
                  dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None,
                  dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
        _base_row(listing_id=_L1S, price=int(order[1]), dlv_amount=5472000, dlv_qty=2),
        _base_row(
            listing_id=bench_ids[0], price=int(bench[0]),
            dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None,
            dlv_amount=None, dlv_first_date=None, dlv_last_date=None,
        ),
        _base_row(
            listing_id=bench_ids[1], price=int(bench[1]),
            dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None,
            dlv_amount=None, dlv_first_date=None, dlv_last_date=None,
        ),
    ]
    ds = _write_dataset(tmp_path, rows)
    raw = _FakeRaw({
        _L1G: _sdetail(_L1G, order[0], "111111111_0"),
        _L1S: _sdetail(_L1S, order[1], "111111111_0"),
        bench_ids[0]: _sdetail(bench_ids[0], bench[0], bench_contract),
        bench_ids[1]: _sdetail(bench_ids[1], bench[1], bench_contract),
    })
    return ds, raw


def _series_bundle(
    tmp_path: Path,
    facts: list,
    pages: dict,
    ds: Dataset,
    raw: _FakeRaw,
    evidence: EvidenceConfig | None = None,
) -> object:
    seen: list = []
    fetch = _fetcher(pages, seen)
    detail = _item_detail(facts)
    cand = CrossCandidate("11111111", 5472000, 672000, ())
    try:
        return build_cross_bundle(cand, detail, ds, raw, fetch, evidence or _sev(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()


def test_series_same_series_benchmark_is_used(tmp_path: Path) -> None:
    ds, raw = _series_setup(tmp_path)
    bundle = _series_bundle(tmp_path, [_fact()], {_L2S: _spage(_L2S, "2,400,000")}, ds, raw)
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2400000")


def test_series_cheaper_other_series_listing_is_never_cited(tmp_path: Path) -> None:
    ds, raw = _series_setup(tmp_path)
    bundle = _series_bundle(tmp_path, [_fact()], {_L2S: _spage(_L2S, "2,400,000")}, ds, raw)
    assert bundle is not None
    for line in bundle.lines:
        assert _L2G not in line.price_evidence[0].quote
    assert _L2G not in " ".join(bundle.caveats)
    assert _L2G not in raw.calls


def test_series_cheaper_seismic_listing_is_compared_with_seismic_facts(tmp_path: Path) -> None:
    ds, raw = _series_setup(tmp_path, bench=("2400000", "2000000"))
    bundle = _series_bundle(tmp_path, [_fact()], {_L2S: _spage(_L2S, "2,000,000")}, ds, raw)
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2000000")


def test_series_unmapped_fact_contract_blocks_the_fact(tmp_path: Path) -> None:
    c3a, c3b = "333333333_0000000001", "333333333_0000000002"
    rows = [
        _base_row(listing_id=c3a, price=2508000,
                  dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None,
                  dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
        _base_row(listing_id=c3b, price=2736000, dlv_amount=5472000, dlv_qty=2),
        _base_row(listing_id=_L2G, price=2000000, dlv_key=None, dlv_lines=None,
                  dlv_qty=None, dlv_unit=None, dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
        _base_row(listing_id=_L2S, price=2400000, dlv_key=None, dlv_lines=None,
                  dlv_qty=None, dlv_unit=None, dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
    ]
    ds = _write_dataset(tmp_path, rows)
    raw = _FakeRaw({
        c3a: _sdetail(c3a, "2508000", "333333333_0"),
        c3b: _sdetail(c3b, "2736000", "333333333_0"),
        _L2G: _sdetail(_L2G, "2000000", "222222222_0"),
        _L2S: _sdetail(_L2S, "2400000", "222222222_0"),
    })
    fact = _fact(contract_no="333333333_0", listing_id=c3b)
    bundle = _series_bundle(tmp_path, [fact], {}, ds, raw)
    assert bundle is not None
    assert bundle.lines == ()
    assert [r.reason for r in bundle.rejected if r.fact_id == "F1"] == ["variant_unresolved"]
    assert _L2G not in raw.calls
    assert _L2S not in raw.calls


def test_series_unmapped_benchmark_contract_is_skipped(tmp_path: Path) -> None:
    other = "444444444_0000000001"
    other2 = "444444444_0000000002"
    ds, raw = _series_setup(tmp_path, bench_contract="444444444_0", bench_ids=(other, other2))
    bundle = _series_bundle(tmp_path, [_fact()], {}, ds, raw)
    assert bundle is None


def test_series_discounted_fact_price_blocks_the_fact(tmp_path: Path) -> None:
    ds, raw = _series_setup(tmp_path)
    fact = _fact(unit_price_krw=Decimal("2600000"), comparable_amount_krw=Decimal("5200000"))
    bundle = _series_bundle(tmp_path, [fact], {}, ds, raw)
    assert bundle is not None
    assert bundle.lines == ()
    assert any(r.reason == "variant_unresolved" for r in bundle.rejected)
    assert bundle.title == "변형 구분 불확실로 비교 제외"


def test_series_resolution_precedes_order_condition_checks(tmp_path: Path) -> None:
    ds, raw = _series_setup(tmp_path)
    raw.mapping[_L1G] = _detail(_L1G, price="2508000", contract_no="111111111_0", terms="")
    raw.mapping[_L1S] = _detail(_L1S, price="2736000", contract_no="111111111_0", terms="")
    fact = _fact(unit_price_krw=Decimal("2600000"), comparable_amount_krw=Decimal("5200000"))
    bundle = _series_bundle(tmp_path, [fact], {}, ds, raw)
    assert bundle is not None
    assert [r.reason for r in bundle.rejected] == ["variant_unresolved"]
    assert _L2S not in raw.calls


def test_series_benchmark_later_used_as_order_reuses_cache(tmp_path: Path) -> None:
    ds, raw = _series_setup(tmp_path)
    facts = [
        _fact(),
        _fact(fact_id="F2", contract_no="222222222_0", listing_id=_L2S,
              unit_price_krw=Decimal("2400000"), comparable_amount_krw=Decimal("4800000")),
    ]
    bundle = _series_bundle(tmp_path, facts, {_L2S: _spage(_L2S, "2,400,000")}, ds, raw)
    assert bundle is not None
    assert len(bundle.lines) == 1
    assert raw.calls.count(_L2S) == 1


def test_series_non_variant_item_is_untouched(tmp_path: Path) -> None:
    rows = [
        _base_row(listing_id=_L1G, price=2736000, dlv_amount=5472000, dlv_qty=2),
        _base_row(listing_id=_L2G, price=2508000, dlv_key=None, dlv_lines=None,
                  dlv_qty=None, dlv_unit=None, dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
    ]
    pages = {_L2G: _spage(_L2G, "2,508,000")}

    def run(policy: object) -> object:
        ds = _write_dataset(tmp_path, rows)
        raw = _FakeRaw({
            _L1G: _sdetail(_L1G, "2736000", "111111111_0"),
            _L2G: _sdetail(_L2G, "2508000", "222222222_0"),
        })
        seen: list = []
        detail = _item_detail([_fact(contract_no="111111111_0", listing_id=_L1G)])
        cand = CrossCandidate("11111111", 5472000, 456000, (_L2G,))
        try:
            return build_cross_bundle(  # type: ignore[arg-type]
                cand, detail, ds, raw, _fetcher(pages, seen),
                EvidenceConfig(variant_policy=policy, series_blocks=(),  # type: ignore[arg-type]
                               cross_markup_threshold=Decimal("1.05")),
                AuditConfig(), today=_TODAY)
        finally:
            ds.close()

    series_bundle = run(VariantPolicy.SERIES)
    off_bundle = run(VariantPolicy.OFF)
    assert series_bundle is not None
    assert off_bundle is not None
    assert series_bundle.lines == off_bundle.lines
    assert series_bundle.title == off_bundle.title


def test_series_benchmark_listing_is_fetched_once(tmp_path: Path) -> None:
    ds, raw = _series_setup(tmp_path)
    facts = [_fact(fact_id=f"F{i}") for i in (1, 2, 3)]
    bundle = _series_bundle(tmp_path, facts, {_L2S: _spage(_L2S, "2,400,000")}, ds, raw)
    assert bundle is not None
    assert len(bundle.lines) == 3
    assert raw.calls.count(_L2S) == 1
    assert raw.calls.count(_L1G) == 1
    assert raw.calls.count(_L1S) == 1
    assert "원시 API 릴리스 rtest" in bundle.reproduction_notes


def test_series_other_series_listings_are_not_fetched(tmp_path: Path) -> None:
    ds, raw = _series_setup(tmp_path)
    fact = _fact(unit_price_krw=Decimal("2508000"), comparable_amount_krw=Decimal("5016000"))
    bundle = _series_bundle(tmp_path, [fact], {_L2G: _spage(_L2G, "2,000,000")}, ds, raw)
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2000000")
    assert _L2S not in raw.calls


def test_series_region_and_date_gates_still_apply(tmp_path: Path) -> None:
    ds, raw = _series_setup(tmp_path)
    raw.mapping[_L2S] = _detail(_L2S, price="2400000", contract_no="222222222_0",
                                terms="서울", region="서울", end="20250101")
    bundle = _series_bundle(tmp_path, [_fact()], {}, ds, raw)
    assert bundle is None


def test_series_scope_and_caveat_describe_the_series(tmp_path: Path) -> None:
    ds, raw = _series_setup(tmp_path)
    bundle = _series_bundle(tmp_path, [_fact()], {_L2S: _spage(_L2S, "2,400,000")}, ds, raw)
    assert bundle is not None
    assert _SERIES_SCOPE in bundle.comparison_scope
    assert f"계열 구분: 계약 222222222_0 등재 {_L2S} (내진형 계열)" in bundle.caveats
    assert not any("등재 단가 복수" in c for c in bundle.caveats)


def test_strict_highest_priced_representative_still_holds(tmp_path: Path) -> None:
    rows = [
        _base_row(listing_id=_L1G, price=5200000, dlv_amount=10400000, dlv_qty=2),
        _base_row(listing_id="222222222_0000000001", price=2100000, dlv_key=None, dlv_lines=None,
                  dlv_qty=None, dlv_unit=None, dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
        _base_row(listing_id="222222222_0000000002", price=4510000, dlv_key=None, dlv_lines=None,
                  dlv_qty=None, dlv_unit=None, dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
    ]
    ds = _write_dataset(tmp_path, rows)
    raw = _FakeRaw({
        _L1G: _sdetail(_L1G, "5200000", "111111111_0"),
        "222222222_0000000001": _sdetail("222222222_0000000001", "2100000", "222222222_0"),
        "222222222_0000000002": _sdetail("222222222_0000000002", "4510000", "222222222_0"),
    })
    fact = _fact(contract_no="111111111_0", listing_id=_L1G,
                 unit_price_krw=Decimal("5200000"), comparable_amount_krw=Decimal("10400000"))
    pages = {"222222222_0000000002": _spage("222222222_0000000002", "4,510,000")}
    seen: list = []
    detail = _item_detail([fact])
    cand = CrossCandidate("11111111", 10400000, 1000000, ("222222222_0000000001", "222222222_0000000002"))
    try:
        bundle = build_cross_bundle(  # type: ignore[arg-type]
            cand, detail, ds, raw, _fetcher(pages, seen),
            _evidence(variant_policy="strict", cross_markup_threshold=Decimal("1.05")),
            AuditConfig(), today=_TODAY)
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("4510000")


def test_series_pool_hygiene_skips_unfit_listings(tmp_path: Path) -> None:
    good = "222222222_0000000002"
    over = "333333333_0000000001"
    missing = "444444444_0000000001"
    foreign = "555555555_0000000001"
    wronged = "666666666_0000000001"
    catalog = "777777777_0000000001"
    barren = "888888888_0000000001"
    dateless = "999999999_0000000001"
    rows = [
        _base_row(listing_id=_L1G, price=2508000,
                  dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None,
                  dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
        _base_row(listing_id=_L1S, price=2736000, dlv_amount=5472000, dlv_qty=2),
        _base_row(listing_id=_L2G, price=2000000,
                  dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None,
                  dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
        _base_row(listing_id=good, price=2300000,
                  dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None,
                  dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
    ]
    for lid, price in ((over, 2700000), (missing, 2400000), (foreign, 2400000),
                       (wronged, 2400000), (catalog, 2400000), (barren, 2400000), (dateless, 2400000)):
        rows.append(
            _base_row(listing_id=lid, price=price,
                      dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None,
                      dlv_amount=None, dlv_first_date=None, dlv_last_date=None)
        )
    ds = _write_dataset(tmp_path, rows)
    blocks = (
        SeriesBlock("111111111_0", 1, 1, ListingSeries.GENERAL),
        SeriesBlock("111111111_0", 2, 2, ListingSeries.SEISMIC),
        SeriesBlock("222222222_0", 1, 1, ListingSeries.GENERAL),
        SeriesBlock("222222222_0", 2, 2, ListingSeries.SEISMIC),
        SeriesBlock("333333333_0", 1, 1, ListingSeries.SEISMIC),
        SeriesBlock("444444444_0", 1, 1, ListingSeries.SEISMIC),
        SeriesBlock("555555555_0", 1, 1, ListingSeries.SEISMIC),
        SeriesBlock("666666666_0", 1, 1, ListingSeries.SEISMIC),
        SeriesBlock("777777777_0", 1, 1, ListingSeries.SEISMIC),
        SeriesBlock("888888888_0", 1, 1, ListingSeries.SEISMIC),
        SeriesBlock("999999999_0", 1, 1, ListingSeries.SEISMIC),
    )
    raw = _FakeRaw({
        _L1G: _sdetail(_L1G, "2508000", "111111111_0"),
        _L1S: _sdetail(_L1S, "2736000", "111111111_0"),
        _L2G: _sdetail(_L2G, "2000000", "222222222_0"),
        good: _sdetail(good, "2300000", "222222222_0"),
        over: _sdetail(over, "2700000", "333333333_0"),
        foreign: _detail(foreign, item_id="99999999", price="2400000",
                         contract_no="555555555_0", terms="서울", region="서울"),
        wronged: _sdetail(wronged, "2400000", "000000000_0"),
        catalog: _detail(catalog, price="2400000", contract_no="777777777_0",
                         terms="서울", region="서울", kind="catalog_reference"),
        barren: _detail(barren, price="2400000", contract_no="888888888_0",
                        terms="", region="서울"),
        dateless: _detail(dateless, price="2400000", contract_no="999999999_0",
                          terms="서울", region="서울", begin=None, end=None),
    })
    bundle = _series_bundle(
        tmp_path, [_fact(), _fact(fact_id="F2")], {good: _spage(good, "2,300,000")}, ds, raw,
        _sev(series_blocks=blocks),
    )
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2300000")
    quote = bundle.lines[0].price_evidence[0].quote
    assert good in quote
    for bad in (_L2G, over, missing, foreign, wronged, catalog, barren, dateless):
        assert bad not in quote
    assert _L2G not in raw.calls
    assert over not in raw.calls
    assert raw.calls.count(missing) == 1


def test_series_failed_page_is_not_refetched_for_next_fact(tmp_path: Path) -> None:
    failer = "222222222_0000000002"
    good = "333333333_0000000001"
    rows = [
        _base_row(listing_id=_L1G, price=2508000,
                  dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None,
                  dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
        _base_row(listing_id=_L1S, price=2736000, dlv_amount=5472000, dlv_qty=2),
        _base_row(listing_id=_L2G, price=2000000,
                  dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None,
                  dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
        _base_row(listing_id=failer, price=2200000,
                  dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None,
                  dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
        _base_row(listing_id=good, price=2300000,
                  dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None,
                  dlv_amount=None, dlv_first_date=None, dlv_last_date=None),
    ]
    ds = _write_dataset(tmp_path, rows)
    blocks = (
        SeriesBlock("111111111_0", 1, 1, ListingSeries.GENERAL),
        SeriesBlock("111111111_0", 2, 2, ListingSeries.SEISMIC),
        SeriesBlock("222222222_0", 1, 1, ListingSeries.GENERAL),
        SeriesBlock("222222222_0", 2, 2, ListingSeries.SEISMIC),
        SeriesBlock("333333333_0", 1, 1, ListingSeries.SEISMIC),
    )
    raw = _FakeRaw({
        _L1G: _sdetail(_L1G, "2508000", "111111111_0"),
        _L1S: _sdetail(_L1S, "2736000", "111111111_0"),
        _L2G: _sdetail(_L2G, "2000000", "222222222_0"),
        failer: _sdetail(failer, "2200000", "222222222_0"),
        good: _sdetail(good, "2300000", "333333333_0"),
    })
    facts = [_fact(fact_id="F1"), _fact(fact_id="F2")]
    bundle = _series_bundle(tmp_path, facts, {good: _spage(good, "2,300,000")}, ds, raw,
                            _sev(series_blocks=blocks))
    assert bundle is not None
    assert len(bundle.lines) == 2
    assert all(line.benchmark_unit_price_krw == Decimal("2300000") for line in bundle.lines)


_TERMS_PAGE = (
    "사진 불러오는 중 제품명 단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
    "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 2025.1.1 ~ 2026.12.31 "
    "납품 조건 서울 납품요구 기록 없음 인증 없음 "
    "같은 물건의 다른 등재 333333333_0000000002 납품 단가 1,000원 "
    "계약 조건 계약번호 222222222_0 계약 방식 다수공급자계약 변경 차수 19 계약일 2025.11.14 "
    "계약 부서 구매과 공급 지역 전지역(제주 및 도서산간지역 운임비 별도) 인증 없음 업체 구분 중소기업 "
    "계약 특수조건 원문 특약 없음"
)


def test_region_is_read_from_contract_terms_block() -> None:
    page = parse_listing_page(_TERMS_PAGE)
    assert page.supply_region == "전지역(제주 및 도서산간지역 운임비 별도)"


def test_change_order_and_contract_date_are_parsed() -> None:
    page = parse_listing_page(_TERMS_PAGE)
    assert page.change_order == "19"
    assert page.contract_date == date(2025, 11, 14)


def test_pre_sibling_region_wins() -> None:
    text = _TERMS_PAGE.replace(
        "납품요구 기록 없음 인증 없음", "납품요구 기록 없음 공급 지역 서울 인증 없음"
    )
    page = parse_listing_page(text)
    assert page.supply_region == "서울"


def test_page_without_terms_block_is_unchanged() -> None:
    page = parse_listing_page(_PAGE)
    assert page.change_order is None
    assert page.contract_date is None


def test_special_terms_text_is_not_parsed() -> None:
    text = (
        "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
        "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 "
        "납품요구 기록 없음 인증 없음 "
        "같은 물건의 다른 등재 333333333_0000000002 납품 단가 1,000원 "
        "계약 조건 계약번호 1 계약 방식 A 변경 차수 2 계약일 2025.1.2 계약 부서 B 인증 C 업체 구분 D "
        "계약 특수조건 원문 공급 지역 부산"
    )
    assert parse_listing_page(text).supply_region is None


def test_terms_block_edge_branches_are_covered() -> None:
    from src.savetax.crosscontract import _parse_terms_block
    from src.savetax.crosscontract import _parse_terms_change_order
    from src.savetax.crosscontract import _parse_terms_contract_date

    assert _parse_terms_block("no block here") == {}
    assert _parse_terms_block("계약 조건 only without number") == {}
    assert _parse_terms_change_order(None) is None
    assert _parse_terms_change_order("차수 없음") is None
    assert _parse_terms_change_order("07차") == "7"
    assert _parse_terms_contract_date(None) is None
    assert _parse_terms_contract_date("soon") is None
    assert _parse_terms_contract_date("2025.13.1") is None
    text = (
        "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
        "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 "
        "납품요구 기록 없음 계약 특수조건 원문 앞 공급 지역 부산 인증 없음"
    )
    assert parse_listing_page(text).supply_region is None


def test_terms_without_siblings_preserve_primary_fields() -> None:
    base = _TERMS_PAGE.split("같은 물건의 다른 등재", 1)[0]
    terms = "계약 조건" + _TERMS_PAGE.split("계약 조건", 1)[1]
    page = parse_listing_page(base + terms)
    assert page.delivery_terms == "서울"
    assert page.vendor_name == "벤더"
    assert page.contract_period == "2025.1.1 ~ 2026.12.31"
    assert page.supply_region == "전지역(제주 및 도서산간지역 운임비 별도)"
    assert page.contract_date == date(2025, 11, 14)


def test_special_terms_cannot_create_contract_block() -> None:
    base = _TERMS_PAGE.split("같은 물건의 다른 등재", 1)[0]
    text = base + "계약 특수조건 원문 계약 조건 계약번호 X 변경 차수 99 계약일 2026.1.1 공급 지역 부산"
    page = parse_listing_page(text)
    assert page.supply_region is None
    assert page.change_order is None
    assert page.contract_date is None
