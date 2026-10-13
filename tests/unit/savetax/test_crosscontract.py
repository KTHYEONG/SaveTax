"""Invariant guards for the cross-contract evidence lane."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pytest

from src.savetax.competition import Fact, ItemDetail, ItemState
from src.savetax.config import AuditConfig, EvidenceConfig, SaveTaxConfig
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
    )


class _FakeRaw:
    def __init__(self, mapping: dict[str, ListingDetail | None]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def get_listing(self, listing_id: str) -> ListingDetail | None:
        self.calls.append(listing_id)
        return self.mapping.get(listing_id)


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

    assert run(11_000_000) == 1
    assert run(10_999_990) == 0


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
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
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
