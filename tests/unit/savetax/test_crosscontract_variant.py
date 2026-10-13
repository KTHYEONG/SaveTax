"""Invariant guards for the hidden-variant gate in the cross-contract lane."""

from __future__ import annotations

from dataclasses import replace
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
    item_listing_groups,
)
from src.savetax.http import HttpResponse
from src.savetax.errors import ApiError, DataIntegrityError
from src.savetax.publicweb import PublicFetcher
from src.savetax.rawapi import ListingDetail

_TODAY = date(2026, 10, 13)
_REQUEST = date(2025, 10, 12)
_ITEM = "99999999"

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

_C1A = "111111111_0000000001"
_C1B = "111111111_0000000002"
_C2A = "222222222_0000000001"
_C2B = "222222222_0000000002"
_C3A = "333333333_0000000001"
_C1NO = "111111111_0"
_C2NO = "222222222_0"
_C3NO = "333333333_0"


def _row(lid: str, price: int, dlv_key: Any = None, amount: Any = None, qty: Any = None) -> dict[str, Any]:
    return {
        "listing_id": lid,
        "item_id": _ITEM,
        "name": "물탱크",
        "detail_category": "물탱크",
        "category_l_name": "기계",
        "manufacturer": "M",
        "model": "T",
        "unit": "대",
        "price": price,
        "contract_begin": "2025-01-01",
        "contract_end": "2026-12-31",
        "vendor_name": "업체",
        "dlv_key": dlv_key,
        "dlv_lines": 1 if dlv_key is not None else None,
        "dlv_qty": qty,
        "dlv_unit": "대" if dlv_key is not None else None,
        "dlv_amount": amount,
        "dlv_first_date": "2025-10-11" if dlv_key is not None else None,
        "dlv_last_date": "2025-10-12" if dlv_key is not None else None,
    }


def _write(rows: list[dict[str, Any]], tmp_path: Path):
    from src.savetax.dataset import Dataset

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


def _variant_rows(*, include_sole: bool = True) -> list[dict[str, Any]]:
    rows = [
        _row(_C1A, 4225000),
        _row(_C1B, 5200000),
        _row(_C2A, 2100000),
        _row(_C2B, 4510000),
        _row(_C3A, 3000000),
        _row("444444444_0000000001", 5200000, dlv_key=7, amount=10400000, qty=2),
    ]
    return [row for row in rows if include_sole or row["listing_id"] != _C3A]


def _detail(
    lid: str,
    price: str,
    contract_no: str,
    terms: str = "서울",
    region: str = "서울",
    contract_date: date | None = _REQUEST,
) -> ListingDetail:
    return ListingDetail(
        listing_id=lid,
        item_id=_ITEM,
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
        contract_date=contract_date,
    )


class _FakeRaw:
    def __init__(
        self,
        mapping: dict[str, ListingDetail | None],
        deliveries: dict[str, tuple | None] | None = None,
    ) -> None:
        self.mapping = mapping
        self.deliveries = deliveries or {}
        self.delivery_calls: list[str] = []

    def get_listing(self, listing_id: str) -> ListingDetail | None:
        return self.mapping.get(listing_id)

    def list_deliveries(self, listing_id: str):  # type: ignore[no-untyped-def]
        self.delivery_calls.append(listing_id)
        return self.deliveries.get(listing_id)


def _fetcher(pages: dict[str, tuple[int, str]]) -> PublicFetcher:
    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        lid = url.rsplit("/", 1)[-1]
        status, text = pages.get(lid, (404, ""))
        return HttpResponse(status=status, headers={}, body=text.encode(), elapsed_seconds=0.01)

    return PublicFetcher(
        SaveTaxConfig(), EvidenceConfig(), transport=transport, sleep=lambda _s: None, clock=lambda: 0.0  # type: ignore[arg-type]
    )


def _page(lid: str, price: str) -> str:
    return (
        f"단위 대 물품식별번호 {_ITEM} 계약품목관리번호 {lid} "
        f"계약 단가 {price}원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P "
        f"납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음"
    )


def _fact(price: str, contract: str = _C1NO, fact_id: str = "F1") -> Fact:
    return Fact(
        fact_id=fact_id,
        listing_id=_C1B,
        contract_no=contract,
        request_no="R1",
        line_no="1",
        request_date=_REQUEST,
        unit="대",
        tax_basis="vat_included",
        basis_id="basis:대:vat_included",
        period_kind="none",
        unit_price_krw=Decimal(price),
        quantity=Decimal("2"),
        comparable_amount_krw=Decimal(int(price) * 2),
        scorable=True,
        exclusion_reasons=(),
    )


def _item(facts: list[Fact]) -> ItemDetail:
    return ItemDetail(
        item_id=_ITEM,
        state=ItemState.AVAILABLE,
        name=None,
        model=None,
        manufacturer=None,
        category=None,
        source_release="rtest",
        facts_total=len(facts),
        facts_truncated=False,
        facts=tuple(facts),
    )


def _raw_map(**over: str | None) -> dict[str, ListingDetail | None]:
    mapping: dict[str, ListingDetail | None] = {
        _C1A: _detail(_C1A, "4225000", _C1NO),
        _C1B: _detail(_C1B, "5200000", _C1NO),
        _C2A: _detail(_C2A, "2100000", _C2NO),
        _C2B: _detail(_C2B, "4510000", _C2NO),
        _C3A: _detail(_C3A, "3000000", _C3NO),
    }
    for key, val in over.items():
        if val is None:
            mapping[key] = None
        else:
            base = mapping[key]
            assert base is not None
            mapping[key] = ListingDetail(
                listing_id=base.listing_id,
                item_id=base.item_id,
                unit=base.unit,
                price_krw=Decimal(str(val)),
                price_kind=base.price_kind,
                contract_no=base.contract_no,
                contract_type=base.contract_type,
                contract_begin=base.contract_begin,
                contract_end=base.contract_end,
                vendor_name=base.vendor_name,
                manufacturer=base.manufacturer,
                model=base.model,
                category=base.category,
                delivery_terms=base.delivery_terms,
                supply_region=base.supply_region,
                release=base.release,
                contract_date=base.contract_date,
            )
    return mapping


def _ev(**over: Any) -> EvidenceConfig:
    base: dict[str, Any] = {"cross_markup_threshold": Decimal("1.10"), "variant_policy": "strict"}
    base.update(over)
    return EvidenceConfig(**base)  # type: ignore[arg-type]


def test_groups_keyed_by_contract(tmp_path: Path) -> None:
    ds = _write(_variant_rows(), tmp_path)
    try:
        groups = item_listing_groups(ds, _ITEM, AuditConfig())
    finally:
        ds.close()
    assert set(groups) >= {_C1NO, _C2NO, _C3NO}
    assert sorted(p for _, p in groups[_C2NO].listings) == [Decimal("2100000"), Decimal("4510000")]


def test_cheaper_sibling_is_never_benchmark(tmp_path: Path) -> None:
    ds = _write(_variant_rows(), tmp_path)
    try:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2B, _C3A)),
            _item([_fact("5200000")]),
            ds,
            _FakeRaw(_raw_map()),  # type: ignore[arg-type]
            _fetcher({_C2B: (200, _page(_C2B, "4,510,000"))}),
            _ev(),
            AuditConfig(),
            today=_TODAY,
        )
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("4510000")


def test_low_fact_rejected_under_strict(tmp_path: Path) -> None:
    ds = _write(_variant_rows(), tmp_path)
    try:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2B, _C3A)),
            _item([_fact("4225000")]),
            ds,
            _FakeRaw(_raw_map()),  # type: ignore[arg-type]
            _fetcher({_C2B: (200, _page(_C2B, "4,510,000")), _C3A: (200, _page(_C3A, "3,000,000"))}),
            _ev(),
            AuditConfig(),
            today=_TODAY,
        )
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines == ()
    assert bundle.rejected[0].reason == "variant_unresolved"


def test_allow_low_fact_admits_sole_benchmark(tmp_path: Path) -> None:
    ds = _write(_variant_rows(), tmp_path)
    try:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2B, _C3A)),
            _item([_fact("4225000")]),
            ds,
            _FakeRaw(_raw_map()),  # type: ignore[arg-type]
            _fetcher({_C3A: (200, _page(_C3A, "3,000,000"))}),
            _ev(variant_policy="allow_low_fact"),
            AuditConfig(),
            today=_TODAY,
        )
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("3000000")


def test_high_fact_never_uses_sole(tmp_path: Path) -> None:
    ds = _write(_variant_rows(), tmp_path)
    try:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C3A,)),
            _item([_fact("5200000")]),
            ds,
            _FakeRaw(_raw_map()),  # type: ignore[arg-type]
            _fetcher({_C3A: (200, _page(_C3A, "3,000,000"))}),
            _ev(variant_policy="allow_low_fact"),
            AuditConfig(),
            today=_TODAY,
        )
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.rejected[0].reason == "variant_unresolved"


def test_unknown_fact_price_unresolved(tmp_path: Path) -> None:
    ds = _write(_variant_rows(), tmp_path)
    try:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C3A,)),
            _item([_fact("5000000")]),
            ds,
            _FakeRaw(_raw_map()),  # type: ignore[arg-type]
            _fetcher({_C3A: (200, _page(_C3A, "3,000,000"))}),
            _ev(variant_policy="allow_low_fact"),
            AuditConfig(),
            today=_TODAY,
        )
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.rejected[0].reason == "variant_unresolved"


def test_pricier_sibling_in_live_api_drops_benchmark(tmp_path: Path) -> None:
    ds = _write(_variant_rows(include_sole=False), tmp_path)
    try:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2B,)),
            _item([_fact("5200000")]),
            ds,
            _FakeRaw(_raw_map(**{_C2A: "4900000"})),  # type: ignore[arg-type]
            _fetcher({}),
            _ev(),
            AuditConfig(),
            today=_TODAY,
        )
    finally:
        ds.close()
    assert bundle is None


def test_unreadable_sibling_drops_benchmark(tmp_path: Path) -> None:
    ds = _write(_variant_rows(include_sole=False), tmp_path)
    mapping = _raw_map()
    mapping[_C2A] = None
    try:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2B,)),
            _item([_fact("5200000")]),
            ds,
            _FakeRaw(mapping),  # type: ignore[arg-type]
            _fetcher({}),
            _ev(),
            AuditConfig(),
            today=_TODAY,
        )
    finally:
        ds.close()
    assert bundle is None


def test_non_variant_item_has_no_caveat(tmp_path: Path) -> None:
    rows = [
        _row("111111111_0000000001", 2760000),
        _row("222222222_0000000001", 2508000),
        _row("444444444_0000000001", 2760000, dlv_key=7, amount=5520000, qty=2),
    ]
    ds = _write(rows, tmp_path)
    order = _detail("111111111_0000000001", "2760000", "111111111_0")
    bench = ListingDetail(
        listing_id="222222222_0000000001",
        item_id=_ITEM,
        unit="대",
        price_krw=Decimal("2508000"),
        price_kind="contract",
        contract_no="222222222_0",
        contract_type=None,
        contract_begin=date(2025, 1, 1),
        contract_end=date(2026, 12, 31),
        vendor_name="벤더",
        manufacturer=None,
        model=None,
        category=None,
        delivery_terms="서울",
        supply_region="서울",
        release="rtest",
        contract_date=_REQUEST,
    )
    try:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 5472000, 456000, ("222222222_0000000001",)),
            _item([_fact("2760000", contract="111111111_0")]),
            ds,
            _FakeRaw({order.listing_id: order, bench.listing_id: bench}),  # type: ignore[arg-type]
            _fetcher({"222222222_0000000001": (200, _page("222222222_0000000001", "2,508,000"))}),
            _ev(),
            AuditConfig(),
            today=_TODAY,
        )
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2508000")
    assert "가장 높은 단가" not in bundle.comparison_scope
    assert not any("등재 단가 복수" in c for c in bundle.caveats)


def test_policy_off_restores_cheaper_choice(tmp_path: Path) -> None:
    ds = _write(_variant_rows(), tmp_path)
    try:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2A,)),
            _item([_fact("5200000")]),
            ds,
            _FakeRaw(_raw_map()),  # type: ignore[arg-type]
            _fetcher({_C2A: (200, _page(_C2A, "2,100,000"))}),
            _ev(variant_policy="off"),
            AuditConfig(),
            today=_TODAY,
        )
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2100000")


def test_candidate_query_ranks_by_representative(tmp_path: Path) -> None:
    ds = _write(
        [
            _row(_C1A, 4225000),
            _row(_C1B, 5200000),
            _row(_C2A, 2100000),
            _row(_C2B, 4510000),
            _row(_C3A, 3000000),
            _row("444444444_0000000001", 5200000, dlv_key=7, amount=52000000, qty=10),
        ],
        tmp_path,
    )
    try:
        found = find_cross_candidates(ds, _ev(min_item_expected_krw=1000), AuditConfig())
    finally:
        ds.close()
    assert len(found) == 1
    assert _C2B in found[0].benchmark_listing_ids
    assert _C2A not in found[0].benchmark_listing_ids
    assert found[0].variant_bearing is True


def test_scope_and_caveat_name_rule(tmp_path: Path) -> None:
    ds = _write(_variant_rows(), tmp_path)
    try:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2B,)),
            _item([_fact("5200000")]),
            ds,
            _FakeRaw(_raw_map()),  # type: ignore[arg-type]
            _fetcher({_C2B: (200, _page(_C2B, "4,510,000"))}),
            _ev(),
            AuditConfig(),
            today=_TODAY,
        )
    finally:
        ds.close()
    assert bundle is not None
    assert "가장 높은 단가" in bundle.comparison_scope
    assert any("등재 단가 복수" in c for c in bundle.caveats)


def test_non_representative_candidate_is_dropped(tmp_path: Path) -> None:
    ds = _write(_variant_rows(include_sole=False), tmp_path)
    try:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2A,)),
            _item([_fact("5200000")]),
            ds,
            _FakeRaw(_raw_map()),  # type: ignore[arg-type]
            _fetcher({}),
            _ev(),
            AuditConfig(),
            today=_TODAY,
        )
    finally:
        ds.close()
    assert bundle is None


def test_window_still_applies_to_representative(tmp_path: Path) -> None:
    ds = _write(_variant_rows(include_sole=False), tmp_path)
    mapping = _raw_map()
    base = mapping[_C2B]
    assert base is not None
    mapping[_C2B] = ListingDetail(
        listing_id=base.listing_id,
        item_id=base.item_id,
        unit=base.unit,
        price_krw=base.price_krw,
        price_kind=base.price_kind,
        contract_no=base.contract_no,
        contract_type=base.contract_type,
        contract_begin=date(2024, 1, 1),
        contract_end=date(2024, 12, 31),
        vendor_name=base.vendor_name,
        manufacturer=base.manufacturer,
        model=base.model,
        category=base.category,
        delivery_terms=base.delivery_terms,
        supply_region=base.supply_region,
        release=base.release,
        contract_date=base.contract_date,
    )
    try:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2B,)),
            _item([_fact("5200000")]),
            ds,
            _FakeRaw(mapping),  # type: ignore[arg-type]
            _fetcher({}),
            _ev(),
            AuditConfig(),
            today=_TODAY,
        )
    finally:
        ds.close()
    assert bundle is None


@pytest.mark.parametrize("field", ["listing_id", "item_id", "contract_no"])
def test_wrong_sibling_identity_fails_closed(tmp_path: Path, field: str) -> None:
    mapping = _raw_map()
    sibling = mapping[_C2A]
    assert sibling is not None
    mapping[_C2A] = replace(sibling, **{field: "wrong"})
    with _write(_variant_rows(include_sole=False), tmp_path) as ds:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2B,)), _item([_fact("5200000")]),
            ds, _FakeRaw(mapping), _fetcher({_C2B: (200, _page(_C2B, "4,510,000"))}),
            _ev(), AuditConfig(), today=_TODAY,
        )  # type: ignore[arg-type]
    assert bundle is None


@pytest.mark.parametrize("error", [ApiError("raw", 403, "denied", "denied", False), DataIntegrityError("[DATA] malformed")])
def test_sibling_read_error_drops_only_that_benchmark(tmp_path: Path, error: Exception) -> None:
    class BrokenRaw(_FakeRaw):
        def get_listing(self, listing_id: str) -> ListingDetail | None:
            if listing_id == _C2A:
                raise error
            return super().get_listing(listing_id)

    with _write(_variant_rows(), tmp_path) as ds:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2B, _C3A)), _item([_fact("4225000")]),
            ds, BrokenRaw(_raw_map()), _fetcher({_C3A: (200, _page(_C3A, "3,000,000"))}),
            _ev(variant_policy="allow_low_fact"), AuditConfig(), today=_TODAY,
        )  # type: ignore[arg-type]
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("3000000")


@pytest.mark.parametrize("policy", ["strict", "allow_low_fact"])
def test_absent_benchmark_group_cannot_bypass_variant_gate(tmp_path: Path, policy: str) -> None:
    unknown = "555555555_0000000001"
    mapping = _raw_map()
    mapping[unknown] = _detail(unknown, "2100000", "555555555_0")
    with _write(_variant_rows(include_sole=False), tmp_path) as ds:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (unknown,)), _item([_fact("5200000")]),
            ds, _FakeRaw(mapping), _fetcher({unknown: (200, _page(unknown, "2,100,000"))}),
            _ev(variant_policy=policy), AuditConfig(), today=_TODAY,
        )  # type: ignore[arg-type]
    assert bundle is None


def test_representative_raw_contract_must_match_dataset(tmp_path: Path) -> None:
    mapping = _raw_map()
    for lid in (_C2A, _C2B):
        listing = mapping[lid]
        assert listing is not None
        mapping[lid] = replace(listing, contract_no="wrong")
    with _write(_variant_rows(include_sole=False), tmp_path) as ds:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2B,)), _item([_fact("5200000")]),
            ds, _FakeRaw(mapping), _fetcher({_C2B: (200, _page(_C2B, "4,510,000"))}),
            _ev(), AuditConfig(), today=_TODAY,
        )  # type: ignore[arg-type]
    assert bundle is None


def test_fact_group_below_minimum_is_rejected(tmp_path: Path) -> None:
    rows = [_row(_C1A, 1), _row(_C2A, 2100000), _row(_C2B, 4510000)]
    mapping = _raw_map()
    mapping[_C1A] = _detail(_C1A, "5200000", _C1NO)
    with _write(rows, tmp_path) as ds:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2B,)), _item([_fact("5200000")]),
            ds, _FakeRaw(mapping), _fetcher({_C2B: (200, _page(_C2B, "4,510,000"))}),
            _ev(), AuditConfig(), today=_TODAY,
        )  # type: ignore[arg-type]
    assert bundle is None


def test_page_failure_is_not_misreported_as_variant_only(tmp_path: Path) -> None:
    results: list[tuple[bool, int]] = []
    with _write(_variant_rows(), tmp_path) as ds:
        bundle = build_cross_bundle(
            CrossCandidate(_ITEM, 10400000, 1000000, (_C2B, _C3A)), _item([_fact("5200000")]),
            ds, _FakeRaw(_raw_map()), _fetcher({}), _ev(), AuditConfig(), today=_TODAY,
            on_variant_result=lambda bearing, count: results.append((bearing, count)),
        )  # type: ignore[arg-type]
    assert bundle is None
    assert results[-1] == (True, 0)
