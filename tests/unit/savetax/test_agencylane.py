"""Invariant guards for the agency evidence lane."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

import pytest

from src.savetax.agencylane import (
    AgencyCandidate,
    agency_benchmark,
    build_agency_bundle,
    find_agency_candidates,
)
from src.savetax.competition import Fact, ItemDetail, ItemState
from src.savetax.config import AuditConfig, EvidenceConfig, SaveTaxConfig
from src.savetax.dataset import Dataset
from src.savetax.errors import ApiError, DataIntegrityError
from src.savetax.publicweb import PublicPage
from src.savetax.rawapi import DeliveryRow

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

_LID = "111111111_0000000001"
_CONTRACT = "111111111_0"
_ITEM = "99999999"


def _write_dataset(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
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
    return data_dir


def _base_row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "listing_id": _LID,
        "item_id": _ITEM,
        "name": "패널",
        "detail_category": "프린터",
        "category_l_name": "전자/정보/통신/영상",
        "manufacturer": "Epson",
        "model": "EB-PU2216B",
        "unit": "대",
        "price": 100000,
        "contract_begin": "2025-01-01",
        "contract_end": "2026-12-31",
        "vendor_name": "주식회사 트리엠",
        "dlv_key": 7,
        "dlv_lines": 6,
        "dlv_qty": 400,
        "dlv_unit": "대",
        "dlv_amount": 39000000,
        "dlv_first_date": "2025-10-11",
        "dlv_last_date": "2025-10-12",
    }
    row.update(over)
    return row


def _dataset(tmp_path: Path, rows: list[dict[str, Any]]) -> Dataset:
    data_dir = _write_dataset(tmp_path, rows)
    return Dataset(SaveTaxConfig(dataset_dir=data_dir), AuditConfig())


def _row(**over: Any) -> DeliveryRow:
    base: dict[str, Any] = {
        "request_no": "R25TA000001",
        "line_no": "1",
        "request_date": date(2025, 10, 12),
        "agency_name": "A시청",
        "unit": "대",
        "quantity": Decimal("10"),
        "unit_price_krw": Decimal("100"),
        "contract_unit_price_krw": Decimal("100"),
        "amount_krw": Decimal("1000"),
    }
    base.update(over)
    return DeliveryRow(**base)


def _scaled_row(**over: Any) -> DeliveryRow:
    base: dict[str, Any] = {
        "request_no": "R25TA000001",
        "line_no": "1",
        "request_date": date(2025, 10, 12),
        "agency_name": "A시청",
        "unit": "대",
        "quantity": Decimal("10"),
        "unit_price_krw": Decimal("100000"),
        "contract_unit_price_krw": Decimal("100000"),
        "amount_krw": Decimal("1000000"),
    }
    base.update(over)
    return DeliveryRow(**base)


def _peers() -> list[DeliveryRow]:
    return [
        _row(
            request_no="R25TA000002", agency_name="B구청", unit_price_krw=Decimal("60"),
            quantity=Decimal("5"), amount_krw=Decimal("300"), request_date=date(2025, 9, 12),
        ),
        _row(
            request_no="R25TA000003", agency_name="C시청", unit_price_krw=Decimal("70"),
            quantity=Decimal("8"), amount_krw=Decimal("560"), request_date=date(2025, 11, 21),
        ),
    ]


def test_cheaper_smaller_orders_on_both_sides_give_a_benchmark() -> None:
    target = _row()
    rows = [target, *_peers()]
    benchmark = agency_benchmark(target, rows, EvidenceConfig())
    assert benchmark is not None
    assert benchmark.price_krw == Decimal("70")
    assert benchmark.anchor == rows[2]
    assert benchmark.supporter_agencies == 2


def test_benchmark_is_not_the_lowest_price() -> None:
    target = _row()
    rows = [
        target,
        _row(request_no="R25TA000002", agency_name="B구청", unit_price_krw=Decimal("50"),
             quantity=Decimal("5"), amount_krw=Decimal("250"), request_date=date(2025, 9, 12)),
        _row(request_no="R25TA000003", agency_name="C시청", unit_price_krw=Decimal("60"),
             quantity=Decimal("5"), amount_krw=Decimal("300"), request_date=date(2025, 9, 20)),
        _row(request_no="R25TA000004", agency_name="D구청", unit_price_krw=Decimal("70"),
             quantity=Decimal("8"), amount_krw=Decimal("560"), request_date=date(2025, 11, 21)),
    ]
    benchmark = agency_benchmark(target, rows, EvidenceConfig())
    assert benchmark is not None
    assert benchmark.price_krw == Decimal("70")


def test_larger_order_is_not_a_peer() -> None:
    target = _row()
    rows = [
        target,
        _row(request_no="R25TA000002", agency_name="B구청", unit_price_krw=Decimal("60"),
             quantity=Decimal("20"), amount_krw=Decimal("1200"), request_date=date(2025, 9, 12)),
    ]
    assert agency_benchmark(target, rows, EvidenceConfig()) is None


def test_one_sided_evidence_is_rejected() -> None:
    target = _row()
    rows = [
        target,
        _row(request_no="R25TA000002", agency_name="B구청", unit_price_krw=Decimal("60"),
             quantity=Decimal("5"), amount_krw=Decimal("300"), request_date=date(2025, 9, 12)),
    ]
    assert agency_benchmark(target, rows, EvidenceConfig()) is None


def test_single_supporting_agency_is_rejected() -> None:
    target = _row()
    rows = [
        target,
        _row(request_no="R25TA000002", agency_name="B구청", unit_price_krw=Decimal("60"),
             quantity=Decimal("5"), amount_krw=Decimal("300"), request_date=date(2025, 9, 12)),
        _row(request_no="R25TA000003", agency_name="B구청", unit_price_krw=Decimal("70"),
             quantity=Decimal("5"), amount_krw=Decimal("350"), request_date=date(2025, 11, 21)),
    ]
    assert agency_benchmark(target, rows, EvidenceConfig()) is None


def test_different_contract_unit_price_is_not_compared() -> None:
    target = _row()
    rows = [
        target,
        _row(request_no="R25TA000002", agency_name="B구청", unit_price_krw=Decimal("60"),
             quantity=Decimal("5"), amount_krw=Decimal("300"), request_date=date(2025, 9, 12),
             contract_unit_price_krw=Decimal("200")),
        _row(request_no="R25TA000003", agency_name="C시청", unit_price_krw=Decimal("70"),
             quantity=Decimal("8"), amount_krw=Decimal("560"), request_date=date(2025, 11, 21),
             contract_unit_price_krw=Decimal("200")),
    ]
    assert agency_benchmark(target, rows, EvidenceConfig()) is None


def test_deep_discount_below_the_floor_is_ignored() -> None:
    target = _row()
    rows = [
        target,
        *_peers(),
        _row(request_no="R25TA000004", agency_name="D구청", unit_price_krw=Decimal("20"),
             quantity=Decimal("5"), amount_krw=Decimal("100"), request_date=date(2025, 9, 15)),
    ]
    benchmark = agency_benchmark(target, rows, EvidenceConfig())
    assert benchmark is not None
    assert benchmark.price_krw == Decimal("70")
    assert benchmark.supporter_agencies == 2


def test_markup_below_the_threshold_is_rejected() -> None:
    target = _row(unit_price_krw=Decimal("80"), amount_krw=Decimal("800"))
    assert agency_benchmark(target, [target, *_peers()], EvidenceConfig()) is None


def _mass_rows(near_day: date) -> tuple[DeliveryRow, list[DeliveryRow]]:
    target = _row()
    rows = [target]
    for index, agency in enumerate(["B구청", "C시청", "D구청", "E기관", "F기관"]):
        rows.append(
            _row(request_no=f"R25TA0001{index:02d}", agency_name=agency, unit_price_krw=Decimal("60"),
                 quantity=Decimal("5"), amount_krw=Decimal("300"),
                 request_date=date(2025, 9, 12) if index else near_day)
        )
    rows.append(
        _row(request_no="R25TA000099", agency_name="G시청", unit_price_krw=Decimal("70"),
             quantity=Decimal("5"), amount_krw=Decimal("350"), request_date=date(2025, 10, 22))
    )
    return target, rows


def test_common_discount_price_in_another_period_is_rejected() -> None:
    target, rows = _mass_rows(date(2025, 9, 8))
    assert agency_benchmark(target, rows, EvidenceConfig()) is None


def test_common_discount_price_on_nearly_the_same_day_is_accepted() -> None:
    target, rows = _mass_rows(date(2025, 10, 10))
    benchmark = agency_benchmark(target, rows, EvidenceConfig())
    assert benchmark is not None
    assert benchmark.price_krw == Decimal("70")


def test_island_agency_target_is_rejected() -> None:
    target = _row(agency_name="제주특별자치도청")
    assert agency_benchmark(target, [target, *_peers()], EvidenceConfig()) is None


def test_row_whose_amount_is_not_price_times_quantity_is_ignored() -> None:
    target = _row()
    rows = [
        target,
        *_peers(),
        _row(request_no="R25TA000004", agency_name="D구청", unit_price_krw=Decimal("65"),
             quantity=Decimal("5"), amount_krw=Decimal("999"), request_date=date(2025, 9, 15)),
    ]
    benchmark = agency_benchmark(target, rows, EvidenceConfig())
    assert benchmark is not None
    assert benchmark.supporter_agencies == 2


def test_candidates_are_ordered_by_discount_mass(tmp_path: Path) -> None:
    rows = [
        _base_row(item_id="AAAAAAAA", listing_id="111111111_0000000001", dlv_key=1,
                  price=1000000, dlv_qty=40, dlv_amount=39000000, dlv_lines=6),
        _base_row(item_id="BBBBBBBB", listing_id="222222222_0000000001", dlv_key=2,
                  price=2000000, dlv_qty=40, dlv_amount=78000000, dlv_lines=6),
        _base_row(item_id="CCCCCCCC", listing_id="333333333_0000000001", dlv_key=3,
                  price=5000000, dlv_qty=40, dlv_amount=195000000, dlv_lines=6),
        _base_row(item_id="DDDDDDDD", listing_id="444444444_0000000001", dlv_key=4,
                  price=9000000, dlv_qty=40, dlv_amount=300000000, dlv_lines=6,
                  detail_category="분전반"),
    ]
    with _dataset(tmp_path, rows) as dataset:
        found = find_agency_candidates(dataset, EvidenceConfig(), AuditConfig())
    assert [c.item_id for c in found] == ["CCCCCCCC", "BBBBBBBB", "AAAAAAAA"]
    assert [c.discount_mass_krw for c in found] == [5000000, 2000000, 1000000]
    assert found[0].keys == (("333333333_0", "333333333_0000000001"),)


def test_requested_item_bypasses_thresholds(tmp_path: Path) -> None:
    rows = [_base_row(listing_id="555555555_0000000001", dlv_key=9, price=1000000,
                      dlv_qty=40, dlv_amount=39500000, dlv_lines=2)]
    with _dataset(tmp_path, rows) as dataset:
        assert find_agency_candidates(dataset, EvidenceConfig(), AuditConfig()) == []
        found = find_agency_candidates(dataset, EvidenceConfig(), AuditConfig(), items=["99999999"])
    assert [c.item_id for c in found] == ["99999999"]
    assert found[0].keys == (("555555555_0", "555555555_0000000001"),)


class _Raw:
    def __init__(self, payloads: dict[str, Any]) -> None:
        self._payloads = payloads
        self.calls: list[str] = []

    def list_deliveries(self, listing_id: str) -> Any:
        self.calls.append(listing_id)
        payload = self._payloads[listing_id]
        if isinstance(payload, Exception):
            raise payload
        return payload


class _Fetcher:
    def __init__(self, pages: dict[str, tuple[int, str]]) -> None:
        self._pages = pages
        self.calls: list[str] = []

    def get(self, url: str) -> PublicPage:
        self.calls.append(url)
        status, text = self._pages[url]
        return PublicPage(url=url, status=status, text=text, fetched_at=datetime(2026, 10, 13, tzinfo=UTC))


def _fact(**over: Any) -> Fact:
    base: dict[str, Any] = {
        "fact_id": "F1",
        "listing_id": _LID,
        "contract_no": _CONTRACT,
        "request_no": "R25TA000001",
        "line_no": "1",
        "request_date": date(2025, 10, 12),
        "unit": "대",
        "tax_basis": "vat_included",
        "basis_id": "basis:대:vat_included",
        "period_kind": "none",
        "unit_price_krw": Decimal("100000"),
        "quantity": Decimal("10"),
        "comparable_amount_krw": Decimal("1000000"),
        "scorable": True,
        "exclusion_reasons": (),
    }
    base.update(over)
    return Fact(**base)


def _detail(facts: list[Fact], name: str | None = "패널") -> ItemDetail:
    return ItemDetail(item_id=_ITEM, state=ItemState.AVAILABLE, name=name, model="M",
                      manufacturer="M", category="C", source_release="r",
                      facts_total=len(facts), facts_truncated=False, facts=tuple(facts))


def _scaled_peers() -> list[DeliveryRow]:
    return [
        _scaled_row(request_no="R25TA000002", agency_name="B구청", unit_price_krw=Decimal("60000"),
                    quantity=Decimal("5"), amount_krw=Decimal("300000"), request_date=date(2025, 9, 12)),
        _scaled_row(request_no="R25TA000003", agency_name="C시청", unit_price_krw=Decimal("70000"),
                    quantity=Decimal("8"), amount_krw=Decimal("560000"), request_date=date(2025, 11, 21)),
    ]


def _table_row(agency: str, day: str, qty: str, price: str, amount: str, req: str) -> str:
    return (
        f"<tr><td>{agency}</td><td>{day}</td><td>{qty}</td>"
        f"<td>{price}</td><td>{amount}</td><td>{req}</td></tr>"
    )


def _listing_head(item: str, lid: str, price: str) -> str:
    return (
        f"단위 대 물품식별번호 {item} 계약품목관리번호 {lid} "
        f"계약 단가 {price}원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P "
        "납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음"
    )


def _candidate() -> AgencyCandidate:
    return AgencyCandidate(item_id=_ITEM, keys=((_CONTRACT, _LID),), order_amount_krw=39000000,
                           discount_mass_krw=1000000)


def _anchor_row_html() -> str:
    return _table_row("C시청", "2025.11.21", "8 대", "70,000원", "560,000원", "R25TA000003")


def test_bundle_cites_the_anchor_row_from_the_public_page(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    pages = {
        base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
                    f"<table>{_table_row('A시청', '2025.10.12', '10 대', '100,000원', '1,000,000원', 'R25TA000001')}</table>"
                    "1 / 2쪽</body></html>"),
        base + "?dp=2": (200, f"<html><body><table>{_anchor_row_html()}</table>2 / 2쪽</body></html>"),
    }
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher(pages), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                     today=date(2026, 10, 13))
    assert bundle is not None
    assert len(bundle.lines) == 1
    assert bundle.lane.value == "agency"
    (evidence,) = bundle.lines[0].price_evidence
    assert "C시청 2025.11.21 8 대 70,000원 560,000원 R25TA000003" in evidence.quote
    assert evidence.observed_price_krw == 70000
    assert evidence.source_url.endswith("?dp=2")
    assert bundle.title == "같은 계약의 다른 기관 납품요구 단가 대비 1.43배"
    assert bundle.unit_notes == ()
    assert bundle.caveats == ("계약 111111111_0: 비교 납품요구 1건 인용",)


def test_fact_that_disagrees_with_its_row_is_rejected(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    pages = {base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
                             f"<table>{_anchor_row_html()}</table></body></html>")}
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    bad = _fact(fact_id="F2", request_no="R25TA000002", unit_price_krw=Decimal("61000"),
                quantity=Decimal("5"), comparable_amount_krw=Decimal("305000"))
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact(), bad]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher(pages), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                     today=date(2026, 10, 13))
    assert bundle is not None
    assert len(bundle.lines) == 1
    assert {item.fact_id: item.reason for item in bundle.rejected} == {"F2": "row_fact_mismatch"}


def test_anchor_missing_from_the_public_page_is_rejected(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    pages = {base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
                             f"<table>{_anchor_row_html()}</table></body></html>")}
    zoning = [
        _scaled_row(request_no="R25TA000004", agency_name="E기관", unit_price_krw=Decimal("150000"),
                    quantity=Decimal("6"), amount_krw=Decimal("900000"), request_date=date(2025, 10, 5)),
        _scaled_row(request_no="R25TA000005", agency_name="F기관", unit_price_krw=Decimal("80000"),
                    quantity=Decimal("5"), amount_krw=Decimal("400000"), request_date=date(2025, 10, 25)),
    ]
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers(), *zoning)})
    second = _fact(fact_id="F2", request_no="R25TA000004", unit_price_krw=Decimal("150000"),
                   quantity=Decimal("6"), comparable_amount_krw=Decimal("900000"),
                   request_date=date(2025, 10, 5))
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact(), second]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher(pages), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                     today=date(2026, 10, 13))
        assert bundle is not None
        assert len(bundle.lines) == 1
        assert {item.fact_id: item.reason for item in bundle.rejected} == {"F2": "page_row_missing"}
        lone = build_agency_bundle(
            AgencyCandidate(item_id=_ITEM, keys=((_CONTRACT, _LID),), order_amount_krw=1, discount_mass_krw=1),
            _detail([second]), dataset, raw, _Fetcher(pages),  # type: ignore[arg-type]
            EvidenceConfig(), AuditConfig(), today=date(2026, 10, 13))
    assert lone is None


def test_row_fetch_failure_rejects_the_contract_facts(tmp_path: Path) -> None:
    other_lid = "222222222_0000000001"
    other_contract = "222222222_0"
    candidate = AgencyCandidate(item_id=_ITEM, keys=((_CONTRACT, _LID), (other_contract, other_lid)),
                                order_amount_krw=2, discount_mass_krw=2)
    second = _fact(fact_id="F2", listing_id=other_lid, contract_no=other_contract,
                   request_no="R25TA000009")
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers()),
                other_lid: ApiError("raw", 500, "BACKEND", "m", True)})
    base = "https://savetax.now/listing/111111111_0000000001"
    pages = {base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
                             f"<table>{_anchor_row_html()}</table></body></html>")}
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(candidate, _detail([_fact(), second]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher(pages), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                     today=date(2026, 10, 13))
    assert bundle is not None
    assert len(bundle.lines) == 1
    assert {item.fact_id: item.reason for item in bundle.rejected} == {"F2": "rows_unavailable"}
    assert raw.calls == [_LID, other_lid]


def test_range_sized_goods_need_review(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    pages = {base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
                             f"<table>{_anchor_row_html()}</table></body></html>")}
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()], name="panel 600 1200~4880"),
                                     dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher(pages), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                     today=date(2026, 10, 13))
    assert bundle is not None
    assert bundle.review_required is True


def test_text_limits_hold(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    pages = {base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
                             f"<table>{_anchor_row_html()}</table></body></html>")}
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher(pages), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                     today=date(2026, 10, 13))
    assert bundle is not None
    assert len(bundle.claim) <= 2000
    assert len(bundle.comparison_scope) <= 1000
    assert len(bundle.reproduction_notes) <= 4000
    for line in bundle.lines:
        for cited in line.price_evidence:
            assert cited.note is not None
            assert len(cited.note) <= 500


def test_unavailable_state_returns_none(tmp_path: Path) -> None:
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    detail = ItemDetail(item_id=_ITEM, state=ItemState.CLAIMED, name="패널", model="M",
                        manufacturer="M", category="C", source_release="r",
                        facts_total=1, facts_truncated=False, facts=(_fact(),))
    with _dataset(tmp_path, [_base_row()]) as dataset:
        assert build_agency_bundle(_candidate(), detail, dataset, raw,  # type: ignore[arg-type]
                                   _Fetcher({}), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                   today=date(2026, 10, 13)) is None


def test_detail_identity_mismatch_raises(tmp_path: Path) -> None:
    raw = _Raw({})
    detail = ItemDetail(item_id="OTHER", state=ItemState.AVAILABLE, name="n", model="m",
                        manufacturer="m", category="c", source_release="r",
                        facts_total=0, facts_truncated=False, facts=())
    with _dataset(tmp_path, [_base_row()]) as dataset, pytest.raises(DataIntegrityError):
            build_agency_bundle(_candidate(), detail, dataset, raw,  # type: ignore[arg-type]
                                _Fetcher({}), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                today=date(2026, 10, 13))


def test_to_decimal_coercions() -> None:
    from src.savetax.agencylane import _to_decimal

    assert _to_decimal(Decimal("1.5")) == Decimal("1.5")
    assert _to_decimal(7) == Decimal("7")
    assert _to_decimal(7.5) == Decimal("7.5")
    assert _to_decimal("70") == Decimal("70")
    assert _to_decimal(True) is None
    assert _to_decimal(None) is None
    assert _to_decimal("abc") is None
    assert _to_decimal(object()) is None


def test_format_decimal_strips_fractional_zeros() -> None:
    from src.savetax.agencylane import _format_decimal

    assert _format_decimal(Decimal("10")) == "10"
    assert _format_decimal(Decimal("10.50")) == "10.5"


def test_empty_items_returns_no_candidates(tmp_path: Path) -> None:
    with _dataset(tmp_path, [_base_row()]) as dataset:
        assert find_agency_candidates(dataset, EvidenceConfig(), AuditConfig(), items=[]) == []


def test_null_aggregates_are_skipped(tmp_path: Path) -> None:
    rows = [
        _base_row(listing_id="666666666_0000000001", dlv_key=11, dlv_lines=None),
        _base_row(item_id="AAAAAAAA", listing_id="111111111_0000000001", dlv_key=1,
                  price=1000000, dlv_qty=40, dlv_amount=39000000, dlv_lines=6),
    ]
    with _dataset(tmp_path, rows) as dataset:
        found = find_agency_candidates(dataset, EvidenceConfig(), AuditConfig())
    assert [c.item_id for c in found] == ["AAAAAAAA"]


def test_unknown_requested_item_returns_nothing(tmp_path: Path) -> None:
    with _dataset(tmp_path, [_base_row()]) as dataset:
        assert find_agency_candidates(dataset, EvidenceConfig(), AuditConfig(), items=["NOPE"]) == []


def test_invalid_target_is_rejected() -> None:
    target = _row(quantity=Decimal("0"), amount_krw=Decimal("0"))
    assert agency_benchmark(target, [target, *_peers()], EvidenceConfig()) is None


def test_blank_agency_peer_is_ignored() -> None:
    target = _row()
    rows = [
        target,
        *_peers(),
        _row(request_no="R25TA000004", agency_name="   ", unit_price_krw=Decimal("50"),
             quantity=Decimal("5"), amount_krw=Decimal("250"), request_date=date(2025, 9, 15)),
    ]
    benchmark = agency_benchmark(target, rows, EvidenceConfig())
    assert benchmark is not None
    assert benchmark.supporter_agencies == 2


def test_mixed_facts_collect_stable_reasons(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    pages = {base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
                             f"<table>{_anchor_row_html()}</table></body></html>")}
    extra = _scaled_row(request_no="R25TA000006", agency_name="G기관", unit_price_krw=Decimal("65000"),
                        quantity=Decimal("5"), amount_krw=Decimal("325000"), request_date=date(2025, 9, 12))
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers(), extra)})
    facts = [
        _fact(),
        _fact(fact_id="F2", scorable=False),
        _fact(fact_id="F3", request_date=None),
        _fact(fact_id="F4", request_no="R25TA009999"),
        _fact(fact_id="F5", request_no="R25TA000002", unit="g", unit_price_krw=Decimal("60000"),
              quantity=Decimal("5"), comparable_amount_krw=Decimal("300000"),
              request_date=date(2025, 9, 12)),
        _fact(fact_id="F6", request_no="R25TA000006", unit_price_krw=Decimal("65000"),
              quantity=Decimal("5"), comparable_amount_krw=Decimal("325000"),
              request_date=date(2025, 9, 12)),
        _fact(fact_id="F7", comparable_amount_krw=Decimal("700000")),
        _fact(fact_id="F8", contract_no="000000000_0", listing_id="000000000_0000000001"),
    ]
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail(facts), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher(pages), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                     today=date(2026, 10, 13))
    assert bundle is not None
    assert [line.fact_id for line in bundle.lines] == ["F1"]
    assert {item.fact_id: item.reason for item in bundle.rejected} == {
        "F2": "not_scorable",
        "F3": "no_request_date",
        "F4": "row_not_found",
        "F5": "unit_mismatch",
        "F6": "no_agency_benchmark",
        "F7": "no_agency_benchmark",
        "F8": "no_agency_benchmark",
    }


def test_fractional_benchmark_is_rejected(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    pages = {base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
                             f"<table>{_anchor_row_html()}</table></body></html>")}
    rows = (
        _scaled_row(),
        _scaled_row(request_no="R25TA000002", agency_name="B구청", unit_price_krw=Decimal("60000"),
                    quantity=Decimal("5"), amount_krw=Decimal("300000"), request_date=date(2025, 9, 12)),
        _scaled_row(request_no="R25TA000003", agency_name="C시청", unit_price_krw=Decimal("70000.5"),
                    quantity=Decimal("8"), amount_krw=Decimal("560004"), request_date=date(2025, 11, 21)),
    )
    raw = _Raw({_LID: rows})
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher(pages), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                     today=date(2026, 10, 13))
    assert bundle is None


def test_unreadable_first_page_is_rejected(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher({base: (403, "")}), EvidenceConfig(),  # type: ignore[arg-type]
                                     AuditConfig(), today=date(2026, 10, 13))
    assert bundle is None


def test_unparseable_first_page_is_rejected(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher({base: (200, "<html><body>hello</body></html>")}),  # type: ignore[arg-type]
                                     EvidenceConfig(), AuditConfig(), today=date(2026, 10, 13))
    assert bundle is None


def test_wrong_identity_page_is_rejected(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    other = _listing_head("00000000", "222222222_0000000001", "100,000")
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher({base: (200, f"<html><body>{other}</body></html>")}),  # type: ignore[arg-type]
                                     EvidenceConfig(), AuditConfig(), today=date(2026, 10, 13))
    assert bundle is None


def test_malformed_delivery_table_is_rejected(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    bad = _table_row("C시청", "2025.13.45", "8 대", "70,000원", "560,000원", "R25TA000003")
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher({base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "  # type: ignore[arg-type]
                                                              f"<table>{bad}</table></body></html>")}),
                                     EvidenceConfig(), AuditConfig(), today=date(2026, 10, 13))
    assert bundle is None


def test_unreadable_later_page_is_rejected(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    first = (f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
             f"<table>{_table_row('A시청', '2025.10.12', '10 대', '100,000원', '1,000,000원', 'R25TA000001')}</table>"
             "1 / 2쪽</body></html>")
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher({base: (200, first), base + "?dp=2": (500, "")}),  # type: ignore[arg-type]
                                     EvidenceConfig(), AuditConfig(), today=date(2026, 10, 13))
    assert bundle is None


def test_malformed_later_page_is_rejected(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    first = (f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
             f"<table>{_table_row('A시청', '2025.10.12', '10 대', '100,000원', '1,000,000원', 'R25TA000001')}</table>"
             "1 / 2쪽</body></html>")
    bad = _table_row("C시청", "2025.13.45", "8 대", "70,000원", "560,000원", "R25TA000003")
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher({base: (200, first), base + "?dp=2": (200, f"<html><body><table>{bad}</table></body></html>")}),  # type: ignore[arg-type]
                                     EvidenceConfig(), AuditConfig(), today=date(2026, 10, 13))
    assert bundle is None


def test_total_below_minimum_returns_none(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    pages = {base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
                             f"<table>{_anchor_row_html()}</table></body></html>")}
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher(pages), EvidenceConfig(min_item_expected_krw=300001),  # type: ignore[arg-type]
                                     AuditConfig(), today=date(2026, 10, 13))
    assert bundle is None


def test_long_identifiers_truncate_the_note(tmp_path: Path) -> None:
    long_req = "R25TA" + "1" * 400
    base = "https://savetax.now/listing/111111111_0000000001"
    anchor = _table_row("C시청", "2025.11.21", "8 대", "70,000원", "560,000원", long_req)
    pages = {base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
                             f"<table>{anchor}</table></body></html>")}
    rows = (
        _scaled_row(request_no=long_req),
        _scaled_row(request_no="R25TA000002", agency_name="B구청", unit_price_krw=Decimal("60000"),
                    quantity=Decimal("5"), amount_krw=Decimal("300000"), request_date=date(2025, 9, 12)),
        _scaled_row(request_no=long_req, line_no="2", agency_name="C시청", unit_price_krw=Decimal("70000"),
                    quantity=Decimal("8"), amount_krw=Decimal("560000"), request_date=date(2025, 11, 21)),
    )
    raw = _Raw({_LID: rows})
    fact = _fact(request_no=long_req)
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([fact]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher(pages), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                     today=date(2026, 10, 13))
    assert bundle is not None
    (evidence,) = bundle.lines[0].price_evidence
    assert evidence.note is not None
    assert len(evidence.note) <= 500


def test_key_without_facts_is_never_fetched(tmp_path: Path) -> None:
    candidate = AgencyCandidate(item_id=_ITEM, keys=((_CONTRACT, _LID), ("999999999_9", "999999999_0000000001")),
                                order_amount_krw=1, discount_mass_krw=1)
    base = "https://savetax.now/listing/111111111_0000000001"
    pages = {base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
                             f"<table>{_anchor_row_html()}</table></body></html>")}
    raw = _Raw({_LID: (_scaled_row(), *_scaled_peers())})
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(candidate, _detail([_fact()]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher(pages), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                     today=date(2026, 10, 13))
    assert bundle is not None
    assert raw.calls == [_LID]


def test_none_rows_reject_the_contract(tmp_path: Path) -> None:
    raw = _Raw({_LID: None})
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()]), dataset, raw,  # type: ignore[arg-type]
                                     _Fetcher({}), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                     today=date(2026, 10, 13))
    assert bundle is None


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity")])
def test_nonfinite_delivery_values_are_ignored(value: Decimal) -> None:
    target = _row(unit_price_krw=value)
    assert agency_benchmark(target, _peers(), EvidenceConfig()) is None
    bad_peer = _row(request_no="bad", amount_krw=value)
    benchmark = agency_benchmark(_row(), [*_peers(), bad_peer], EvidenceConfig())
    assert benchmark is not None
    assert benchmark.supporter_agencies == 2


def test_row_amount_equality_is_exact_beyond_default_precision() -> None:
    from decimal import localcontext

    from src.savetax.agencylane import _is_valid_row

    price = Decimal("12345678901234567890123456789")
    quantity = Decimal("2")
    exact_amount = Decimal("24691357802469135780246913578")
    row = _row(unit_price_krw=price, quantity=quantity, amount_krw=exact_amount)
    with localcontext() as context:
        context.prec = 6
        assert _is_valid_row(row)
        rounded_amount = price * quantity
        assert not _is_valid_row(_row(unit_price_krw=price, quantity=quantity, amount_krw=rounded_amount))


def test_any_excluded_category_blocks_the_shared_order_key(tmp_path: Path) -> None:
    rows = [
        _base_row(detail_category="가구"),
        _base_row(listing_id="111111111_0000000002", detail_category="분전반"),
    ]
    with _dataset(tmp_path, rows) as dataset:
        assert find_agency_candidates(dataset, EvidenceConfig(), AuditConfig()) == []
        assert find_agency_candidates(dataset, EvidenceConfig(), AuditConfig(), items=[_ITEM]) == []


def test_multiple_anchors_reuse_every_scanned_page(tmp_path: Path) -> None:
    from dataclasses import replace

    from src.savetax.agencylane import _cite_anchor

    target = _scaled_row()
    peers = _scaled_peers()
    benchmark = agency_benchmark(target, [target, *peers], EvidenceConfig())
    assert benchmark is not None
    base = f"https://savetax.now/listing/{_LID}"
    pages = {
        base: (200, f"<html>{_listing_head(_ITEM, _LID, '100,000')} 1 / 3쪽</html>"),
        base + "?dp=2": (200, "<html>2 / 3쪽</html>"),
        base + "?dp=3": (200, f"<table>{_anchor_row_html()}</table>3 / 3쪽"),
    }
    fetcher = _Fetcher(pages)
    cache: dict = {}
    citation = _cite_anchor(_ITEM, _LID, benchmark, date(2026, 10, 13), EvidenceConfig(), fetcher, cache)  # type: ignore[arg-type]
    assert isinstance(citation, tuple)
    absent = replace(benchmark, anchor=peers[0])
    for _ in range(2):
        assert _cite_anchor(_ITEM, _LID, absent, date(2026, 10, 13), EvidenceConfig(), fetcher, cache) is None  # type: ignore[arg-type]
    assert fetcher.calls == [base, base + "?dp=2", base + "?dp=3"]


@pytest.mark.parametrize("first_status", [200, 403])
def test_unreadable_pages_are_cached(first_status: int) -> None:
    from src.savetax.agencylane import _cite_anchor

    target = _scaled_row()
    benchmark = agency_benchmark(target, [target, *_scaled_peers()], EvidenceConfig())
    assert benchmark is not None
    base = f"https://savetax.now/listing/{_LID}"
    fetcher = _Fetcher({
        base: (first_status, f"<html>{_listing_head(_ITEM, _LID, '100,000')} 1 / 2쪽</html>"),
        base + "?dp=2": (403, ""),
    })
    cache: dict = {}
    for _ in range(2):
        assert _cite_anchor(_ITEM, _LID, benchmark, date(2026, 10, 13), EvidenceConfig(), fetcher, cache) == "page_unreadable"  # type: ignore[arg-type]
    assert fetcher.calls == ([base, base + "?dp=2"] if first_status == 200 else [base])


def test_note_prints_scaled_prices_without_fraction_digits(tmp_path: Path) -> None:
    base = "https://savetax.now/listing/111111111_0000000001"
    anchor = _table_row("C시청", "2025.11.21", "8 대", "70,000원", "560,000원", "R25TA000003")
    pages = {base: (200, f"<html><body>{_listing_head(_ITEM, _LID, '100,000')} "
                             f"<table>{anchor}</table></body></html>")}
    scaled = Decimal("100000.0000")
    rows = (
        _scaled_row(unit_price_krw=scaled, contract_unit_price_krw=scaled),
        _scaled_row(request_no="R25TA000002", agency_name="B구청", unit_price_krw=Decimal("60000.0000"),
                    contract_unit_price_krw=scaled, quantity=Decimal("5"), amount_krw=Decimal("300000"),
                    request_date=date(2025, 9, 12)),
        _scaled_row(request_no="R25TA000003", agency_name="C시청", unit_price_krw=Decimal("70000.0000"),
                    contract_unit_price_krw=scaled, quantity=Decimal("8"), amount_krw=Decimal("560000"),
                    request_date=date(2025, 11, 21)),
    )
    with _dataset(tmp_path, [_base_row()]) as dataset:
        bundle = build_agency_bundle(_candidate(), _detail([_fact()]), dataset, _Raw({_LID: rows}),  # type: ignore[arg-type]
                                     _Fetcher(pages), EvidenceConfig(), AuditConfig(),  # type: ignore[arg-type]
                                     today=date(2026, 10, 13))
    assert bundle is not None
    (evidence,) = bundle.lines[0].price_evidence
    assert evidence.note is not None
    assert "계약단가 100,000원" in evidence.note
    assert "60,000원" in evidence.note
    assert "70,000원" in evidence.note
    assert ".0" not in evidence.note
