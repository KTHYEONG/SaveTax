"""Proof-gated cross-contract lane guards."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from tests.unit.savetax.test_crosscontract import (
    _TODAY,
    _base_row,
    _bundle_dataset,
    _bundle_setup,
    _detail,
    _evidence,
    _fact,
    _FakeRaw,
    _fetcher,
    _item_detail,
    _series_setup,
    _sev,
    _spage,
    _L2S,
    _L2G,
    _write_dataset,
)

from src.savetax.config import AuditConfig
from src.savetax.crosscontract import CrossCandidate, build_cross_bundle
from src.savetax.errors import ApiError


def _proof_page(lid: str, price: str = "2,508,000") -> tuple[int, str]:
    return (
        200,
        f"단위 대 물품식별번호 11111111 계약품목관리번호 {lid} "
        f"계약 단가 {price}원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P "
        "납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음",
    )


def _proof_row(**over: Any) -> Any:
    from src.savetax.rawapi import DeliveryRow

    base: dict[str, Any] = {
        "request_no": "R9",
        "line_no": "1",
        "request_date": date(2025, 10, 1),
        "agency_name": "기관",
        "unit": "대",
        "quantity": Decimal("2"),
        "unit_price_krw": Decimal("2508000"),
        "contract_unit_price_krw": Decimal("2508000"),
        "amount_krw": Decimal("5016000"),
    }
    base.update(over)
    return DeliveryRow(**base)


def _high_fact(**over: Any) -> Any:
    base: dict[str, Any] = {
        "unit_price_krw": Decimal("3100000"),
        "comparable_amount_krw": Decimal("6200000"),
    }
    base.update(over)
    return _fact(**base)


def _setup(
    tmp_path: Path, order: Any, bench: Any, pages: Any, facts: Any | None = None
) -> Any:
    raw_map = {order.listing_id: order, bench.listing_id: bench}
    return _bundle_setup(tmp_path, raw_map, pages, facts=facts or [_high_fact()])


def test_benchmark_amended_before_request_is_admitted(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="3100000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울",
                    contract_date=date(2025, 10, 1))
    raw = _FakeRaw({order.listing_id: order, bench.listing_id: bench})
    ds, _, fetch, _s, detail, cand = _setup(
        tmp_path, order, bench, {"222222222_0000000001": _proof_page("222222222_0000000001")})
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert (bundle.lines[0].price_evidence[0].note or "").startswith("비교 단가 유효 근거")
    assert raw.delivery_calls == []


def test_benchmark_amended_after_request_is_rejected_without_orders(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="3100000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울",
                    contract_date=date(2025, 10, 13))
    raw = _FakeRaw({order.listing_id: order, bench.listing_id: bench})
    ds, _, fetch, _s, detail, cand = _setup(tmp_path, order, bench, {})
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines == ()
    assert bundle.rejected[0].reason == "benchmark_price_unproven"


def test_earlier_order_rescues_later_amendment(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="3100000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울",
                    contract_date=date(2025, 10, 13))
    raw = _FakeRaw({order.listing_id: order, bench.listing_id: bench},
                   {"222222222_0000000001": (_proof_row(),)})
    ds, _, fetch, _s, detail, cand = _setup(
        tmp_path, order, bench, {"222222222_0000000001": _proof_page("222222222_0000000001")})
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert "R9" in (bundle.lines[0].price_evidence[0].note or "")


def test_proven_dearer_replaces_unproven_cheaper(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="3100000", terms="서울", region="서울")
    cheap = _detail("222222222_0000000001", price="2000000", terms="서울", region="서울",
                    contract_date=date(2025, 10, 13))
    dear = _detail("333333333_0000000002", price="2508000", terms="서울", region="서울",
                   contract_date=date(2025, 10, 1))
    raw = _FakeRaw({order.listing_id: order, cheap.listing_id: cheap, dear.listing_id: dear})
    ds = _bundle_dataset(tmp_path)
    seen: list = []
    fetch = _fetcher({"222222222_0000000001": _proof_page("222222222_0000000001", "2,000,000"),
                      "333333333_0000000002": _proof_page("333333333_0000000002")}, seen)
    detail = _item_detail([_high_fact()])
    cand = CrossCandidate("11111111", 6200000, 2000000, ("222222222_0000000001", "333333333_0000000002"))
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2508000")


def test_delivery_rows_fetched_once_per_contract(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="3100000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울",
                    contract_date=date(2025, 10, 13))
    raw = _FakeRaw({order.listing_id: order, bench.listing_id: bench},
                   {"222222222_0000000001": (_proof_row(),)})
    facts = [_high_fact(fact_id=f"F{i}") for i in (1, 2, 3)]
    ds, _, fetch, _s, _d, cand = _bundle_setup(
        tmp_path, {order.listing_id: order, bench.listing_id: bench},
        {"222222222_0000000001": _proof_page("222222222_0000000001")}, facts=facts)
    detail = _item_detail(facts)
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert len(bundle.lines) == 3
    assert raw.delivery_calls == ["222222222_0000000001"]


def test_row_fetch_failure_is_not_proof(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="3100000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울",
                    contract_date=date(2025, 10, 13))

    class _FailRaw(_FakeRaw):
        def list_deliveries(self, listing_id: str):  # type: ignore[no-untyped-def]
            raise ApiError("raw", 500, "X", "boom", True)

    raw = _FailRaw({order.listing_id: order, bench.listing_id: bench})
    ds, _, fetch, _s, detail, cand = _setup(tmp_path, order, bench, {})
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.rejected[0].reason == "benchmark_price_unproven"


def test_gate_can_be_switched_off(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="3100000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울",
                    contract_date=date(2025, 10, 13))
    raw = _FakeRaw({order.listing_id: order, bench.listing_id: bench})
    ds, _, fetch, _s, detail, cand = _setup(
        tmp_path, order, bench, {"222222222_0000000001": _proof_page("222222222_0000000001")})
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(require_price_proof=False), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert len(bundle.lines) == 1
    assert "비교 단가가 납품요구일에 유효" not in bundle.comparison_scope


def test_threshold_is_1_2_by_default(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="2877000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    raw = _FakeRaw({order.listing_id: order, bench.listing_id: bench})
    fact = _fact(unit_price_krw=Decimal("2877000"), comparable_amount_krw=Decimal("5754000"))
    ds = _write_dataset(tmp_path, [_base_row(listing_id="111111111_0000000001", price=2877000, dlv_amount=5754000, dlv_qty=2)])
    fetch = _fetcher({}, [])
    detail = _item_detail([fact])
    cand = CrossCandidate("11111111", 5754000, 500000, ("222222222_0000000001",))
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is None


def test_quote_carries_region_change_order_and_contract_date(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="3100000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울",
                    contract_date=date(2025, 11, 14))
    page_text = (
        "사진 불러오는 중 제품명 단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
        "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 2025.1.1 ~ 2026.12.31 "
        "납품 조건 서울 납품요구 기록 없음 인증 없음 "
        "계약 조건 계약번호 222222222_0 계약 방식 다수공급자계약 변경 차수 19 계약일 2025.11.14 "
        "계약 부서 구매과 공급 지역 서울 인증 없음 업체 구분 중소기업 "
        "계약 특수조건 원문 특약 없음"
    )
    raw = _FakeRaw({order.listing_id: order, bench.listing_id: bench})
    ds, _, fetch, _s, detail, cand = _setup(tmp_path, order, bench, {"222222222_0000000001": (200, page_text)})
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(require_price_proof=False, cross_markup_threshold=Decimal("1.05")), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    quote = bundle.lines[0].price_evidence[0].quote
    assert "공급 지역" in quote
    assert "변경 차수 19" in quote
    assert "계약일 2025.11.14" in quote


def test_page_and_raw_contract_dates_must_agree(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="3100000", terms="서울", region="서울")
    bad = _detail("222222222_0000000001", price="2000000", terms="서울", region="서울",
                  contract_date=date(2025, 10, 1))
    good = _detail("333333333_0000000002", price="2508000", terms="서울", region="서울",
                   contract_date=date(2025, 10, 1))
    page_text = (
        "단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
        "계약 단가 2,000,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P 납품 조건 서울 "
        "납품요구 기록 없음 인증 없음 "
        "계약 조건 계약번호 222222222_0 계약 방식 A 변경 차수 1 계약일 2025.11.14 계약 부서 B 공급 지역 서울 인증 C 업체 구분 D "
        "계약 특수조건 원문 없음"
    )
    raw = _FakeRaw({order.listing_id: order, bad.listing_id: bad, good.listing_id: good})
    ds = _bundle_dataset(tmp_path)
    seen: list = []
    fetch = _fetcher({"222222222_0000000001": (200, page_text),
                      "333333333_0000000002": _proof_page("333333333_0000000002")}, seen)
    detail = _item_detail([_high_fact()])
    cand = CrossCandidate("11111111", 6200000, 2000000, ("222222222_0000000001", "333333333_0000000002"))
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal("2508000")

    order2 = _detail("111111111_0000000001", contract_no="111111111_0", price="3100000", terms="서울", region="서울")
    solo = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울",
                   contract_date=date(2025, 10, 1))
    raw2 = _FakeRaw({order2.listing_id: order2, solo.listing_id: solo})
    facts = [_high_fact(fact_id="EARLY", request_date=date(2025, 9, 1)), _high_fact()]
    ds2, _, fetch2, _s, detail2, cand2 = _setup(
        tmp_path, order2, solo,
        {"222222222_0000000001": (200, page_text.replace("2,000,000", "2,508,000"))},
        facts=facts,
    )
    try:
        excluded = build_cross_bundle(cand2, detail2, ds2, raw2, fetch2, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds2.close()
    assert excluded is not None
    assert excluded.lines == ()
    assert [(r.fact_id, r.reason) for r in excluded.rejected] == [
        ("EARLY", "benchmark_price_unproven"), ("F1", "page_contract_date_mismatch"),
    ]


def test_text_names_organizer_type(tmp_path: Path) -> None:
    order = _detail("111111111_0000000001", contract_no="111111111_0", price="3100000", terms="서울", region="서울")
    bench = _detail("222222222_0000000001", price="2508000", terms="서울", region="서울")
    raw = _FakeRaw({order.listing_id: order, bench.listing_id: bench})
    ds, _, fetch, _s, detail, cand = _setup(
        tmp_path, order, bench, {"222222222_0000000001": _proof_page("222222222_0000000001")})
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _evidence(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert "「같은 물건, 다른 가격」" in bundle.claim
    assert "같은 물품의 두 계약이 동시에 다른 값일 때만 「같은 물건, 다른 가격」으로 따로 적습니다" in bundle.reproduction_notes


def test_text_limits_hold(tmp_path: Path) -> None:
    ds, raw = _series_setup(tmp_path)
    fetch = _fetcher({_L2S: _spage(_L2S, "2,400,000")}, [])
    detail = _item_detail([_fact()])
    cand = CrossCandidate("11111111", 5472000, 672000, ())
    try:
        bundle = build_cross_bundle(cand, detail, ds, raw, fetch, _sev(), AuditConfig(), today=_TODAY)  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert "등재 순번으로 계열을 구분" in bundle.comparison_scope
    assert len(bundle.comparison_scope) <= 1000
    assert len(bundle.reproduction_notes) <= 4000


def test_series_proof_is_evaluated_per_fact_despite_cached_page(tmp_path: Path) -> None:
    from dataclasses import replace

    ds, raw = _series_setup(tmp_path)
    bench = raw.mapping[_L2S]
    assert bench is not None
    raw.mapping[_L2S] = replace(bench, contract_date=date(2025, 10, 13))
    raw.deliveries[_L2S] = (_proof_row(contract_unit_price_krw=Decimal("2400000")),)
    facts = [_fact(), _fact(fact_id="EARLY", request_date=date(2025, 9, 1)), _fact(fact_id="LATE")]
    fetch = _fetcher({_L2S: _spage(_L2S, "2,400,000")}, [])
    try:
        bundle = build_cross_bundle(
            CrossCandidate("11111111", 5472000, 672000, ()), _item_detail(facts),
            ds, raw, fetch, _sev(), AuditConfig(), today=_TODAY,
        )  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert [line.fact_id for line in bundle.lines] == ["F1", "LATE"]
    assert [(r.fact_id, r.reason) for r in bundle.rejected] == [("EARLY", "benchmark_price_unproven")]
    assert raw.delivery_calls == [_L2S]


def test_series_siblings_share_rows_but_prove_each_price(tmp_path: Path) -> None:
    from dataclasses import replace

    ds, raw = _series_setup(tmp_path)
    for lid in (_L2G, _L2S):
        bench = raw.mapping[lid]
        assert bench is not None
        raw.mapping[lid] = replace(bench, contract_date=date(2025, 10, 13))
    raw.deliveries[_L2G] = (
        _proof_row(request_no="GENERAL", contract_unit_price_krw=Decimal("2000000")),
        _proof_row(request_no="SEISMIC", contract_unit_price_krw=Decimal("2400000")),
    )
    facts = [
        _fact(fact_id="GENERAL", unit_price_krw=Decimal("2508000"), comparable_amount_krw=Decimal("5016000")),
        _fact(fact_id="SEISMIC"),
    ]
    fetch = _fetcher({_L2G: _spage(_L2G, "2,000,000"), _L2S: _spage(_L2S, "2,400,000")}, [])
    try:
        bundle = build_cross_bundle(
            CrossCandidate("11111111", 5472000, 672000, ()), _item_detail(facts),
            ds, raw, fetch, _sev(), AuditConfig(), today=_TODAY,
        )  # type: ignore[arg-type]
    finally:
        ds.close()
    assert bundle is not None
    assert [line.benchmark_unit_price_krw for line in bundle.lines] == [Decimal("2000000"), Decimal("2400000")]
    assert "GENERAL" in (bundle.lines[0].price_evidence[0].note or "")
    assert "SEISMIC" in (bundle.lines[1].price_evidence[0].note or "")
    assert raw.delivery_calls == [_L2G]
