"""Invariant guards for benchmark price proofs."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from src.savetax.priceproof import PriceProofKind, price_in_effect, proof_note
from src.savetax.rawapi import DeliveryRow, ListingDetail

_REQUEST = date(2025, 10, 12)


def _bench(**over) -> ListingDetail:
    base: dict = {
        "listing_id": "222222222_0000000001",
        "item_id": "11111111",
        "unit": "대",
        "price_krw": Decimal("2508000"),
        "price_kind": "contract",
        "contract_no": "222222222_0",
        "contract_type": None,
        "contract_begin": date(2025, 1, 1),
        "contract_end": date(2026, 12, 31),
        "vendor_name": "벤더",
        "manufacturer": None,
        "model": None,
        "category": None,
        "delivery_terms": "서울",
        "supply_region": "서울",
        "release": "r",
        "contract_date": date(2025, 10, 1),
        "change_order": "19",
    }
    base.update(over)
    return ListingDetail(**base)


def _row(**over) -> DeliveryRow:
    base: dict = {
        "request_no": "R1",
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


def test_amendment_on_request_date_is_proof() -> None:
    bench = _bench(contract_date=_REQUEST)
    calls: list = []

    def load():
        calls.append(1)
        return None

    proof = price_in_effect(bench, _REQUEST, load)
    assert proof is not None
    assert proof.kind is PriceProofKind.AMENDED_BEFORE
    assert calls == []


def test_later_amendment_needs_order() -> None:
    bench = _bench(contract_date=date(2025, 10, 13))
    assert price_in_effect(bench, _REQUEST, lambda: None) is None
    assert price_in_effect(bench, _REQUEST, lambda: []) is None


def test_earlier_order_at_contract_price_is_proof() -> None:
    bench = _bench(contract_date=date(2025, 10, 13))
    row = _row()
    proof = price_in_effect(bench, _REQUEST, lambda: [row])
    assert proof is not None
    assert proof.kind is PriceProofKind.ORDER_BEFORE
    assert proof.order == row


def test_latest_earlier_order_is_chosen() -> None:
    bench = _bench(contract_date=date(2025, 10, 13))
    early = _row(request_no="R1", request_date=date(2025, 9, 1))
    late = _row(request_no="R2", request_date=date(2025, 10, 1))
    proof = price_in_effect(bench, _REQUEST, lambda: [early, late])
    assert proof is not None
    assert proof.order == late


def test_order_after_request_date_is_not_proof() -> None:
    bench = _bench(contract_date=date(2025, 10, 13))
    row = _row(request_date=date(2025, 10, 13))
    assert price_in_effect(bench, _REQUEST, lambda: [row]) is None


def test_discounted_order_is_not_proof() -> None:
    bench = _bench(contract_date=date(2025, 10, 13))
    row = _row(unit_price_krw=Decimal("2508000"), contract_unit_price_krw=Decimal("2000000"))
    assert price_in_effect(bench, _REQUEST, lambda: [row]) is None


def test_cancelled_order_is_not_proof() -> None:
    bench = _bench(contract_date=date(2025, 10, 13))
    row = _row(quantity=Decimal("0"))
    assert price_in_effect(bench, _REQUEST, lambda: [row]) is None


def test_missing_contract_date_falls_through_to_orders() -> None:
    bench = _bench(contract_date=None)
    row = _row()
    proof = price_in_effect(bench, _REQUEST, lambda: [row])
    assert proof is not None
    assert proof.kind is PriceProofKind.ORDER_BEFORE


def test_memo_names_amendment() -> None:
    bench = _bench(contract_date=date(2025, 10, 1), change_order="19")
    proof = price_in_effect(bench, _REQUEST, lambda: (_ for _ in ()).throw(AssertionError()))
    assert proof is not None
    memo = proof_note(proof, bench, _REQUEST)
    assert "2025-10-01" in memo
    assert "2025-10-12" in memo
    assert "19" in memo


def test_memo_names_order() -> None:
    bench = _bench(contract_date=date(2025, 10, 13))
    row = _row(request_no="R9", request_date=date(2025, 10, 1))
    proof = price_in_effect(bench, _REQUEST, lambda: [row])
    assert proof is not None
    memo = proof_note(proof, bench, _REQUEST)
    assert "R9" in memo
    assert "2,508,000" in memo
    assert "2025-10-01" in memo
    assert "2025-10-12" in memo


def test_unit_mismatch_row_is_skipped() -> None:
    bench = _bench(contract_date=date(2025, 10, 13))
    bad = _row(unit="개")
    assert price_in_effect(bench, _REQUEST, lambda: [bad]) is None
    good = _row(unit="대")
    proof = price_in_effect(bench, _REQUEST, lambda: [bad, good])
    assert proof is not None
    assert proof.order == good


def test_fractional_price_is_formatted() -> None:
    bench = _bench(price_krw=Decimal("2508000.5"), contract_date=date(2025, 10, 13))
    row = _row(contract_unit_price_krw=Decimal("2508000.5"), unit_price_krw=Decimal("2508000.5"))
    proof = price_in_effect(bench, _REQUEST, lambda: [row])
    assert proof is not None
    assert "2,508,000.5" in proof_note(proof, bench, _REQUEST)


def test_proof_note_rejects_bad_proofs() -> None:
    from src.savetax.errors import DomainValidationError
    from src.savetax.priceproof import PriceProof

    bench = _bench()
    with pytest.raises(DomainValidationError):
        proof_note(
            PriceProof(kind=PriceProofKind.AMENDED_BEFORE, contract_date=None, change_order=None, order=None),
            bench,
            _REQUEST,
        )
    with pytest.raises(DomainValidationError):
        proof_note(
            PriceProof(kind=PriceProofKind.ORDER_BEFORE, contract_date=None, change_order=None, order=None),
            bench,
            _REQUEST,
        )


def test_proof_note_rejects_overlong_memo() -> None:
    from src.savetax.errors import DomainValidationError
    from src.savetax.priceproof import PriceProof

    bench = _bench()
    with pytest.raises(DomainValidationError):
        proof_note(
            PriceProof(kind=PriceProofKind.AMENDED_BEFORE, contract_date=date(2025, 10, 1),
                       change_order="9" * 500, order=None),
            bench,
            _REQUEST,
        )
    row = _row(request_no="R" * 500)
    with pytest.raises(DomainValidationError):
        proof_note(
            PriceProof(kind=PriceProofKind.ORDER_BEFORE, contract_date=None, change_order=None, order=row),
            bench,
            _REQUEST,
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_same_day_orders_use_smallest_request_and_line(reverse: bool) -> None:
    rows = [_row(request_no="B", line_no="1"), _row(request_no="A", line_no="2"), _row(request_no="A", line_no="1")]
    chosen = rows[-1]
    proof = price_in_effect(_bench(contract_date=None), _REQUEST, lambda: rows[::-1] if reverse else rows)
    assert proof is not None
    assert proof.order == chosen


def test_missing_date_without_orders_is_not_proof() -> None:
    calls: list[int] = []

    def load() -> None:
        calls.append(1)

    assert price_in_effect(_bench(contract_date=None), _REQUEST, load) is None
    assert calls == [1]


def test_order_on_request_date_and_missing_unit_are_admissible() -> None:
    row = _row(request_date=_REQUEST, unit=None, agency_name="가" * 80)
    bench = _bench(contract_date=None)
    proof = price_in_effect(bench, _REQUEST, lambda: [row])
    assert proof is not None
    note = proof_note(proof, bench, _REQUEST)
    assert "가" * 60 in note
    assert "가" * 61 not in note
