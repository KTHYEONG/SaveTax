"""Proof that a benchmark price held on a request date."""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from src.savetax.errors import DomainValidationError
from src.savetax.rawapi import DeliveryRow, ListingDetail
from src.savetax.units import reconcile_units

__all__ = ["PriceProof", "PriceProofKind", "price_in_effect", "proof_note"]


class PriceProofKind(enum.StrEnum):
    """How the benchmark price is known to hold."""

    AMENDED_BEFORE = "amended_before"
    ORDER_BEFORE = "order_before"


@dataclass(frozen=True)
class PriceProof:
    """Why a contract unit price is known to hold on a request date."""

    kind: PriceProofKind
    contract_date: date | None
    change_order: str | None
    order: DeliveryRow | None


def price_in_effect(
    benchmark: ListingDetail,
    request_date: date,
    load_rows: Callable[[], Sequence[DeliveryRow] | None],
) -> PriceProof | None:
    """Return the proof that the benchmark's current price applied on the request date, or None."""
    if benchmark.contract_date is not None and benchmark.contract_date <= request_date:
        return PriceProof(
            kind=PriceProofKind.AMENDED_BEFORE,
            contract_date=benchmark.contract_date,
            change_order=benchmark.change_order,
            order=None,
        )
    rows = load_rows()
    if not rows:
        return None
    best: DeliveryRow | None = None
    for row in rows:
        if row.contract_unit_price_krw != benchmark.price_krw:
            continue
        if row.quantity <= 0:
            continue
        if row.request_date > request_date:
            continue
        if (
            benchmark.unit is not None
            and row.unit is not None
            and not reconcile_units(benchmark.unit, row.unit, None).comparable
        ):
            continue
        if (
            best is None
            or row.request_date > best.request_date
            or (
                row.request_date == best.request_date
                and (row.request_no, row.line_no) < (best.request_no, best.line_no)
            )
        ):
            best = row
    if best is None:
        return None
    return PriceProof(
        kind=PriceProofKind.ORDER_BEFORE,
        contract_date=benchmark.contract_date,
        change_order=benchmark.change_order,
        order=best,
    )


def _format_price(value: Decimal) -> str:
    if isinstance(value, Decimal) and value == int(value):
        return f"{int(value):,}"
    return f"{value:,}"


def proof_note(proof: PriceProof, benchmark: ListingDetail, request_date: date) -> str:
    """Return the Korean memo (at most 500 chars) describing the proof."""
    if proof.kind is PriceProofKind.AMENDED_BEFORE:
        if proof.contract_date is None:
            raise DomainValidationError("[DATA] invalid proof: missing contract date")
        base = (
            f"비교 단가 유효 근거: 비교 계약의 최종 변경계약일 {proof.contract_date.isoformat()}"
        )
        if proof.change_order is not None:
            base += f"(변경 차수 {proof.change_order})"
        base += f"이 납품요구일 {request_date.isoformat()} 이전"
        if len(base) > 500:
            raise DomainValidationError("[DATA] invalid note: must be within 500 chars")
        return base
    order = proof.order
    if order is None:
        raise DomainValidationError("[DATA] invalid proof: missing order")
    agency = (order.agency_name or "")[:60]
    note = (
        f"비교 단가 유효 근거: 비교 계약에서 같은 계약단가 {_format_price(benchmark.price_krw)}원의 "
        f"납품요구가 {order.request_date.isoformat()}에 있었음({agency}, {order.request_no}), "
        f"납품요구일 {request_date.isoformat()} 이전"
    )
    if len(note) > 500:
        raise DomainValidationError("[DATA] invalid note: must be within 500 chars")
    return note
