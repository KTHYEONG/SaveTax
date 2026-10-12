"""Procurement audit arithmetic and competition scoring."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from decimal import ROUND_FLOOR, Decimal, localcontext

from src.savetax.config import AuditConfig
from src.savetax.errors import DomainValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "amount_lower_bound_krw",
    "expected_amount_krw",
    "is_anomaly",
    "markup_ratio",
    "price_delta",
    "tax_waste_amount",
]


def _require_decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, Decimal) or isinstance(value, bool):
        raise DomainValidationError(f"[DATA] invalid {name}: expected Decimal")
    if not value.is_finite():
        raise DomainValidationError(f"[DATA] invalid {name}: non-finite amount")
    return value


def _require_positive_price(value: Decimal, name: str) -> Decimal:
    if value <= 0:
        raise DomainValidationError(f"[DATA] invalid {name}: must be > 0")
    return value


def _require_benchmark(value: object) -> Decimal:
    checked = _require_decimal(value, "benchmark_unit_price_krw")
    if checked <= 0:
        raise DomainValidationError("[DATA] invalid benchmark_unit_price_krw: must be > 0")
    exponent = checked.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -6:
        raise DomainValidationError("[DATA] invalid benchmark_unit_price_krw: more than 6 decimals")
    return checked


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise DomainValidationError(f"[DATA] invalid {name}: expected int")
    return value


def _arithmetic_precision(*values: Decimal) -> int:
    return 2 + sum(len(value.as_tuple().digits) + abs(int(value.as_tuple().exponent)) for value in values)


def price_delta(procurement_unit_price: Decimal, market_reference_price: Decimal) -> Decimal:
    """Return procurement minus market unit price in KRW."""
    procurement = _require_positive_price(
        _require_decimal(procurement_unit_price, "procurement_unit_price"), "procurement_unit_price"
    )
    market = _require_positive_price(
        _require_decimal(market_reference_price, "market_reference_price"), "market_reference_price"
    )
    with localcontext() as context:
        context.prec = _arithmetic_precision(procurement, market)
        return procurement - market


def markup_ratio(procurement_unit_price: Decimal, market_reference_price: Decimal) -> Decimal:
    """Return procurement divided by market unit price as a multiplier."""
    procurement = _require_positive_price(
        _require_decimal(procurement_unit_price, "procurement_unit_price"), "procurement_unit_price"
    )
    market = _require_positive_price(
        _require_decimal(market_reference_price, "market_reference_price"), "market_reference_price"
    )
    return procurement / market


def tax_waste_amount(
    procurement_unit_price: Decimal, market_reference_price: Decimal, contract_quantity: Decimal
) -> Decimal:
    """Return price delta times quantity in KRW."""
    procurement = _require_positive_price(
        _require_decimal(procurement_unit_price, "procurement_unit_price"), "procurement_unit_price"
    )
    market = _require_positive_price(
        _require_decimal(market_reference_price, "market_reference_price"), "market_reference_price"
    )
    quantity = _require_decimal(contract_quantity, "contract_quantity")
    if quantity <= 0:
        raise DomainValidationError("[DATA] invalid contract_quantity: must be > 0")
    with localcontext() as context:
        context.prec = _arithmetic_precision(procurement, market, quantity)
        return (procurement - market) * quantity


def is_anomaly(procurement_unit_price: Decimal, market_reference_price: Decimal, audit: AuditConfig) -> bool:
    """Return True when procurement exceeds market times markup threshold."""
    procurement = _require_positive_price(
        _require_decimal(procurement_unit_price, "procurement_unit_price"), "procurement_unit_price"
    )
    market = _require_positive_price(
        _require_decimal(market_reference_price, "market_reference_price"), "market_reference_price"
    )
    with localcontext() as context:
        context.prec = _arithmetic_precision(procurement, market, audit.markup_threshold)
        return procurement > market * audit.markup_threshold


def _floor_non_negative(value: Decimal) -> int:
    if value <= 0:
        return 0
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def expected_amount_krw(lines: Iterable[tuple[Decimal, Decimal]], benchmark_unit_price_krw: Decimal) -> int:
    """Return the competition score in whole KRW for comparable lines at benchmark price."""
    benchmark = _require_benchmark(benchmark_unit_price_krw)
    total = 0
    for comparable_amount, quantity in lines:
        amount = _require_decimal(comparable_amount, "comparable_amount_krw")
        qty = _require_decimal(quantity, "quantity")
        if amount < 0:
            raise DomainValidationError("[DATA] invalid comparable_amount_krw: must be >= 0")
        if qty <= 0:
            raise DomainValidationError("[DATA] invalid quantity: must be > 0")
        with localcontext() as context:
            context.prec = _arithmetic_precision(amount, benchmark, qty)
            total += _floor_non_negative(amount - benchmark * qty)
    return total


def amount_lower_bound_krw(order_amount_krw: int, order_qty: int, benchmark_unit_price_krw: Decimal) -> int:
    """Return the score in whole KRW implied by aggregated order amount and quantity."""
    amount = _require_int(order_amount_krw, "order_amount_krw")
    qty = _require_int(order_qty, "order_qty")
    if amount < 0:
        raise DomainValidationError("[DATA] invalid order_amount_krw: must be >= 0")
    if qty <= 0:
        raise DomainValidationError("[DATA] invalid order_qty: must be > 0")
    benchmark = _require_benchmark(benchmark_unit_price_krw)
    with localcontext() as context:
        context.prec = _arithmetic_precision(Decimal(amount), benchmark, Decimal(qty))
        return _floor_non_negative(Decimal(amount) - benchmark * Decimal(qty))
