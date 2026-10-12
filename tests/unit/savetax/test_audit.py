"""Invariant guards for audit arithmetic and scoring."""

from decimal import Decimal, localcontext

import pytest

from src.savetax.audit import (
    amount_lower_bound_krw,
    expected_amount_krw,
    is_anomaly,
    markup_ratio,
    price_delta,
    tax_waste_amount,
)
from src.savetax.config import AuditConfig
from src.savetax.errors import DomainValidationError


def test_waste_equals_delta_times_quantity() -> None:
    assert price_delta(Decimal("218000"), Decimal("132950")) == Decimal("85050")
    assert tax_waste_amount(Decimal("218000"), Decimal("132950"), Decimal("4902")) == Decimal("416915100")


def test_fractional_quantity_keeps_precision() -> None:
    assert tax_waste_amount(Decimal("1000.5"), Decimal("999.25"), Decimal("2.500001")) == Decimal("1.25") * Decimal(
        "2.500001"
    )


def test_cheaper_procurement_yields_negative_delta() -> None:
    assert price_delta(Decimal("900"), Decimal("1000")) == Decimal("-100")
    assert is_anomaly(Decimal("900"), Decimal("1000"), AuditConfig()) is False
    assert markup_ratio(Decimal("900"), Decimal("1000")) == Decimal("0.9")


def test_threshold_boundary_is_exclusive() -> None:
    audit = AuditConfig(markup_threshold=Decimal("1.2"))
    assert is_anomaly(Decimal("1200"), Decimal("1000"), audit) is False
    assert is_anomaly(Decimal("1200.01"), Decimal("1000"), audit) is True


@pytest.mark.parametrize("bad_price", [Decimal("0"), Decimal("-5")])
def test_non_positive_price_rejected(bad_price: Decimal) -> None:
    with pytest.raises(DomainValidationError):
        price_delta(bad_price, Decimal("100"))
    with pytest.raises(DomainValidationError):
        price_delta(Decimal("100"), bad_price)
    with pytest.raises(DomainValidationError):
        markup_ratio(bad_price, Decimal("100"))
    with pytest.raises(DomainValidationError):
        tax_waste_amount(bad_price, Decimal("100"), Decimal("1"))
    with pytest.raises(DomainValidationError):
        is_anomaly(bad_price, Decimal("100"), AuditConfig())


@pytest.mark.parametrize("bad_qty", [Decimal("0"), Decimal("-2")])
def test_non_positive_quantity_rejected(bad_qty: Decimal) -> None:
    with pytest.raises(DomainValidationError):
        tax_waste_amount(Decimal("100"), Decimal("50"), bad_qty)
    with pytest.raises(DomainValidationError):
        expected_amount_krw([(Decimal("100"), bad_qty)], Decimal("10"))
    with pytest.raises(DomainValidationError):
        expected_amount_krw([(Decimal("-1"), Decimal("1"))], Decimal("10"))


@pytest.mark.parametrize("bad_benchmark", [Decimal("0"), Decimal("-1")])
def test_non_positive_benchmark_rejected(bad_benchmark: Decimal) -> None:
    with pytest.raises(DomainValidationError):
        expected_amount_krw([(Decimal("100"), Decimal("1"))], bad_benchmark)
    with pytest.raises(DomainValidationError):
        amount_lower_bound_krw(100, 1, bad_benchmark)
    with pytest.raises(DomainValidationError):
        amount_lower_bound_krw(-1, 1, Decimal("10"))
    with pytest.raises(DomainValidationError):
        amount_lower_bound_krw(100, 0, Decimal("10"))


@pytest.mark.parametrize("bad", [1.0, "100", True, Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_non_decimal_currency_rejected(bad: object) -> None:
    with pytest.raises(DomainValidationError):
        price_delta(bad, Decimal("100"))  # type: ignore[arg-type]
    with pytest.raises(DomainValidationError):
        price_delta(Decimal("100"), bad)  # type: ignore[arg-type]
    with pytest.raises(DomainValidationError):
        tax_waste_amount(bad, Decimal("100"), Decimal("1"))  # type: ignore[arg-type]
    with pytest.raises(DomainValidationError):
        expected_amount_krw([(bad, Decimal("1"))], Decimal("10"))  # type: ignore[list-item]
    with pytest.raises(DomainValidationError):
        amount_lower_bound_krw(bad, 1, Decimal("10"))  # type: ignore[arg-type]


def test_score_floors_each_line_separately() -> None:
    benchmark = Decimal("333.4")
    assert expected_amount_krw([(Decimal("1001"), Decimal("3"))], benchmark) == 0
    assert expected_amount_krw([(Decimal("1002"), Decimal("3"))], benchmark) == 1
    assert expected_amount_krw([(Decimal("1001"), Decimal("3")), (Decimal("1001"), Decimal("3"))], benchmark) == 0


def test_lines_below_benchmark_contribute_zero() -> None:
    assert (
        expected_amount_krw([(Decimal("1000"), Decimal("10")), (Decimal("5000"), Decimal("10"))], Decimal("200"))
        == 3000
    )


def test_benchmark_precision_above_six_decimals_rejected() -> None:
    with pytest.raises(DomainValidationError):
        expected_amount_krw([(Decimal("10"), Decimal("1"))], Decimal("1.0000001"))
    with pytest.raises(DomainValidationError):
        amount_lower_bound_krw(10, 1, Decimal("1.0000001"))
    assert expected_amount_krw([(Decimal("10"), Decimal("1"))], Decimal("1.000001")) >= 0


def test_aggregate_estimate_within_one_krw_per_line() -> None:
    benchmark = Decimal("333.4")
    lines = [(Decimal("1001"), Decimal("3")), (Decimal("1001"), Decimal("3"))]
    lower = amount_lower_bound_krw(2002, 6, benchmark)
    assert lower == 1
    assert expected_amount_krw(lines, benchmark) >= lower - (2 - 1)


def test_aggregate_understates_when_some_lines_below_benchmark() -> None:
    lines = [(Decimal("1000"), Decimal("10")), (Decimal("5000"), Decimal("10"))]
    assert amount_lower_bound_krw(6000, 20, Decimal("200")) == 2000
    assert expected_amount_krw(lines, Decimal("200")) == 3000


def test_empty_fact_list_scores_zero() -> None:
    assert expected_amount_krw([], Decimal("100")) == 0


def test_exact_arithmetic_ignores_ambient_precision() -> None:
    with localcontext() as context:
        context.prec = 3
        assert price_delta(Decimal("1000.01"), Decimal("999")) == Decimal("1.01")
        assert tax_waste_amount(Decimal("1000.01"), Decimal("999"), Decimal("2.500001")) == Decimal("2.52500101")
        assert is_anomaly(Decimal("1200.011"), Decimal("1000.01"), AuditConfig()) is False
        assert expected_amount_krw([(Decimal("1001"), Decimal("3"))], Decimal("333.4")) == 0
        assert amount_lower_bound_krw(1001, 3, Decimal("333.4")) == 0
        assert context.prec == 3


def test_large_scores_keep_units_above_default_precision() -> None:
    amount = Decimal("100000000000000000000000000001")
    assert expected_amount_krw([(amount, Decimal("1"))], Decimal("0.000001")) == 100000000000000000000000000000
    assert amount_lower_bound_krw(int(amount), 1, Decimal("0.000001")) == 100000000000000000000000000000
