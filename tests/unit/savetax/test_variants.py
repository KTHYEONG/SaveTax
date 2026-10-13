"""Invariant guards for the hidden-variant gate."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.savetax.config import VariantPolicy
from src.savetax.errors import DomainValidationError
from src.savetax.variants import (
    ListingGroup,
    VariantTier,
    benchmark_admissible,
    distinct_prices,
    fact_tier,
    is_variant_bearing,
    representative,
)


def _group(prices: list[tuple[str, int]], contract: str = "C1", item: str = "I") -> ListingGroup:
    return ListingGroup(
        contract_no=contract, item_id=item, listings=tuple((lid, Decimal(str(p))) for lid, p in prices)
    )


def test_representative_is_highest_price() -> None:
    group = _group([("A", 2100000), ("B", 4510000)])
    assert representative(group) == ("B", Decimal("4510000"))


def test_representative_tie_breaks_on_listing_id() -> None:
    group = _group([("B", 3000000), ("A", 3000000)])
    assert representative(group)[0] == "A"


def test_variant_bearing_with_multi_and_sole_groups() -> None:
    multi = _group([("A", 1), ("B", 2)], contract="C1")
    soles = [_group([("X", 5)], contract=f"C{i}") for i in range(2, 5)]
    assert is_variant_bearing([multi, *soles]) is True
    assert is_variant_bearing(soles) is False


def test_equal_priced_duplicates_are_sole() -> None:
    group = _group([("A", 100), ("B", 100)])
    assert distinct_prices(group) == (Decimal("100"),)
    assert is_variant_bearing([group]) is False


def test_fact_tiers_within_multi_group() -> None:
    group = _group([("A", 4225000), ("B", 5200000)])
    assert fact_tier(group, Decimal("4225000")) is VariantTier.LOW
    assert fact_tier(group, Decimal("5200000")) is VariantTier.HIGH
    assert fact_tier(group, Decimal("5000000")) is VariantTier.UNKNOWN


def test_fact_tier_sole_group_always_sole() -> None:
    group = _group([("A", 3000000)])
    assert fact_tier(group, Decimal("3000000")) is VariantTier.SOLE
    assert fact_tier(group, Decimal("999")) is VariantTier.SOLE


def test_strict_needs_multi_benchmark() -> None:
    sole = _group([("A", 3000000)], contract="C3")
    assert benchmark_admissible(VariantPolicy.STRICT, True, VariantTier.LOW, sole) is False
    assert benchmark_admissible(VariantPolicy.ALLOW_LOW_FACT, True, VariantTier.LOW, sole) is True


def test_high_and_unknown_never_meet_sole() -> None:
    sole = _group([("A", 3000000)], contract="C3")
    for tier in (VariantTier.HIGH, VariantTier.UNKNOWN):
        assert benchmark_admissible(VariantPolicy.STRICT, True, tier, sole) is False
        assert benchmark_admissible(VariantPolicy.ALLOW_LOW_FACT, True, tier, sole) is False


def test_non_variant_items_are_unaffected() -> None:
    sole = _group([("A", 3000000)], contract="C3")
    for tier in VariantTier:
        for policy in VariantPolicy:
            assert benchmark_admissible(policy, False, tier, sole) is True
    assert benchmark_admissible(VariantPolicy.OFF, True, VariantTier.HIGH, sole) is True


def test_empty_group_is_rejected() -> None:
    group = ListingGroup(contract_no="C1", item_id="I", listings=())
    with pytest.raises(DomainValidationError):
        representative(group)


def test_group_validation_rejects_bad_inputs() -> None:
    with pytest.raises(DomainValidationError):
        ListingGroup(contract_no="", item_id="I", listings=(("A", Decimal("1")),))
    with pytest.raises(DomainValidationError):
        ListingGroup(contract_no="C", item_id="I", listings=(("A", Decimal("1")), ("A", Decimal("2"))))
    with pytest.raises(DomainValidationError):
        ListingGroup(contract_no="C", item_id="I", listings=(("A", Decimal("0")),))
    with pytest.raises(DomainValidationError):
        ListingGroup(contract_no="C", item_id="I", listings=(("A", 100),))  # type: ignore[arg-type]
