"""Hidden-variant gate for the cross-contract lane."""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from src.savetax.config import VariantPolicy
from src.savetax.errors import DomainValidationError

__all__ = [
    "ListingGroup",
    "VariantTier",
    "benchmark_admissible",
    "distinct_prices",
    "fact_tier",
    "is_variant_bearing",
    "representative",
]


class VariantTier(enum.StrEnum):
    """Position of a delivery price inside its own contract group."""

    SOLE = "sole"
    LOW = "low"
    HIGH = "high"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ListingGroup:
    """All priced listings of one item under one contract; prices in KRW."""

    contract_no: str
    item_id: str
    listings: tuple[tuple[str, Decimal], ...]

    def __post_init__(self) -> None:
        if not self.contract_no.strip() or not self.item_id.strip():
            raise DomainValidationError("[DATA] invalid listing group: empty ids")
        seen: set[str] = set()
        for listing_id, price in self.listings:
            if not listing_id.strip() or listing_id in seen:
                raise DomainValidationError("[DATA] invalid listing group: duplicate or empty listing id")
            seen.add(listing_id)
            if not isinstance(price, Decimal) or isinstance(price, bool):
                raise DomainValidationError("[DATA] invalid listing group: price must be Decimal")
            if not price.is_finite() or price <= 0:
                raise DomainValidationError("[DATA] invalid listing group: price must be > 0")


def distinct_prices(group: ListingGroup) -> tuple[Decimal, ...]:
    """Return the group's distinct prices in ascending order."""
    return tuple(sorted({price for _, price in group.listings}))


def is_variant_bearing(groups: Iterable[ListingGroup]) -> bool:
    """Tell whether any group of the item carries two or more distinct prices."""
    return any(len(distinct_prices(group)) >= 2 for group in groups)


def representative(group: ListingGroup) -> tuple[str, Decimal]:
    """Return the highest-priced listing of the group; raises DomainValidationError for an empty group."""
    if not group.listings:
        raise DomainValidationError("[DATA] invalid listing group: no listings")
    top_price = max(price for _, price in group.listings)
    top_ids = sorted(lid for lid, price in group.listings if price == top_price)
    return top_ids[0], top_price


def fact_tier(group: ListingGroup, fact_unit_price_krw: Decimal) -> VariantTier:
    """Place a delivery unit price within its own contract's listing group."""
    prices = distinct_prices(group)
    if len(prices) <= 1:
        return VariantTier.SOLE
    if fact_unit_price_krw == prices[0]:
        return VariantTier.LOW
    if fact_unit_price_krw in prices[1:]:
        return VariantTier.HIGH
    return VariantTier.UNKNOWN


def benchmark_admissible(policy: VariantPolicy, variant_bearing: bool, tier: VariantTier, benchmark: ListingGroup) -> bool:
    """Decide whether a benchmark group may be compared with a fact of the given tier."""
    if policy is VariantPolicy.OFF or not variant_bearing:
        return True
    multi = len(distinct_prices(benchmark)) >= 2
    if policy is VariantPolicy.STRICT:
        return multi
    return multi or tier is VariantTier.LOW
