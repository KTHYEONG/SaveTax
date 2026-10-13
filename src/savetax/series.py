"""Series gate for water-tank option listings inside one contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from src.savetax.config import ListingSeries, SeriesBlock
from src.savetax.variants import ListingGroup

__all__ = ["fact_series", "group_series", "listing_sequence", "listing_series"]


def listing_sequence(listing_id: str) -> int | None:
    """Return the trailing integer of a listing id, or None when it has none."""
    if "_" not in listing_id:
        return None
    tail = listing_id.rsplit("_", 1)[-1]
    if not tail or not tail.isascii() or not tail.isdigit():
        return None
    return int(tail)


def listing_series(listing_id: str, contract_no: str, blocks: Sequence[SeriesBlock]) -> ListingSeries | None:
    """Return the series of a listing from the block table, or None when unmapped."""
    seq = listing_sequence(listing_id)
    if seq is None:
        return None
    for block in blocks:
        if block.contract_no == contract_no and block.first_sequence <= seq <= block.last_sequence:
            return block.series
    return None


def group_series(group: ListingGroup, blocks: Sequence[SeriesBlock]) -> dict[str, ListingSeries]:
    """Map the listings of one contract group to series; listings that cannot be told apart are omitted."""
    mapped: dict[str, ListingSeries] = {}
    for listing_id, _ in group.listings:
        series = listing_series(listing_id, group.contract_no, blocks)
        if series is None:
            return {}
        mapped[listing_id] = series
    if len(group.listings) >= 2 and len(set(mapped.values())) != len(mapped):
        return {}
    return mapped


def fact_series(
    group: ListingGroup, live_prices: Mapping[str, Decimal], fact_unit_price_krw: Decimal,
    blocks: Sequence[SeriesBlock],
) -> ListingSeries | None:
    """Return the series a delivery was ordered under, or None when it cannot be determined."""
    mapping = group_series(group, blocks)
    if not mapping:
        return None
    candidates = [lid for lid in mapping if live_prices.get(lid) == fact_unit_price_krw]
    if not candidates:
        return None
    series = {mapping[lid] for lid in candidates}
    if len(series) != 1:
        return None
    return next(iter(series))
