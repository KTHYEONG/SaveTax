"""Invariant guards for the listing-sequence series gate."""

from __future__ import annotations

from decimal import Decimal

from src.savetax.config import ListingSeries, SeriesBlock
from src.savetax.series import fact_series, group_series, listing_sequence, listing_series
from src.savetax.variants import ListingGroup


def _group(contract: str = "C1", *seqs: int, item: str = "11111111") -> ListingGroup:
    return ListingGroup(
        contract_no=contract,
        item_id=item,
        listings=tuple((f"331930619_{s}", Decimal("100")) for s in seqs),
    )


def _blocks() -> tuple[SeriesBlock, ...]:
    return (
        SeriesBlock("C1", 10, 20, ListingSeries.GENERAL),
        SeriesBlock("C1", 30, 40, ListingSeries.SEISMIC),
    )


def test_sequence_is_the_trailing_integer() -> None:
    assert listing_sequence("331930619_1190000568") == 1190000568


def test_non_numeric_tail_has_no_sequence() -> None:
    assert listing_sequence("abc_x1") is None
    assert listing_sequence("abc") is None


def test_listing_inside_a_block_is_mapped() -> None:
    blocks = _blocks()
    assert listing_series("331930619_10", "C1", blocks) is ListingSeries.GENERAL
    assert listing_series("331930619_20", "C1", blocks) is ListingSeries.GENERAL


def test_listing_outside_every_block_is_unmapped() -> None:
    assert listing_series("331930619_21", "C1", _blocks()) is None


def test_block_of_another_contract_does_not_apply() -> None:
    assert listing_series("331930619_10", "C9", _blocks()) is None


def test_sequence_less_listing_is_unmapped() -> None:
    assert listing_series("no-sequence", "C1", _blocks()) is None


def test_two_listings_in_different_series_are_both_mapped() -> None:
    group = ListingGroup(
        contract_no="C1", item_id="11111111",
        listings=(("331930619_10", Decimal("100")), ("331930619_30", Decimal("120"))),
    )
    assert group_series(group, _blocks()) == {
        "331930619_10": ListingSeries.GENERAL,
        "331930619_30": ListingSeries.SEISMIC,
    }


def test_two_listings_in_one_series_void_the_group() -> None:
    assert group_series(_group("C1", 10, 11), _blocks()) == {}


def test_partly_mapped_pair_voids_the_group() -> None:
    assert group_series(_group("C1", 10, 21), _blocks()) == {}


def test_cheaper_listing_may_be_seismic() -> None:
    group = ListingGroup(
        contract_no="C1", item_id="11111111",
        listings=(("331930619_10", Decimal("120")), ("331930619_30", Decimal("100"))),
    )
    mapping = group_series(group, _blocks())
    assert mapping["331930619_30"] is ListingSeries.SEISMIC


def test_fact_follows_its_price_matched_listing() -> None:
    group = ListingGroup(
        contract_no="C1", item_id="11111111",
        listings=(("331930619_10", Decimal("100")), ("331930619_30", Decimal("120"))),
    )
    live = {"331930619_10": Decimal("100"), "331930619_30": Decimal("120")}
    assert fact_series(group, live, Decimal("120"), _blocks()) is ListingSeries.SEISMIC


def test_discounted_fact_price_is_unresolved() -> None:
    group = ListingGroup(
        contract_no="C1", item_id="11111111",
        listings=(("331930619_10", Decimal("100")), ("331930619_30", Decimal("120"))),
    )
    live = {"331930619_10": Decimal("100"), "331930619_30": Decimal("120")}
    assert fact_series(group, live, Decimal("110"), _blocks()) is None


def test_equal_live_prices_in_both_series_are_unresolved() -> None:
    group = ListingGroup(
        contract_no="C1", item_id="11111111",
        listings=(("331930619_10", Decimal("100")), ("331930619_30", Decimal("100"))),
    )
    live = {"331930619_10": Decimal("100"), "331930619_30": Decimal("100")}
    assert fact_series(group, live, Decimal("100"), _blocks()) is None


def test_missing_live_price_is_unresolved() -> None:
    group = ListingGroup(
        contract_no="C1", item_id="11111111",
        listings=(("331930619_10", Decimal("100")), ("331930619_30", Decimal("120"))),
    )
    live = {"331930619_10": Decimal("100")}
    assert fact_series(group, live, Decimal("120"), _blocks()) is None
