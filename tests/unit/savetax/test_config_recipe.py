"""Invariant guards for recipe configuration."""

from decimal import Decimal

import pytest

from src.savetax.config import EvidenceConfig, RecipeAdder, RecipePart
from src.savetax.errors import ConfigError


def _evidence(**over: object) -> EvidenceConfig:
    return EvidenceConfig(**over)  # type: ignore[arg-type]


def test_duplicate_part_key_rejected() -> None:
    parts = (RecipePart("k", "L", "1", ("A",)), RecipePart("k", "L", "2", ("B",)))
    with pytest.raises(ConfigError):
        _evidence(recipe_parts=parts)


def test_non_numeric_pcode_rejected() -> None:
    with pytest.raises(ConfigError):
        _evidence(recipe_parts=(RecipePart("k", "L", "abc", ("A",)),))


def test_empty_name_tokens_rejected() -> None:
    with pytest.raises(ConfigError):
        _evidence(recipe_parts=(RecipePart("k", "L", "1", ()),))


def test_non_positive_adder_rejected() -> None:
    with pytest.raises(ConfigError):
        _evidence(recipe_adders=(RecipeAdder("k", "L", 0, "b"),))


def test_threshold_at_or_below_one_rejected() -> None:
    with pytest.raises(ConfigError):
        _evidence(recipe_markup_threshold=Decimal("1"))
    with pytest.raises(ConfigError):
        _evidence(recipe_markup_threshold=Decimal("0.9"))


def test_offer_and_listing_limits_rejected() -> None:
    with pytest.raises(ConfigError):
        _evidence(recipe_min_offers=0)
    with pytest.raises(ConfigError):
        _evidence(recipe_listings_max=True)  # type: ignore[arg-type]


def test_non_tuple_parts_and_adders_rejected() -> None:
    with pytest.raises(ConfigError):
        _evidence(recipe_parts=["k"])  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        _evidence(recipe_adders=["k"])  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        _evidence(recipe_parts=(RecipePart("", "L", "1", ("A",)),))
    with pytest.raises(ConfigError):
        _evidence(recipe_parts=("nope",))  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        _evidence(recipe_adders=("nope",))  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        _evidence(recipe_adders=(RecipeAdder("", "L", 5, "b"),))
    with pytest.raises(ConfigError):
        _evidence(recipe_adders=(RecipeAdder("k", "L", True, "b"),))  # type: ignore[arg-type]


@pytest.mark.parametrize("tokens", [(" ",), (None,), "A"])
def test_recipe_tokens_must_be_nonblank_strings(tokens) -> None:
    with pytest.raises(ConfigError):
        _evidence(recipe_parts=(RecipePart("k", "L", "1", tokens),))


@pytest.mark.parametrize("threshold", [True, "1.2", Decimal("sNaN"), Decimal("Infinity")])
def test_recipe_threshold_rejects_wrong_types_and_nonfinite_values(threshold) -> None:
    with pytest.raises(ConfigError):
        _evidence(recipe_markup_threshold=threshold)
