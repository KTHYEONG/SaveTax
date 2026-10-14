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


def test_desktop_pc_ref_defaults_link_parts() -> None:
    from src.savetax.config import DEFAULT_DESKTOP_PC_REFS

    ev = _evidence()
    assert [r.cpu_key for r in ev.desktop_pc_refs] == ["ULTRA 5 225", "ULTRA 5 235", "ULTRA 7 265"]
    assert ev.desktop_chipset_rank == ("H810", "B860")
    assert ev.desktop_pc_max_power_w == 500
    assert len(DEFAULT_DESKTOP_PC_REFS) == 3
    keys = {p.key for p in ev.recipe_parts}
    assert {"pc:ULTRA 5 225", "pc:ULTRA 5 235", "pc:ULTRA 7 265", "reader"} <= keys


def test_desktop_pc_ref_rejects_bad_refs() -> None:
    from src.savetax.config import DesktopPcRef

    with pytest.raises(ConfigError, match="desktop_pc_refs"):
        _evidence(desktop_pc_refs=(DesktopPcRef("ULTRA 5 225", "missing:key", "H810"),))
    with pytest.raises(ConfigError, match="desktop_pc_refs"):
        _evidence(desktop_pc_refs=(DesktopPcRef("ULTRA 5 225", "pc:ULTRA 5 225", "Z999"),))
    with pytest.raises(ConfigError, match="desktop_pc_refs"):
        _evidence(desktop_pc_refs=(
            DesktopPcRef("ULTRA 5 225", "pc:ULTRA 5 225", "H810"),
            DesktopPcRef("ULTRA 5 225", "pc:ULTRA 5 235", "H810"),
        ))
    with pytest.raises(ConfigError, match="desktop_pc_refs"):
        _evidence(desktop_pc_max_power_w=0)
    with pytest.raises(ConfigError, match="desktop_pc_refs"):
        _evidence(desktop_pc_refs=("nope",))  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="desktop_pc_refs"):
        _evidence(desktop_chipset_rank=())
    with pytest.raises(ConfigError, match="desktop_pc_refs"):
        _evidence(desktop_pc_refs=())


@pytest.mark.parametrize("rank", [None, 500, "H810", ["H810", "B860"]])
def test_invalid_chipset_rank_types_raise_config_error(rank) -> None:
    with pytest.raises(ConfigError, match="desktop_pc_refs"):
        _evidence(desktop_chipset_rank=rank)


def test_invalid_reference_field_types_raise_config_error() -> None:
    from src.savetax.config import DesktopPcRef

    for ref in (DesktopPcRef(None, "pc:ULTRA 5 225", "H810"), DesktopPcRef("ULTRA 5 225", [], "H810")):
        with pytest.raises(ConfigError, match="desktop_pc_refs"):
            _evidence(desktop_pc_refs=(ref,))


def test_recipe_part_list_inputs_normalised() -> None:
    part = RecipePart("k", "L", ["https://compuzone.co.kr/a"], ["A"])  # type: ignore[arg-type]
    assert part.sources == ("https://compuzone.co.kr/a",)
    assert part.name_tokens == ("A",)
    part2 = RecipePart("k2", "L", "https://compuzone.co.kr/b", ("A",))
    assert part2.sources == ("https://compuzone.co.kr/b",)


def test_recipe_part_sources_type_guards() -> None:
    from src.savetax.config import _check_recipe_parts

    good = RecipePart("k", "L", ("https://compuzone.co.kr/a",), ("A",))
    assert _check_recipe_parts((good,)) == (good,)
    bad_str = RecipePart.__new__(RecipePart)
    object.__setattr__(bad_str, "key", "k")
    object.__setattr__(bad_str, "label", "L")
    object.__setattr__(bad_str, "sources", "https://x")
    object.__setattr__(bad_str, "name_tokens", ("A",))
    object.__setattr__(bad_str, "mount_form", None)
    object.__setattr__(bad_str, "pcode", "")
    with pytest.raises(ConfigError):
        _check_recipe_parts((bad_str,))
    bad_int = RecipePart.__new__(RecipePart)
    object.__setattr__(bad_int, "key", "k")
    object.__setattr__(bad_int, "label", "L")
    object.__setattr__(bad_int, "sources", 123)
    object.__setattr__(bad_int, "name_tokens", ("A",))
    object.__setattr__(bad_int, "mount_form", None)
    object.__setattr__(bad_int, "pcode", "")
    with pytest.raises(ConfigError):
        _check_recipe_parts((bad_int,))
    with pytest.raises(ConfigError):
        _evidence(recipe_parts=(RecipePart("k", "L", ("not a url",), ("A",)),))
    with pytest.raises(ConfigError):
        _evidence(recipe_parts=(RecipePart("k", "L", (), ("A",)),))
    with pytest.raises(ConfigError):
        _evidence(recipe_parts=(RecipePart("k", "L", ("https://compuzone.co.kr/a",), ("A",), "shelf"),))  # type: ignore[arg-type]


def test_recipe_band_upper_and_evidence_hosts_guarded() -> None:
    with pytest.raises(ConfigError):
        _evidence(recipe_band_upper=Decimal("1.20"))
    with pytest.raises(ConfigError):
        _evidence(evidence_hosts=())
    with pytest.raises(ConfigError):
        _evidence(evidence_hosts=("hackathon.savetax.now",))
    with pytest.raises(ConfigError):
        _evidence(recipe_parts=(RecipePart("k", "L", ("https://other.example/a",), ("A",)),))
    with pytest.raises(ConfigError):
        _evidence(recipe_parts=(RecipePart("k", "L", ("",), ("A",)),))
