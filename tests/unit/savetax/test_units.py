"""Invariant guards for display-unit reconciliation."""

from src.savetax.units import UnitRelation, compare_units, normalize_unit, reconcile_units


def test_identical_units_compare() -> None:
    assert compare_units("조", "조") is UnitRelation.IDENTICAL


def test_case_variants_are_tolerated() -> None:
    assert compare_units("EA", "ea") is UnitRelation.CASE_OR_SPACING


def test_spacing_and_width_variants_are_tolerated() -> None:
    assert compare_units("대 ", "대") is UnitRelation.CASE_OR_SPACING
    assert compare_units("ＥＡ", "EA") is UnitRelation.CASE_OR_SPACING  # noqa: RUF001


def test_different_words_are_never_merged() -> None:
    assert compare_units("조", "개") is UnitRelation.INCOMPATIBLE


def test_placeholder_unit_is_unknown() -> None:
    assert compare_units("미상", "대") is UnitRelation.UNKNOWN
    assert normalize_unit("미상") is None
    assert reconcile_units("미상", "대", None).comparable is False


def test_missing_source_unit_is_not_comparable() -> None:
    assert reconcile_units("대", None, None).comparable is False


def test_display_conflict_does_not_block_comparability() -> None:
    rec = reconcile_units("조", "조", "개")
    assert rec.comparable is True
    assert rec.display_conflict is True
    assert rec.annotation is not None
    assert "개" in rec.annotation
    assert "조" in rec.annotation


def test_missing_display_label_is_no_conflict() -> None:
    rec = reconcile_units("조", "조", None)
    assert rec.display_conflict is False
    assert rec.annotation is None


def test_data_layer_mismatch_blocks_regardless_of_display() -> None:
    assert reconcile_units("조", "개", "조").comparable is False
