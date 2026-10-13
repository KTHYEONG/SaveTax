"""Canonical unit comparison without semantic aliasing."""

from __future__ import annotations

import enum
import re
import unicodedata
from dataclasses import dataclass

__all__ = [
    "UnitReconciliation",
    "UnitRelation",
    "compare_units",
    "normalize_unit",
    "reconcile_units",
]

_WS_RUN = re.compile(r"\s+")


class UnitRelation(enum.StrEnum):
    """How two unit labels relate after normalization."""

    IDENTICAL = "identical"
    CASE_OR_SPACING = "case_or_spacing"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class UnitReconciliation:
    """Outcome of comparing fact, data-layer and display units."""

    comparable: bool
    data_relation: UnitRelation
    display_conflict: bool
    fact_unit: str
    source_unit: str | None
    display_unit: str | None
    annotation: str | None


def normalize_unit(value: str | None) -> str | None:
    """Return the canonical comparison form of a unit label, or None when blank."""
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", value).strip()
    if not text or text == "미상":
        return None
    return _WS_RUN.sub(" ", text).casefold()


def compare_units(left: str | None, right: str | None) -> UnitRelation:
    """Classify two unit labels without any semantic aliasing."""
    l_norm = normalize_unit(left)
    r_norm = normalize_unit(right)
    if l_norm is None or r_norm is None:
        return UnitRelation.UNKNOWN
    if left == right:
        return UnitRelation.IDENTICAL
    if l_norm == r_norm:
        return UnitRelation.CASE_OR_SPACING
    return UnitRelation.INCOMPATIBLE


def reconcile_units(fact_unit: str, source_unit: str | None, display_unit: str | None) -> UnitReconciliation:
    """Decide comparability from data-layer units and describe any display-label conflict."""
    data_relation = compare_units(fact_unit, source_unit)
    comparable = data_relation in (UnitRelation.IDENTICAL, UnitRelation.CASE_OR_SPACING)
    conflict = display_unit is not None and compare_units(source_unit, display_unit) is UnitRelation.INCOMPATIBLE
    annotation = f"등재 화면 표기 단위 {display_unit}, 원시 데이터·납품 사실 단위 {source_unit}" if conflict else None
    return UnitReconciliation(
        comparable=comparable,
        data_relation=data_relation,
        display_conflict=conflict,
        fact_unit=fact_unit,
        source_unit=source_unit,
        display_unit=display_unit,
        annotation=annotation,
    )
