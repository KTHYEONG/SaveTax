"""Fixture recipes for patch panels and network racks."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation

from src.savetax.config import EvidenceConfig, RecipeAdder, RecipePart
from src.savetax.errors import ConfigError
from src.savetax.recipe import Recipe

__all__ = ["panel_recipe", "rack_recipe"]

_PORT_24_RE = re.compile(r"(?<!\d)24\s*(?:PORT|포트|P)(?![0-9A-Za-z])", re.IGNORECASE)
_PORT_OTHER_RE = re.compile(r"(?<!\d)(?:8|12|16|32|48)\s*(?:PORT|포트)", re.IGNORECASE)
_CAT_RE = re.compile(r"CAT\.?\s*6A?", re.IGNORECASE)
_UNTERMINATED_RE = re.compile(r"성단\s*제외")
_REVIEW_RE = re.compile(r"측면개폐|접지|멀티탭")
_DIM_KEEP_RE = re.compile(r"[^0-9.]")
_ABOVE_CLASS_RE = re.compile(r"CAT\.?\s*[78]|광|FIBER|OPTIC|(?<![A-Za-z])(?:LC|SC)(?![A-Za-z])", re.IGNORECASE)
_LABOR_DIFFERENCE = "공임은 주최 측이 공개한 정보통신 표준품셈 4-3-3 기준가로 납품에 성단이 포함됐는지와 관계없이 더했다"
_GRADE_DIFFERENCE = "조달 품목에 케이블 등급 표기가 없어 가장 높은 구리선 등급인 CAT.6A 패치판넬을 기준으로 삼았다"


def _by_key(parts: Sequence["RecipePart"], key: str) -> RecipePart:  # noqa: UP037
    for part in parts:
        if part.key == key:
            return part
    raise ConfigError(f"[SYS] invalid recipe_parts: missing {key}")


def _adder_by_key(adders: Sequence["RecipeAdder"], key: str) -> RecipeAdder:  # noqa: UP037
    for adder in adders:
        if adder.key == key:
            return adder
    raise ConfigError(f"[SYS] invalid recipe_adders: missing {key}")


def _search_text(name: str, attributes: Mapping[str, str]) -> str:
    return " ".join([name, *[str(v) for v in attributes.values()]])


def _mount_proc(attributes: Mapping[str, str]) -> str:
    for key, value in attributes.items():
        if "취부" in key and str(value).strip():
            return str(value).strip()
    return "-"


def panel_recipe(name: str, attributes: Mapping[str, str], evidence: EvidenceConfig) -> Recipe | str:
    """Map a 24-port CAT.6/6A patch panel to the organizer recipe; returns a reason code otherwise."""
    text = _search_text(name, attributes)
    kind = (attributes.get("종류") or "").strip()
    if kind not in ("패치판넬", "1U") and "패치" not in name:
        return "not_patch_panel"
    if _PORT_24_RE.search(text) is None:
        return "ports_unknown"
    if _PORT_OTHER_RE.search(text) is not None:
        return "ports_ambiguous"
    cat = _CAT_RE.search(text)
    # The reference panel is CAT.6A, so an unstated copper grade cannot exceed it; only higher or optical grades do.
    if cat is None and _ABOVE_CLASS_RE.search(text) is not None:
        return "category_out_of_class"
    grade = cat.group(0).strip() if cat is not None else "표기 없음"
    unterminated = _UNTERMINATED_RE.search(text) is not None
    adder_key = "panel:labor-mounted" if unterminated else "panel:labor-terminated"
    floor = evidence.panel_floor_unterminated_krw if unterminated else evidence.panel_floor_krw
    part = _by_key(evidence.recipe_parts, evidence.panel_part_key)
    adder = _adder_by_key(evidence.recipe_adders, adder_key)
    spec_rows = (
        ("포트 수", "24", "24"),
        ("등급", grade, "CAT.6A"),
        ("취부형태", _mount_proc(attributes), "19인치 랙형"),
    )
    return Recipe(
        kind="panel",
        parts=((part, 1),),
        adders=(adder,),
        spec_rows=spec_rows,
        differences=(_LABOR_DIFFERENCE, *(() if cat is not None else (_GRADE_DIFFERENCE,))),
        review_required=False,
        precedent_urls=("https://savetax.now/222", "https://savetax.now/540", "https://savetax.now/237"),
        title_phrase="동급 패치판넬 판매가와 공식 설치 공임 합계",
        floor_krw=floor,
    )


def _parse_dim(raw: str | None) -> Decimal | None:
    if raw is None:
        return None
    cleaned = _DIM_KEEP_RE.sub("", raw)
    if not cleaned or cleaned in (".",):
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not value.is_finite() or value <= 0:
        return None
    return value


def _fmt_mm(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{text}mm"


def rack_recipe(name: str, attributes: Mapping[str, str], evidence: EvidenceConfig) -> Recipe | str:
    """Map a 19-inch floor rack to the reference rack of its height class; returns a reason code otherwise."""
    text = _search_text(name, attributes)
    width = _parse_dim(attributes.get("외형크기(폭)"))
    height = _parse_dim(attributes.get("외형크기(높이)"))
    depth = _parse_dim(attributes.get("외형크기(깊이)"))
    if width is None or height is None or depth is None:
        return "dimensions_unknown"
    low, high = evidence.rack_width_mm
    if not low <= width <= high:
        return "width_out_of_class"
    if depth > evidence.rack_max_depth_mm:
        return "depth_out_of_class"
    if re.search(evidence.rack_exclusion_pattern, text) is not None:
        return "special_rack"
    chosen = None
    for entry in evidence.rack_classes:
        if height <= entry.max_height_mm:
            chosen = entry
            break
    if chosen is None:
        return "height_out_of_class"
    part = _by_key(evidence.recipe_parts, chosen.part_key)
    adder = _adder_by_key(evidence.recipe_adders, "rack:labor")
    spec_rows = (
        ("폭", _fmt_mm(width), "600mm"),
        ("높이", _fmt_mm(height), f"{chosen.max_height_mm}mm"),
        ("깊이", _fmt_mm(depth), "750mm"),
    )
    return Recipe(
        kind="rack",
        parts=((part, 1),),
        adders=(adder,),
        spec_rows=spec_rows,
        differences=(
            f"높이 {_fmt_mm(height)}·깊이 {_fmt_mm(depth)} 제품에 기준 높이 {chosen.max_height_mm}mm·깊이 750mm 제품을 썼다. "
            "도어 재질·측판·액세서리는 가격에 반영하지 않았다",
        ),
        review_required=_REVIEW_RE.search(text) is not None,
        precedent_urls=("https://savetax.now/176", "https://savetax.now/161", "https://savetax.now/167"),
        title_phrase="동급 허브랙 판매가와 공식 설치 공임 합계",
        floor_krw=chosen.floor_krw,
    )
