"""Desktop parts-sum recipe over labelled item attributes."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence

from src.savetax.config import EvidenceConfig, RecipePart
from src.savetax.dataset import Dataset
from src.savetax.errors import ConfigError
from src.savetax.rawapi import RawApiClient
from src.savetax.recipe import Recipe
from src.savetax.redaction import redact

logger = logging.getLogger(__name__)

__all__ = ["delivery_terms_ok", "desktop_pc_recipe", "desktop_recipe", "find_recipe_candidates"]

_CPU = "중앙처리장치(CPU)규격"
_RAM = "메모리(RAM)용량"
_SSD = "SSD용량"
_HDD = "하드디스크(HDD)용량"
_MB = "메인보드"
_PSU = "스위칭모드파워서플라이형식"
_FORM = "컴퓨터형식"
_GPU = "그래픽"
_OS = "운영체계"
_ODD = "광디스크(ODD)규격"

_GPU_RE = re.compile(r"내장|UHD|Intel Graphics|인텔 UHD|미장착", re.IGNORECASE)
_ODD_OK = ("옵션제공", "옵션 제공", "선택", "무", "미장착")
_HDD_NONE = ("미장착", "미포함", "-", "0")
_READER = "멀티메모리카드리더유무"
_READER_OK = ("유", "기본장착(제거불가)")
_PSU_WATT_RE = re.compile(r"(\d+)\s*W", re.IGNORECASE)
_PC_DIFF_BASE = "기준 완제품은 조립 PC이며 조달품의 복구·원격관리 솔루션과 보증 조건은 포함하지 않는다"
_CPU_TARGETS = ("ULTRA 5 225", "ULTRA 5 235", "ULTRA 7 265", "I5 14400", "I5 14500", "I7 14700")
_CPU_PART = {
    "ULTRA 5 225": "cpu:ULTRA 5 225",
    "ULTRA 5 235": "cpu:ULTRA 5 235",
    "ULTRA 7 265": "cpu:ULTRA 7 265",
    "I5 14400": "cpu:I5 14400",
    "I5 14500": "cpu:I5 14500",
    "I7 14700": "cpu:I7 14700",
}


def _by_key(parts: Sequence["RecipePart"], key: str) -> RecipePart:  # noqa: UP037
    for part in parts:
        if part.key == key:
            return part
    raise ConfigError(f"[SYS] invalid recipe_parts: missing {key}")


def _norm_cpu(value: str) -> str:
    text = value.upper().replace("®", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    for token in ("INTEL", "CORE"):
        text = text.replace(token, " ")
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def _cpu_key(value: str) -> str | None:
    norm = _norm_cpu(value)
    if norm == "I5 13500":
        return "I5 13500"
    return norm if norm in _CPU_TARGETS else None


def _ram_qty(value: str) -> int | None:
    match = re.fullmatch(r"(?:DDR5\s*)?(16|32)\s*(?:GB)?", value.strip(), re.IGNORECASE)
    return int(match.group(1)) // 16 if match else None


def _ssd_ok(value: str) -> bool:
    return re.fullmatch(r"(?:500|512|500\s*[~_-]\s*512|512\s*[~_-]\s*500)\s*(?:GB)?", value.strip(), re.IGNORECASE) is not None


def _mb_key(value: str) -> str | None:
    text = value.upper().replace("INTEL", "").strip()
    return text if text in ("H610", "H810", "B760", "B860") else None


def _form_key(value: str) -> str | None:
    text = value.strip()
    if text.endswith("형"):
        text = text[:-1]
    text = text.strip()
    if text in ("타워", "미들타워", "미들", "미니타워"):
        return "tower"
    if text in ("슬림", "슬림타워"):
        return "slim"
    return None


def desktop_recipe(attributes: Mapping[str, str], evidence: EvidenceConfig) -> Recipe | str:
    """Map labelled desktop attributes to the parts-sum recipe; returns a reason code when any core spec is unmapped."""
    parts: list[tuple["RecipePart", int]] = []  # noqa: UP037
    differences: list[str] = []
    review = False
    raw_cpu = (attributes.get(_CPU) or "").strip()
    cpu_key = _cpu_key(raw_cpu)
    if cpu_key is None:
        return "cpu_unmapped"
    if cpu_key == "I5 13500":
        parts.append((_by_key(evidence.recipe_parts, "cpu:I5 14500"), 1))
        differences.append("i5-13500은 시판 재고가 적어 후속 동급 i5-14500 가격을 썼다(주최 측 판정과 같은 처리)")
        review = True
        cpu_ref = "cpu:I5 14500"
    else:
        parts.append((_by_key(evidence.recipe_parts, _CPU_PART[cpu_key]), 1))
        cpu_ref = _CPU_PART[cpu_key]
    raw_ram = (attributes.get(_RAM) or "").strip()
    ram_qty = _ram_qty(raw_ram)
    if ram_qty is None:
        return "ram_unmapped"
    parts.append((_by_key(evidence.recipe_parts, "ram:16"), ram_qty))
    raw_ssd = (attributes.get(_SSD) or "").strip()
    if not _ssd_ok(raw_ssd):
        return "ssd_unmapped"
    parts.append((_by_key(evidence.recipe_parts, "ssd:500"), 1))
    hdd_ref = ""
    hdd_proc = ""
    if _HDD not in attributes:
        return "hdd_unknown"
    raw_hdd = (attributes.get(_HDD) or "").strip()
    hdd_proc = raw_hdd
    if raw_hdd in _HDD_NONE:
        hdd_ref = "없음"
    elif raw_hdd.upper().replace(" ", "") in ("2", "2TB"):
        parts.append((_by_key(evidence.recipe_parts, "hdd:2"), 1))
        hdd_ref = "HDD 2TB"
    else:
        return "hdd_unmapped"
    raw_mb = (attributes.get(_MB) or "").strip()
    mb_key = _mb_key(raw_mb)
    if mb_key is None:
        return "mainboard_unmapped"
    mb_part_key = {"H610": "mb:H610", "H810": "mb:H810", "B760": "mb:B760", "B860": "mb:B860"}[mb_key]
    parts.append((_by_key(evidence.recipe_parts, mb_part_key), 1))
    raw_form = (attributes.get(_FORM) or "").strip()
    form = _form_key(raw_form)
    if form is None:
        return "form_unmapped"
    raw_psu = (attributes.get(_PSU) or "").strip()
    psu_key: str | None = None
    if _PSU not in attributes:
        psu_key = "psu:500W" if form == "tower" else "psu:400W"
        differences.append("전원 용량 속성이 없어 형식에 맞는 기준 전원을 썼다")
        review = True
    else:
        norm_psu = raw_psu.upper().replace(" ", "")
        if norm_psu == "500W":
            psu_key = "psu:500W"
        elif norm_psu == "400W":
            psu_key = "psu:400W"
        else:
            return "psu_unmapped"
    parts.append((_by_key(evidence.recipe_parts, psu_key), 1))
    parts.append((_by_key(evidence.recipe_parts, "case:tower" if form == "tower" else "case:slim"), 1))
    raw_gpu = attributes.get(_GPU)
    if raw_gpu is not None and not _GPU_RE.search(raw_gpu):
        return "discrete_graphics"
    raw_os = attributes.get(_OS)
    if raw_os not in tuple(evidence.desktop_os_optional_values):
        return "os_included"
    raw_odd = attributes.get(_ODD)
    if raw_odd is not None and raw_odd.strip() not in _ODD_OK:
        return "odd_included"
    parts.append((_by_key(evidence.recipe_parts, "kbm"), 1))
    adders = tuple(a for a in evidence.recipe_adders if a.key in ("desktop:assembly", "desktop:reader"))
    if len(adders) != 2:
        raise ConfigError("[SYS] invalid recipe_adders: desktop adders missing")
    spec_rows = (
        ("CPU", raw_cpu, _by_key(evidence.recipe_parts, cpu_ref).label),
        ("메모리", raw_ram, _by_key(evidence.recipe_parts, "ram:16").label + (" x2" if ram_qty == 2 else "")),
        ("SSD", raw_ssd, _by_key(evidence.recipe_parts, "ssd:500").label),
        ("HDD", hdd_proc if hdd_proc else "-", hdd_ref if hdd_ref else "-"),
        ("메인보드", raw_mb, _by_key(evidence.recipe_parts, mb_part_key).label),
        ("전원", raw_psu if raw_psu else "-", _by_key(evidence.recipe_parts, psu_key).label),
        ("형식", raw_form, _by_key(evidence.recipe_parts, "case:tower" if form == "tower" else "case:slim").label),
    )
    return Recipe(
        kind="desktop",
        parts=tuple(parts),
        adders=tuple(adders),
        spec_rows=spec_rows,
        differences=tuple(differences),
        review_required=review,
        precedent_urls=("https://savetax.now/505", "https://savetax.now/405"),
        title_phrase="동급 사양 부품 합산가",
    )


def desktop_pc_recipe(attributes: Mapping[str, str], evidence: EvidenceConfig) -> Recipe | str:
    """Map desktop attributes to a finished reference PC plus separately sold parts; returns a reason code otherwise."""
    raw_cpu = (attributes.get(_CPU) or "").strip()
    cpu_key = _cpu_key(raw_cpu)
    if cpu_key == "I5 13500":
        return "cpu_unmapped"
    ref = None
    for candidate in evidence.desktop_pc_refs:
        if candidate.cpu_key == cpu_key:
            ref = candidate
            break
    if ref is None:
        return "cpu_unmapped"
    raw_ram = (attributes.get(_RAM) or "").strip()
    if re.fullmatch(r"(?:DDR5\s*)?16\s*(?:GB)?", raw_ram, re.IGNORECASE) is None:
        return "ram_unmapped"
    raw_ssd = (attributes.get(_SSD) or "").strip()
    if not _ssd_ok(raw_ssd):
        return "ssd_unmapped"
    if _HDD not in attributes:
        return "hdd_unknown"
    raw_hdd = (attributes.get(_HDD) or "").strip()
    hdd_proc = raw_hdd
    hdd_key: str | None = None
    if raw_hdd in _HDD_NONE:
        hdd_ref = "없음"
    elif raw_hdd.upper().replace(" ", "") in ("2", "2TB"):
        hdd_key = "hdd:2"
        hdd_ref = "별매 2TB 합산"
    else:
        return "hdd_unmapped"
    raw_mb = (attributes.get(_MB) or "").strip()
    mb_key = _mb_key(raw_mb)
    if mb_key is None:
        return "mainboard_unmapped"
    rank = list(evidence.desktop_chipset_rank)
    if mb_key not in rank or ref.chipset not in rank:
        return "mainboard_unmapped"
    if rank.index(mb_key) > rank.index(ref.chipset):
        return "mainboard_above_reference"
    raw_form = (attributes.get(_FORM) or "").strip()
    form = _form_key(raw_form)
    if form is None:
        return "form_unmapped"
    if _PSU in attributes:
        raw_psu = (attributes.get(_PSU) or "").strip()
        match = _PSU_WATT_RE.fullmatch(raw_psu.upper().replace(" ", ""))
        watt = int(match.group(1)) if match is not None and len(match.group(1)) <= 9 else None
        if watt is None or watt > evidence.desktop_pc_max_power_w:
            return "psu_above_reference"
    else:
        raw_psu = ""
    raw_gpu = attributes.get(_GPU)
    if raw_gpu is not None and not _GPU_RE.search(raw_gpu):
        return "discrete_graphics"
    raw_os = attributes.get(_OS)
    if raw_os not in tuple(evidence.desktop_os_optional_values):
        return "os_included"
    raw_odd = attributes.get(_ODD)
    if raw_odd is not None and raw_odd.strip() not in _ODD_OK:
        return "odd_included"
    parts: list[tuple["RecipePart", int]] = [(_by_key(evidence.recipe_parts, ref.part_key), 1)]  # noqa: UP037
    if hdd_key is not None:
        parts.append((_by_key(evidence.recipe_parts, hdd_key), 1))
    parts.append((_by_key(evidence.recipe_parts, "kbm"), 1))
    if (attributes.get(_READER) or "").strip() in _READER_OK:
        parts.append((_by_key(evidence.recipe_parts, "reader"), 1))
    differences = [_PC_DIFF_BASE]
    if form != "tower":
        differences.append(f"형식이 다르다(조달 {raw_form} / 기준 타워)")
    if mb_key != ref.chipset:
        differences.append(f"메인보드 등급은 기준 제품이 같거나 높다(조달 {raw_mb} / 기준 {ref.chipset})")
    spec_rows = (
        ("CPU", raw_cpu, cpu_key or ""),
        ("메모리", raw_ram, "16GB"),
        ("SSD", raw_ssd, "512GB"),
        ("메인보드", raw_mb, ref.chipset),
        ("전원", raw_psu if raw_psu else "-", "500W"),
        ("형식", raw_form, "타워"),
        ("HDD", hdd_proc if hdd_proc else "-", hdd_ref),
    )
    return Recipe(
        kind="desktop-pc",
        parts=tuple(parts),
        adders=(),
        spec_rows=spec_rows,
        differences=tuple(differences),
        review_required=False,
        precedent_urls=("https://savetax.now/407",),
        title_phrase="동급 사양 완제품 PC와 별매 부품 판매가 합계",
        sum_label="판매가 합계",
        exact_sum=True,
    )


def find_recipe_candidates(dataset: Dataset, category: str, *, items: Sequence[str] | None = None) -> list[str]:
    """Return item ids of a detail category that have orders, largest order amount first."""
    wanted = list(dict.fromkeys(items)) if items is not None else None
    if wanted is not None and not wanted:
        return []
    params: list[object] = [category]
    item_filter = ""
    if wanted is not None:
        item_filter = f"AND item_id IN ({', '.join(['?'] * len(wanted))})"
        params.extend(wanted)
    rows = dataset.connection.execute(
        "SELECT dlv_key, MIN(item_id), MAX(dlv_amount) FROM listings "  # noqa: S608
        "WHERE detail_category = ? AND dlv_key IS NOT NULL AND dlv_qty > 0 "  # noqa: S608
        f"{item_filter} GROUP BY dlv_key",  # noqa: S608
        params,
    ).fetchall()
    totals: dict[str, int] = {}
    for _key, item_id, amount in rows:
        item = str(item_id)
        try:
            value = int(amount)
        except (TypeError, ValueError):
            continue
        totals[item] = totals.get(item, 0) + value
    return sorted(totals, key=lambda item: (-totals[item], item))


def delivery_terms_ok(
    dataset: Dataset, raw: RawApiClient, item_id: str, contract_no: str, allowed: Sequence[str], limit: int,
) -> bool:
    """Return True when every listing of the (item, contract) carries one of the allowed delivery terms."""
    rows = dataset.connection.execute(
        "SELECT listing_id FROM listings WHERE item_id = ? AND contract_no = ?",  # noqa: S608
        [item_id, contract_no],
    ).fetchall()
    if not rows or len(rows) > limit:
        return False
    allowed_set = set(allowed)
    for (listing_id,) in rows:
        detail = raw.get_listing(str(listing_id))
        if detail is None or detail.listing_id != str(listing_id) or detail.item_id != item_id or detail.contract_no != contract_no:
            return False
        if (detail.delivery_terms or "").strip() not in allowed_set:
            logger.info("[EVIDENCE] item_id=%s contract=%s terms_mismatch", redact(item_id), redact(contract_no))
            return False
    return True
