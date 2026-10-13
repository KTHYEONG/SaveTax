"""Cross-contract evidence lane over the local dataset and public listing pages."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal

from src.savetax.audit import amount_lower_bound_krw
from src.savetax.competition import Fact, ItemDetail, ItemState
from src.savetax.config import AuditConfig, EvidenceConfig, ListingSeries, UnitConflictPolicy, VariantPolicy
from src.savetax.dataset import Dataset
from src.savetax.errors import ApiError, DataIntegrityError, DomainValidationError
from src.savetax.evidence import EvidenceBundle, Lane, LineProposal, PriceEvidence, RejectedFact, make_line
from src.savetax.publicweb import PublicFetcher, visible_text
from src.savetax.rawapi import ListingDetail, RawApiClient
from src.savetax.redaction import redact
from src.savetax.series import fact_series, group_series
from src.savetax.units import UnitRelation, compare_units, reconcile_units
from src.savetax.variants import (
    ListingGroup,
    VariantTier,
    benchmark_admissible,
    distinct_prices,
    fact_tier,
    is_variant_bearing,
    representative,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CrossCandidate",
    "ListingPage",
    "build_cross_bundle",
    "find_cross_candidates",
    "item_listing_groups",
    "parse_listing_page",
    "region_covers",
]

_VARIANT_SCOPE_SENTENCE = "비교 계약에 같은 물품 등재가 여러 단가로 있으면 가장 높은 단가의 등재만 비교 단가로 사용하였다."
_SERIES_SCOPE_SENTENCE = "같은 물품식별번호가 한 계약 안에 제품 계열별로 따로 등재된 경우 등재 순번으로 계열을 구분하여 같은 계열의 등재끼리만 비교하였다."

_SERIES_LABEL = {"general": "일반형", "seismic": "내진형"}

_CLAIM = "납품요구 기록의 단가가 같은 물품식별번호의 다른 조달계약 단가보다 높게 나타난다."
_COMPARISON_SCOPE = "같은 물품식별번호의 다른 조달계약 가운데 납품 조건이 같고 요청일에 유효하며 공급 지역이 포함되는 등재를 비교하였다."
_REPRODUCTION_NOTES = (
    "데이터셋 {version}과 원시 API 릴리스 {release}의 납품요구·계약 조건을 대조하고 공개 등재 화면의 단가를 확인하였다."
)

_LABELS = (
    "단위",
    "물품식별번호",
    "계약품목관리번호",
    "계약 단가",
    "공급 업체",
    "납품요구",
    "계약 기간",
    "납품 조건",
    "납품요구 기록",
    "공급 지역",
    "인증",
)

_SIBLING_MARKER = "같은 물건의 다른 등재"
_PHOTO_MARKER = "사진 불러오는 중"


@dataclass(frozen=True)
class CrossCandidate:
    """One item with cheaper listings of the same item under other contracts; amounts in KRW."""

    item_id: str
    order_amount_krw: int
    gap_lower_bound_krw: int
    benchmark_listing_ids: tuple[str, ...]
    variant_bearing: bool = False


def item_listing_groups(dataset: Dataset, item_id: str, audit: AuditConfig) -> dict[str, ListingGroup]:
    """Return the item's listing groups keyed by contract number."""
    rows = dataset.connection.execute(
        "SELECT listing_id, price, contract_no FROM listings WHERE item_id = ? AND price >= CAST(? AS DECIMAL(38, 6))",
        [item_id, format(audit.min_unit_price_krw, "f")],
    ).fetchall()
    bucket: dict[str, list[tuple[str, Decimal]]] = {}
    for listing_id, price, contract_no in rows:
        dec = price if isinstance(price, Decimal) else Decimal(str(price))
        bucket.setdefault(str(contract_no), []).append((str(listing_id), dec))
    groups: dict[str, ListingGroup] = {}
    for contract_no, pairs in bucket.items():
        pairs.sort(key=lambda p: p[0])
        groups[contract_no] = ListingGroup(contract_no=contract_no, item_id=item_id, listings=tuple(pairs))
    return groups


@dataclass(frozen=True)
class ListingPage:
    """Fields printed on the public listing page; `None` when a label is absent."""

    listing_id: str
    item_id: str
    name: str | None
    unit_price_krw: int
    unit: str | None
    unit_label: str | None
    vendor_name: str | None
    contract_period: str | None
    delivery_terms: str | None
    supply_region: str | None
    change_order: str | None = None
    contract_date: date | None = None


_TERMS_LABELS = (
    "계약번호",
    "계약 방식",
    "변경 차수",
    "계약일",
    "계약 부서",
    "공급 지역",
    "인증",
    "업체 구분",
)
_TERMS_START = "계약 조건"
_TERMS_END = "계약 특수조건 원문"


def _parse_terms_block(text: str) -> dict[str, str | None]:
    text = text.split(_TERMS_END, 1)[0]
    match = re.search(r"계약 조건\s+계약번호", text)
    if match is None:
        return {}
    block = text[match.start():]
    positions: dict[str, tuple[int, int] | None] = {}
    cursor = 0
    for label in _TERMS_LABELS:
        pos = block.find(label, cursor)
        if pos < 0:
            positions[label] = None
        else:
            positions[label] = (pos, pos + len(label))
            cursor = pos + len(label)
    values: dict[str, str | None] = {}
    ordered = [label for label in _TERMS_LABELS if positions[label] is not None]
    for index, label in enumerate(ordered):
        label_end = positions[label][1]  # type: ignore[index]
        next_start = positions[ordered[index + 1]][0] if index + 1 < len(ordered) else len(block)  # type: ignore[index]
        chunk = block[label_end:next_start].strip()
        values[label] = chunk or None
    for label in _TERMS_LABELS:
        values.setdefault(label, None)
    return values


def _parse_terms_change_order(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", value)
    if not digits:
        return None
    return str(int(digits))


def _parse_terms_contract_date(value: str | None) -> date | None:
    if value is None:
        return None
    match = re.fullmatch(r"\s*(\d{4})\.(\d{1,2})\.(\d{1,2})\s*", value)
    if match is None:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def find_cross_candidates(dataset: Dataset, evidence: EvidenceConfig, audit: AuditConfig) -> list[CrossCandidate]:
    """Return every item with a cheaper cross-contract listing, largest gap first."""
    if evidence.variant_policy in (VariantPolicy.OFF, VariantPolicy.SERIES):
        return _find_cross_candidates_legacy(dataset, evidence, audit)
    threshold = format(audit.min_unit_price_krw, "f")
    markup = format(evidence.cross_markup_threshold, "f")
    query = (
        "WITH ranked AS ("
        "SELECT item_id, contract_no, listing_id, price, unit, contract_begin, contract_end, "
        "ROW_NUMBER() OVER (PARTITION BY item_id, contract_no ORDER BY price DESC, listing_id) AS rn "
        "FROM listings WHERE price >= CAST(? AS DECIMAL(38, 6))"
        "), reps AS ("
        "SELECT item_id, contract_no, listing_id AS rep_id, price AS rep_price, unit AS rep_unit, "
        "contract_begin AS rep_begin, contract_end AS rep_end FROM ranked WHERE rn = 1"
        "), gcounts AS ("
        "SELECT item_id, contract_no, COUNT(DISTINCT price) AS dcnt FROM listings "
        "WHERE price >= CAST(? AS DECIMAL(38, 6)) GROUP BY item_id, contract_no"
        "), vbearing AS ("
        "SELECT item_id, MAX(dcnt) >= 2 AS vb FROM gcounts GROUP BY item_id"
        "), qualifying AS ("
        "SELECT k.dlv_key, k.item_id, k.order_amount_krw, k.order_qty, r.rep_id AS listing_id, r.rep_price AS price, "
        "v.vb AS vb "
        "FROM order_keys k JOIN reps r ON k.item_id = r.item_id AND lower(trim(k.unit)) = lower(trim(r.rep_unit)) "
        "AND k.contract_no <> r.contract_no "
        "JOIN gcounts g ON g.item_id = r.item_id AND g.contract_no = r.contract_no "
        "JOIN vbearing v ON v.item_id = k.item_id "
        "WHERE r.rep_begin <= k.last_order_date AND r.rep_end >= k.first_order_date "
        "AND k.avg_order_unit_price_krw >= r.rep_price * CAST(? AS DECIMAL(38, 12)) "
        "AND (NOT ? OR NOT v.vb OR g.dcnt >= 2)"
        "), benchmarks AS ("
        "SELECT item_id, list(listing_id ORDER BY price, listing_id)[:?] AS ids "
        "FROM (SELECT DISTINCT item_id, listing_id, price FROM qualifying) GROUP BY item_id"
        "), cheapest AS ("
        "SELECT * FROM qualifying QUALIFY row_number() OVER (PARTITION BY dlv_key ORDER BY price, listing_id) = 1"
        ") SELECT c.item_id, c.order_amount_krw, c.order_qty, c.price, b.ids, c.vb "
        "FROM cheapest c JOIN benchmarks b USING (item_id)"
    )
    rows = dataset.connection.execute(
        query,
        [
            threshold,
            threshold,
            markup,
            evidence.variant_policy is VariantPolicy.STRICT,
            evidence.max_benchmarks_per_item,
        ],
    ).fetchall()
    benchmark_ids: dict[str, tuple[str, ...]] = {}
    gaps: dict[str, int] = {}
    order_totals: dict[str, int] = {}
    bearing: dict[str, bool] = {}
    for item, order_amount, order_qty, price, ids, vb in rows:
        gap = amount_lower_bound_krw(int(order_amount), int(order_qty), Decimal(str(price)))
        gaps[item] = gaps.get(item, 0) + gap
        order_totals[item] = order_totals.get(item, 0) + order_amount
        benchmark_ids[item] = tuple(ids)
        bearing[item] = bool(vb)
    candidates: list[CrossCandidate] = []
    for item, gap in gaps.items():
        if gap < evidence.min_item_expected_krw:
            continue
        candidates.append(
            CrossCandidate(
                item_id=item,
                order_amount_krw=order_totals.get(item, 0),
                gap_lower_bound_krw=gap,
                benchmark_listing_ids=benchmark_ids[item],
                variant_bearing=bearing.get(item, False),
            )
        )
    candidates.sort(key=lambda c: (-c.gap_lower_bound_krw, c.item_id))
    return candidates


def _find_cross_candidates_legacy(
    dataset: Dataset, evidence: EvidenceConfig, audit: AuditConfig
) -> list[CrossCandidate]:
    rows = dataset.connection.execute(
        "WITH qualifying AS ("
        "SELECT k.dlv_key, k.item_id, k.order_amount_krw, k.order_qty, b.listing_id, b.price "
        "FROM order_keys k JOIN listings b ON k.item_id = b.item_id AND lower(trim(k.unit)) = lower(trim(b.unit)) "
        "AND k.contract_no <> b.contract_no "
        "WHERE b.price >= CAST(? AS DECIMAL(38, 6)) "
        "AND b.contract_begin <= k.last_order_date AND b.contract_end >= k.first_order_date "
        "AND k.avg_order_unit_price_krw >= b.price * CAST(? AS DECIMAL(38, 12))"
        "), benchmarks AS ("
        "SELECT item_id, list(listing_id ORDER BY price, listing_id)[:?] AS ids "
        "FROM (SELECT DISTINCT item_id, listing_id, price FROM qualifying) GROUP BY item_id"
        "), cheapest AS ("
        "SELECT * FROM qualifying QUALIFY row_number() OVER (PARTITION BY dlv_key ORDER BY price, listing_id) = 1"
        "), bearing AS ("
        "SELECT item_id, bool_or(dcnt >= 2) AS vb FROM ("
        "SELECT item_id, contract_no, count(DISTINCT price) AS dcnt FROM listings "
        "WHERE price >= CAST(? AS DECIMAL(38, 6)) GROUP BY item_id, contract_no) GROUP BY item_id"
        ") SELECT c.item_id, c.order_amount_krw, c.order_qty, c.price, b.ids, v.vb "
        "FROM cheapest c JOIN benchmarks b USING (item_id) JOIN bearing v USING (item_id)",
        [
            format(audit.min_unit_price_krw, "f"),
            format(evidence.cross_markup_threshold, "f"),
            evidence.max_benchmarks_per_item,
            format(audit.min_unit_price_krw, "f"),
        ],
    ).fetchall()
    benchmark_ids: dict[str, tuple[str, ...]] = {}
    gaps: dict[str, int] = {}
    order_totals: dict[str, int] = {}
    bearing: dict[str, bool] = {}
    for item, order_amount, order_qty, price, ids, vb in rows:
        gap = amount_lower_bound_krw(int(order_amount), int(order_qty), Decimal(str(price)))
        gaps[item] = gaps.get(item, 0) + gap
        order_totals[item] = order_totals.get(item, 0) + order_amount
        benchmark_ids[item] = tuple(ids)
        bearing[item] = bool(vb)
    candidates: list[CrossCandidate] = []
    for item, gap in gaps.items():
        if gap < evidence.min_item_expected_krw:
            continue
        candidates.append(
            CrossCandidate(
                item_id=item,
                order_amount_krw=order_totals.get(item, 0),
                gap_lower_bound_krw=gap,
                benchmark_listing_ids=benchmark_ids[item],
                variant_bearing=bearing[item],
            )
        )
    candidates.sort(key=lambda c: (-c.gap_lower_bound_krw, c.item_id))
    return candidates


def _core_region(value: str | None) -> str | None:
    if value is None:
        return None
    core = value.split("(", 1)[0].strip()
    return core or None


def region_covers(benchmark_region: str | None, order_region: str | None, evidence: EvidenceConfig) -> bool:
    """Return True when the benchmark supply region includes the ordering region."""
    bench = _core_region(benchmark_region)
    order = _core_region(order_region)
    if not bench or not order:
        return False
    return bench == order or bench == evidence.region_all_token


def _find_label(text: str, label: str, start: int) -> int | None:
    cursor = start
    while True:
        pos = text.find(label, cursor)
        if pos is None or pos < 0:
            return None
        if label == "납품요구":
            after = text[pos : pos + len("납품요구 기록")]
            if after == "납품요구 기록":
                cursor = pos + 1
                continue
        return pos


def parse_listing_page(text: str) -> ListingPage:
    """Parse visible page text; raises DataIntegrityError when the listing id, item id or unit price is missing."""
    segment = text.split(_SIBLING_MARKER, 1)[0]
    segment = segment.split(_TERMS_START, 1)[0].split(_TERMS_END, 1)[0].strip()
    positions: dict[str, tuple[int, int] | None] = {}
    cursor = 0
    for label in _LABELS:
        pos = _find_label(segment, label, cursor)
        if pos is None:
            positions[label] = None
        else:
            positions[label] = (pos, pos + len(label))
            cursor = pos + len(label)
    values: dict[str, str | None] = {}
    ordered_present = [label for label in _LABELS if positions[label] is not None]
    for index, label in enumerate(ordered_present):
        start = positions[label][1]  # type: ignore[index]
        end = positions[ordered_present[index + 1]][0] if index + 1 < len(ordered_present) else len(segment)  # type: ignore[index]
        chunk = segment[start:end].strip()
        values[label] = chunk or None
    for label in _LABELS:
        values.setdefault(label, None)
    name: str | None = None
    photo_pos = segment.find(_PHOTO_MARKER)
    unit_pos = positions["단위"][0] if positions["단위"] is not None else None
    if photo_pos >= 0 and unit_pos is not None and unit_pos > photo_pos:
        candidate = segment[photo_pos + len(_PHOTO_MARKER) : unit_pos].strip()
        name = candidate or None
    listing_id = (values.get("계약품목관리번호") or "").strip()
    item_id = (values.get("물품식별번호") or "").strip()
    price_text = values.get("계약 단가")
    unit_price: int | None = None
    unit: str | None = None
    if price_text:
        before_won, sep, after = price_text.partition("원")
        if sep:
            digits = re.sub(r"\D", "", before_won)
            if digits:
                unit_price = int(digits)
            slash = after.find("/")
            if slash >= 0:
                tail = after[slash + 1 :].strip()
                unit = tail or None
    if not listing_id or not item_id or unit_price is None:
        raise DataIntegrityError("[DATA] malformed listing page: missing identity or price")
    unit_label = (values.get("단위") or "").strip() or None
    if (
        unit_label is not None
        and unit is not None
        and compare_units(unit, unit_label) is UnitRelation.INCOMPATIBLE
    ):
        raise DataIntegrityError("[DATA] malformed listing page: conflicting unit labels")
    terms_values = _parse_terms_block(text)
    supply_region = values.get("공급 지역")
    if supply_region is None:
        supply_region = terms_values.get("공급 지역")
    return ListingPage(
        listing_id=listing_id,
        item_id=item_id,
        name=name,
        unit_price_krw=unit_price,
        unit=unit,
        unit_label=unit_label,
        vendor_name=values.get("공급 업체"),
        contract_period=values.get("계약 기간"),
        delivery_terms=values.get("납품 조건"),
        supply_region=supply_region,
        change_order=_parse_terms_change_order(terms_values.get("변경 차수")),
        contract_date=_parse_terms_contract_date(terms_values.get("계약일")),
    )


def _quote(
    page: ListingPage,
    today: date,
    fact_unit: str,
    display_unit: str | None = None,
    annotation: str | None = None,
) -> str:
    """Build the page citation with an explicit comparison-unit segment."""
    price_unit = display_unit if annotation is not None and display_unit else page.unit
    if price_unit is not None and annotation is not None:
        price_seg = f"계약 단가 {page.unit_price_krw:,}원 / {price_unit}({annotation})"
    elif page.unit is not None:
        price_seg = f"계약 단가 {page.unit_price_krw:,}원 / {page.unit}"
    else:
        price_seg = f"계약 단가 {page.unit_price_krw:,}원"
    vendor_seg = f"공급 업체 {page.vendor_name}" if page.vendor_name else None
    period_seg = f"계약 기간 {page.contract_period}" if page.contract_period else None
    region_seg = f"공급 지역 {page.supply_region}" if page.supply_region else None

    def _assemble(include_period: bool, include_region: bool) -> str:
        parts: list[str] = []
        if page.name:
            parts.append(page.name)
        parts.append(f"물품식별번호 {page.item_id}")
        parts.append(f"계약품목관리번호 {page.listing_id}")
        parts.append(price_seg)
        if vendor_seg:
            parts.append(vendor_seg)
        if period_seg and include_period:
            parts.append(period_seg)
        if page.delivery_terms:
            parts.append(f"납품 조건 {page.delivery_terms}")
        if region_seg and include_region:
            parts.append(region_seg)
        parts.append(f"비교 단위 {fact_unit}")
        parts.append(f"savetax.now 등재 화면, {today.isoformat()} 확인")
        return " / ".join(parts)

    quote = _assemble(True, True)
    if len(quote) <= 2000:
        return quote
    quote = _assemble(False, True)
    if len(quote) <= 2000:
        return quote
    quote = _assemble(False, False)
    if len(quote) <= 2000:
        return quote
    trimmed: list[str] = []
    if page.name:
        trimmed.append(page.name[:64])
    trimmed.append(f"물품식별번호 {page.item_id}")
    trimmed.append(f"계약품목관리번호 {page.listing_id}")
    trimmed.append(price_seg)
    trimmed.append(f"비교 단위 {fact_unit}")
    trimmed.append(f"savetax.now 등재 화면, {today.isoformat()} 확인")
    quote = " / ".join(trimmed)
    if len(quote) > 2000:
        raise DomainValidationError("[DATA] required quote segments exceed 2000 chars")
    return quote


def _order_listings(dataset: Dataset, item_id: str, contract_no: str) -> list[tuple[str, Decimal]]:
    rows = dataset.connection.execute(
        "SELECT listing_id, price FROM listings WHERE item_id = ? AND contract_no = ?",
        [item_id, contract_no],
    ).fetchall()
    out: list[tuple[str, Decimal]] = []
    for listing_id, price in rows:
        dec = price if isinstance(price, Decimal) else Decimal(str(price))
        out.append((str(listing_id), dec))
    return out


def _series_pool(
    candidate: CrossCandidate,
    fact: Fact,
    groups: dict[str, ListingGroup],
    series: ListingSeries,
    series_map: dict[str, dict[str, ListingSeries]],
    lid_to_contract: dict[str, str],
    cache: dict[str, ListingDetail | None],
    fetched: list[ListingDetail],
    raw: RawApiClient,
    evidence: EvidenceConfig,
    page_failed: set[str],
) -> list[ListingDetail]:
    """Fetch same-series listings of other contracts passing dataset and live checks."""
    pool: list[ListingDetail] = []
    for contract_no, group in groups.items():
        if contract_no == fact.contract_no:
            continue
        mapping = series_map.get(contract_no, {})
        for lid, price in group.listings:
            if mapping.get(lid) is not series:
                continue
            if fact.unit_price_krw < price * evidence.cross_markup_threshold:
                continue
            if lid not in cache:
                got = raw.get_listing(lid)
                cache[lid] = got
            got = cache[lid]
            if got is None or got.item_id != candidate.item_id or got.listing_id != lid:
                continue
            if got.contract_no != lid_to_contract[lid]:
                continue
            if got.price_kind == "catalog_reference":
                continue
            if not (got.delivery_terms or "").strip() or not (got.supply_region or "").strip():
                continue
            if got.contract_begin is None or got.contract_end is None:
                continue
            if got.listing_id in page_failed:
                continue
            pool.append(got)
            fetched.append(got)
    return pool


def build_cross_bundle(
    candidate: CrossCandidate,
    detail: ItemDetail,
    dataset: Dataset,
    raw: RawApiClient,
    fetcher: PublicFetcher,
    evidence: EvidenceConfig,
    audit: AuditConfig,
    *,
    today: date,
    on_unit_conflict: Callable[[dict[str, str]], None] | None = None,
    allowed_states: frozenset[ItemState] = frozenset({ItemState.AVAILABLE}),
    on_variant_result: Callable[[bool, int], None] | None = None,
) -> EvidenceBundle | None:
    """Return the bundle of admissible facts for one item, or None when the total is below the minimum."""
    if detail.state not in allowed_states:
        return None
    if detail.item_id != candidate.item_id:
        raise DataIntegrityError("[DATA] item detail identity mismatch")
    rejected: list[RejectedFact] = []
    for fact in detail.facts:
        if not fact.scorable:
            rejected.append(RejectedFact(fact.fact_id, "not_scorable"))
        elif fact.request_date is None:
            rejected.append(RejectedFact(fact.fact_id, "no_request_date"))
    groups = item_listing_groups(dataset, candidate.item_id, audit)
    variant_bearing = is_variant_bearing(groups.values())
    multi_groups = sum(1 for g in groups.values() if len(distinct_prices(g)) >= 2)
    logger.info(
        "[EVIDENCE] item_id=%s variant_bearing=%s groups=%d multi_groups=%d policy=%s",
        redact(candidate.item_id),
        variant_bearing,
        len(groups),
        multi_groups,
        evidence.variant_policy.value,
    )
    if on_variant_result is not None:
        on_variant_result(variant_bearing, 0)
    live_facts = [f for f in detail.facts if f.scorable and f.request_date is not None]
    if not live_facts:
        logger.info(
            "[EVIDENCE] item_id=%s lane=cross lines=0 rejected=%d expected_krw=0",
            redact(candidate.item_id), len(rejected),
        )
        return None
    series_mode = evidence.variant_policy is VariantPolicy.SERIES and variant_bearing
    use_variant_gate = evidence.variant_policy not in (VariantPolicy.OFF, VariantPolicy.SERIES)
    lid_to_contract: dict[str, str] = {}
    series_map: dict[str, dict[str, ListingSeries]] = {}
    series_cache: dict[str, ListingDetail | None] = {}
    series_fetched: list[ListingDetail] = []
    series_resolved = 0
    if series_mode:
        for contract_no, group in groups.items():
            series_map[contract_no] = group_series(group, evidence.series_blocks)
            for lid, _ in group.listings:
                lid_to_contract[lid] = contract_no
    rep_id_by_contract: dict[str, str] = {}
    if use_variant_gate:
        for contract_no, group in groups.items():
            if group.listings:
                rid, _ = representative(group)
                rep_id_by_contract[contract_no] = rid
            for lid, _ in group.listings:
                lid_to_contract[lid] = contract_no
    wanted_ids = candidate.benchmark_listing_ids
    if use_variant_gate:
        rep_ids = set(rep_id_by_contract.values())
        wanted_ids = tuple(
            lid for lid in candidate.benchmark_listing_ids if lid in rep_ids
        )
        if variant_bearing and evidence.variant_policy is VariantPolicy.STRICT:
            diagnostic_ids = tuple(
                representative(group)[0]
                for group in groups.values()
                if len(distinct_prices(group)) == 1
                and any(
                    fact.contract_no != group.contract_no
                    and fact.unit_price_krw >= representative(group)[1] * evidence.cross_markup_threshold
                    for fact in live_facts
                )
            )
            wanted_ids = tuple(dict.fromkeys((*wanted_ids, *diagnostic_ids)))
    benchmarks: list[ListingDetail] = []
    for listing_id in (() if series_mode else wanted_ids):
        got = raw.get_listing(listing_id)
        if got is None or got.item_id != candidate.item_id or got.listing_id != listing_id:
            continue
        if use_variant_gate and got.contract_no != lid_to_contract[listing_id]:
            continue
        if got.price_kind == "catalog_reference":
            continue
        if not (got.delivery_terms or "").strip() or not (got.supply_region or "").strip():
            continue
        if got.contract_begin is None or got.contract_end is None:
            continue
        benchmarks.append(got)
    if use_variant_gate:
        verified: list[ListingDetail] = []
        for bench in benchmarks:
            bench_contract = lid_to_contract[bench.listing_id]
            bench_group = groups[bench_contract]
            if len(distinct_prices(bench_group)) < 2:
                verified.append(bench)
                continue
            drop = False
            for sib_id, _ in bench_group.listings:
                if sib_id == bench.listing_id:
                    continue
                try:
                    sib = raw.get_listing(sib_id)
                except (ApiError, DataIntegrityError):
                    drop = True
                    break
                if (
                    sib is None
                    or sib.listing_id != sib_id
                    or sib.item_id != candidate.item_id
                    or sib.contract_no != bench.contract_no
                    or sib.price_krw > bench.price_krw
                ):
                    drop = True
                    break
            if not drop:
                verified.append(bench)
        benchmarks = verified
    page_failed: set[str] = set()
    page_cache: dict[str, ListingPage] = {}
    lines: list[LineProposal] = []
    cited_ids: list[str] = []
    caveats: list[str] = []
    unit_notes: list[str] = []
    unit_note_keys: set[tuple[str, str]] = set()
    conflict_keys: set[tuple[str, str, str]] = set()
    release = next((b.release for b in benchmarks if b.release), "자료 접근 대기")
    for fact in live_facts:
        fact_group = groups.get(fact.contract_no)
        if fact_group is None:
            rejected.append(RejectedFact(fact.fact_id, "order_listing_not_served"))
            continue
        tier = VariantTier.SOLE if series_mode else fact_tier(fact_group, fact.unit_price_krw)
        order_rows = _order_listings(dataset, candidate.item_id, fact.contract_no)
        if not order_rows:
            rejected.append(RejectedFact(fact.fact_id, "order_listing_not_served"))
            continue
        if len(order_rows) > audit.max_condition_listings:
            rejected.append(RejectedFact(fact.fact_id, "too_many_order_listings"))
            continue
        if series_mode:
            for lid, _ in order_rows:
                if lid not in series_cache:
                    series_cache[lid] = raw.get_listing(lid)
            order_details = [series_cache[lid] for lid, _ in order_rows]
        else:
            order_details = [raw.get_listing(lid) for lid, _ in order_rows]
        if any(
            d is None or d.listing_id != lid or d.item_id != candidate.item_id or d.contract_no != fact.contract_no
            for (lid, _), d in zip(order_rows, order_details, strict=True)
        ):
            rejected.append(RejectedFact(fact.fact_id, "order_listing_not_served"))
            continue
        series: ListingSeries | None = None
        if series_mode:
            live = {d.listing_id: d.price_krw for d in order_details if d is not None}
            series = fact_series(fact_group, live, fact.unit_price_krw, evidence.series_blocks)
            if series is None:
                rejected.append(RejectedFact(fact.fact_id, "variant_unresolved"))
                continue
            series_resolved += 1
        assert fact.request_date is not None
        matched = [
            (lid, price)
            for (lid, price), d in zip(order_rows, order_details, strict=True)
            if d is not None and d.price_krw == fact.unit_price_krw
        ]
        selected_ids = [lid for lid, _ in matched] if matched else [lid for lid, _ in order_rows]
        selected = [d for (lid, _), d in zip(order_rows, order_details, strict=True) if lid in set(selected_ids)]
        if any(
            d is None or not (d.delivery_terms or "").strip() or not _core_region(d.supply_region) for d in selected
        ):
            rejected.append(RejectedFact(fact.fact_id, "order_listing_not_served"))
            continue
        pairs = {
            (str(d.delivery_terms).strip(), _core_region(d.supply_region) or "") for d in selected if d is not None
        }
        if len(pairs) != 1:
            rejected.append(RejectedFact(fact.fact_id, "order_conditions_ambiguous"))
            continue
        order_terms, order_core = next(iter(pairs))
        order_region_full = next(str(d.supply_region) for d in selected if d is not None)
        price_matched = bool(matched)
        if series_mode:
            assert series is not None
            pool = _series_pool(
                candidate, fact, groups, series, series_map, lid_to_contract,
                series_cache, series_fetched, raw, evidence, page_failed,
            )
        else:
            pool = [b for b in benchmarks if b.listing_id not in page_failed]
        chosen: ListingDetail | None = None
        chosen_page: ListingPage | None = None
        chosen_rec = None
        failure_reason = "no_admissible_benchmark"
        variant_blocked = False
        while pool:
            admissible: list[ListingDetail] = []
            for b in pool:
                if b.contract_no == fact.contract_no:
                    continue
                if not reconcile_units(fact.unit, b.unit, None).comparable:
                    continue
                assert b.contract_begin is not None
                assert b.contract_end is not None
                if not (b.contract_begin <= fact.request_date <= b.contract_end):
                    continue
                if (b.delivery_terms or "").strip() != order_terms:
                    continue
                if not region_covers(b.supply_region, order_region_full, evidence):
                    continue
                if fact.unit_price_krw < b.price_krw * evidence.cross_markup_threshold:
                    continue
                if use_variant_gate:
                    b_group = groups[b.contract_no]
                    if not benchmark_admissible(
                        evidence.variant_policy, variant_bearing, tier, b_group
                    ):
                        variant_blocked = True
                        continue
                admissible.append(b)
            if not admissible:
                if variant_blocked and failure_reason == "no_admissible_benchmark":
                    failure_reason = "variant_unresolved"
                break
            admissible.sort(key=lambda b: (b.price_krw, b.listing_id))
            top = admissible[0]
            if top.listing_id in page_cache:
                cached = page_cache[top.listing_id]
                _display = cached.unit_label if cached.unit_label is not None else cached.unit
                _rec = reconcile_units(fact.unit, top.unit, _display)
                if _rec.display_conflict and evidence.unit_conflict_policy == UnitConflictPolicy.REJECT:
                    failure_reason = "unit_display_conflict"
                    pool = [b for b in pool if b.listing_id != top.listing_id]
                    continue
                chosen = top
                chosen_page = cached
                chosen_rec = _rec
                break
            url = evidence.listing_page_url_template.format(listing_id=top.listing_id)
            page = fetcher.get(url)
            if page.status != 200:
                failure_reason = "page_unreadable"
                page_failed.add(top.listing_id)
                pool = [b for b in pool if b.listing_id != top.listing_id]
                if not pool:
                    rejected.append(RejectedFact(fact.fact_id, "page_unreadable"))
                    break
                continue
            try:
                parsed = parse_listing_page(visible_text(page.text))
            except DataIntegrityError:
                failure_reason = "page_unparseable"
                page_failed.add(top.listing_id)
                pool = [b for b in pool if b.listing_id != top.listing_id]
                if not pool:
                    rejected.append(RejectedFact(fact.fact_id, "page_unparseable"))
                continue
            if parsed.listing_id != top.listing_id or parsed.item_id != candidate.item_id:
                failure_reason = "page_identity_mismatch"
                page_failed.add(top.listing_id)
                pool = [b for b in pool if b.listing_id != top.listing_id]
                if not pool:
                    rejected.append(RejectedFact(fact.fact_id, "page_identity_mismatch"))
                continue
            if Decimal(parsed.unit_price_krw) != top.price_krw:
                failure_reason = "page_price_mismatch"
                page_failed.add(top.listing_id)
                pool = [b for b in pool if b.listing_id != top.listing_id]
                if not pool:
                    rejected.append(RejectedFact(fact.fact_id, "page_price_mismatch"))
                continue
            chosen = top
            chosen_page = parsed
            page_cache[top.listing_id] = parsed
            _display = parsed.unit_label if parsed.unit_label is not None else parsed.unit
            _rec = reconcile_units(fact.unit, top.unit, _display)
            if _rec.display_conflict and _display is not None:
                conflict_key = (top.listing_id, _display, fact.unit)
                if conflict_key not in conflict_keys:
                    conflict_keys.add(conflict_key)
                    if on_unit_conflict is not None:
                        on_unit_conflict({
                            "item_id": candidate.item_id,
                            "listing_id": top.listing_id,
                            "fact_unit": fact.unit,
                            "source_unit": top.unit or "",
                            "display_unit": _display,
                            "observed_at": today.isoformat(),
                        })
            if _rec.display_conflict and evidence.unit_conflict_policy == UnitConflictPolicy.REJECT:
                failure_reason = "unit_display_conflict"
                chosen = None
                chosen_page = None
                chosen_rec = None
                pool = [b for b in pool if b.listing_id != top.listing_id]
                continue
            chosen_rec = _rec
            break
        if chosen is None or chosen_page is None:
            if not any(r.fact_id == fact.fact_id for r in rejected):
                rejected.append(RejectedFact(fact.fact_id, failure_reason))
            continue
        url = evidence.listing_page_url_template.format(listing_id=chosen.listing_id)
        _display_unit = chosen_page.unit_label if chosen_page.unit_label is not None else chosen_page.unit
        _annotation: str | None = chosen_rec.annotation if chosen_rec is not None else None
        if _annotation is None:
            _probe = reconcile_units(fact.unit, chosen.unit, _display_unit)
            _annotation = _probe.annotation
            _display_unit = _probe.display_unit
        try:
            proof = PriceEvidence(
                source_url=url,
                quote=_quote(chosen_page, today, fact.unit, _display_unit, _annotation),
                observed_price_krw=int(chosen.price_krw),
                observed_at=today,
                vat_included=True,
                shipping_included=True,
            )
            line = make_line(fact, chosen.price_krw, [proof])
        except DomainValidationError:
            rejected.append(RejectedFact(fact.fact_id, "no_admissible_benchmark"))
            continue
        lines.append(line)
        cited_ids.append(chosen.listing_id)
        if _annotation is not None and _display_unit is not None:
            _key = (chosen.listing_id, _display_unit)
            if _key not in unit_note_keys:
                unit_note_keys.add(_key)
                unit_notes.append(
                    f"단위 표기 차이: 등재 {chosen.listing_id} 화면 {_display_unit} / 데이터 {chosen.unit} / 사실 {fact.unit}"
                )
        if (chosen.supply_region or "") != order_region_full and _core_region(chosen.supply_region) == order_core:
            caveats.append(f"공급 지역 표기 차이: 등재 {chosen.supply_region} / 주문 {order_region_full}")
        if not price_matched:
            caveats.append(f"주문 단가 불일치로 계약 전체 조건을 사용: {fact.fact_id}")
    total = sum(line.expected_krw for line in lines)
    if on_variant_result is not None:
        on_variant_result(variant_bearing, sum(r.reason == "variant_unresolved" for r in rejected))
    logger.info(
        "[EVIDENCE] item_id=%s lane=cross lines=%d rejected=%d expected_krw=%d",
        redact(candidate.item_id),
        len(lines),
        len(rejected),
        total,
    )
    if series_mode:
        logger.info(
            "[EVIDENCE] item_id=%s series_resolved=%d series_unresolved=%d",
            redact(candidate.item_id),
            series_resolved,
            sum(r.reason == "variant_unresolved" for r in rejected),
        )
    if lines and total < evidence.min_item_expected_krw:
        return None
    if not lines and not any(r.reason in ("unit_display_conflict", "variant_unresolved") for r in rejected):
        return None
    ratios = [line.fact_unit_price_krw / line.benchmark_unit_price_krw for line in lines]
    title = (
        f"같은 물품의 다른 계약 단가 대비 {max(ratios):.2f}배" if ratios
        else "변형 구분 불확실로 비교 제외" if any(r.reason == "variant_unresolved" for r in rejected)
        else "단위 표기 충돌로 비교 제외"
    )
    if series_mode:
        release = next((b.release for b in series_fetched if b.release), "자료 접근 대기")
    notes = _REPRODUCTION_NOTES.format(version=dataset.version, release=release)
    comparison_scope = _COMPARISON_SCOPE
    if use_variant_gate and variant_bearing:
        comparison_scope = f"{_COMPARISON_SCOPE} {_VARIANT_SCOPE_SENTENCE}"
        cited: dict[str, str] = {}
        for match_id in cited_ids:
            match_contract = lid_to_contract.get(match_id)
            if match_contract is not None:
                cited.setdefault(match_contract, match_id)
        for cited_contract in sorted(cited):
            cited_group = groups.get(cited_contract)
            if cited_group is None or len(distinct_prices(cited_group)) < 2:
                continue
            prices = distinct_prices(cited_group)
            caveats.append(
                f"등재 단가 복수: 계약 {cited_contract} 등재 {len(cited_group.listings)}건 "
                f"({prices[0]:,f}원 ~ {prices[-1]:,f}원), 최고가 {cited[cited_contract]} 사용"
            )
    elif series_mode:
        comparison_scope = f"{_COMPARISON_SCOPE} {_SERIES_SCOPE_SENTENCE}"
        cited = {}
        for match_id in cited_ids:
            match_contract = lid_to_contract.get(match_id)
            if match_contract is not None:
                cited.setdefault(match_contract, match_id)
        for cited_contract in sorted(cited):
            hit = series_map.get(cited_contract, {}).get(cited[cited_contract])
            label = _SERIES_LABEL.get(hit.value if hit is not None else "", "")
            caveats.append(f"계열 구분: 계약 {cited_contract} 등재 {cited[cited_contract]} ({label} 계열)")
    bundle = EvidenceBundle(
        item_id=candidate.item_id,
        lane=Lane.CROSS_CONTRACT,
        title=title,
        claim=_CLAIM,
        comparison_scope=comparison_scope,
        reproduction_notes=notes,
        lines=tuple(lines),
        rejected=tuple(rejected),
        caveats=tuple(caveats),
        review_required=False,
        generated_at=datetime.now(UTC),
        unit_notes=tuple(unit_notes),
    )
    return bundle
