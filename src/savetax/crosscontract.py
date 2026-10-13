"""Cross-contract evidence lane over the local dataset and public listing pages."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal

from src.savetax.audit import amount_lower_bound_krw
from src.savetax.competition import ItemDetail, ItemState
from src.savetax.config import AuditConfig, EvidenceConfig, UnitConflictPolicy
from src.savetax.dataset import Dataset
from src.savetax.errors import DataIntegrityError, DomainValidationError
from src.savetax.evidence import EvidenceBundle, Lane, LineProposal, PriceEvidence, RejectedFact, make_line
from src.savetax.publicweb import PublicFetcher, visible_text
from src.savetax.rawapi import ListingDetail, RawApiClient
from src.savetax.redaction import redact
from src.savetax.units import UnitRelation, compare_units, reconcile_units

logger = logging.getLogger(__name__)

__all__ = [
    "CrossCandidate",
    "ListingPage",
    "build_cross_bundle",
    "find_cross_candidates",
    "parse_listing_page",
    "region_covers",
]

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


def find_cross_candidates(dataset: Dataset, evidence: EvidenceConfig, audit: AuditConfig) -> list[CrossCandidate]:
    """Return every item with a cheaper cross-contract listing, largest gap first."""
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
        ") SELECT c.item_id, c.order_amount_krw, c.order_qty, c.price, b.ids "
        "FROM cheapest c JOIN benchmarks b USING (item_id)",
        [
            format(audit.min_unit_price_krw, "f"),
            format(evidence.cross_markup_threshold, "f"),
            evidence.max_benchmarks_per_item,
        ],
    ).fetchall()
    benchmark_ids: dict[str, tuple[str, ...]] = {}
    gaps: dict[str, int] = {}
    order_totals: dict[str, int] = {}
    for item, order_amount, order_qty, price, ids in rows:
        gap = amount_lower_bound_krw(int(order_amount), int(order_qty), Decimal(str(price)))
        gaps[item] = gaps.get(item, 0) + gap
        order_totals[item] = order_totals.get(item, 0) + order_amount
        benchmark_ids[item] = tuple(ids)
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
        supply_region=values.get("공급 지역"),
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
) -> EvidenceBundle | None:
    """Return the bundle of admissible facts for one item, or None when the total is below the minimum."""
    if detail.state is not ItemState.AVAILABLE:
        return None
    if detail.item_id != candidate.item_id:
        raise DataIntegrityError("[DATA] item detail identity mismatch")
    rejected: list[RejectedFact] = []
    for fact in detail.facts:
        if not fact.scorable:
            rejected.append(RejectedFact(fact.fact_id, "not_scorable"))
        elif fact.request_date is None:
            rejected.append(RejectedFact(fact.fact_id, "no_request_date"))
    live_facts = [f for f in detail.facts if f.scorable and f.request_date is not None]
    if not live_facts:
        logger.info(
            "[EVIDENCE] item_id=%s lane=cross lines=0 rejected=%d expected_krw=0",
            redact(candidate.item_id),
            len(rejected),
        )
        return None
    benchmarks: list[ListingDetail] = []
    for listing_id in candidate.benchmark_listing_ids:
        got = raw.get_listing(listing_id)
        if got is None or got.item_id != candidate.item_id or got.listing_id != listing_id:
            continue
        if got.price_kind == "catalog_reference":
            continue
        if not (got.delivery_terms or "").strip() or not (got.supply_region or "").strip():
            continue
        if got.contract_begin is None or got.contract_end is None:
            continue
        benchmarks.append(got)
    page_failed: set[str] = set()
    page_cache: dict[str, ListingPage] = {}
    lines: list[LineProposal] = []
    caveats: list[str] = []
    unit_notes: list[str] = []
    unit_note_keys: set[tuple[str, str]] = set()
    conflict_keys: set[tuple[str, str, str]] = set()
    release = next((b.release for b in benchmarks if b.release), "자료 접근 대기")
    for fact in live_facts:
        order_rows = _order_listings(dataset, candidate.item_id, fact.contract_no)
        if not order_rows:
            rejected.append(RejectedFact(fact.fact_id, "order_listing_not_served"))
            continue
        if len(order_rows) > audit.max_condition_listings:
            rejected.append(RejectedFact(fact.fact_id, "too_many_order_listings"))
            continue
        order_details: list[ListingDetail | None] = [raw.get_listing(lid) for lid, _ in order_rows]
        if any(
            d is None or d.listing_id != lid or d.item_id != candidate.item_id or d.contract_no != fact.contract_no
            for (lid, _), d in zip(order_rows, order_details, strict=True)
        ):
            rejected.append(RejectedFact(fact.fact_id, "order_listing_not_served"))
            continue
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
        pool = [b for b in benchmarks if b.listing_id not in page_failed]
        chosen: ListingDetail | None = None
        chosen_page: ListingPage | None = None
        chosen_rec = None
        failure_reason = "no_admissible_benchmark"
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
                admissible.append(b)
            if not admissible:
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
                page_failed.add(top.listing_id)
                pool = [b for b in pool if b.listing_id != top.listing_id]
                if not pool:
                    rejected.append(RejectedFact(fact.fact_id, "page_unreadable"))
                    break
                continue
            try:
                parsed = parse_listing_page(visible_text(page.text))
            except DataIntegrityError:
                page_failed.add(top.listing_id)
                pool = [b for b in pool if b.listing_id != top.listing_id]
                if not pool:
                    rejected.append(RejectedFact(fact.fact_id, "page_unparseable"))
                continue
            if parsed.listing_id != top.listing_id or parsed.item_id != candidate.item_id:
                page_failed.add(top.listing_id)
                pool = [b for b in pool if b.listing_id != top.listing_id]
                if not pool:
                    rejected.append(RejectedFact(fact.fact_id, "page_identity_mismatch"))
                continue
            if Decimal(parsed.unit_price_krw) != top.price_krw:
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
    logger.info(
        "[EVIDENCE] item_id=%s lane=cross lines=%d rejected=%d expected_krw=%d",
        redact(candidate.item_id),
        len(lines),
        len(rejected),
        total,
    )
    if lines and total < evidence.min_item_expected_krw:
        return None
    if not lines and not any(r.reason == "unit_display_conflict" for r in rejected):
        return None
    ratios = [line.fact_unit_price_krw / line.benchmark_unit_price_krw for line in lines]
    title = f"같은 물품의 다른 계약 단가 대비 {max(ratios):.2f}배" if ratios else "단위 표기 충돌로 비교 제외"
    notes = _REPRODUCTION_NOTES.format(version=dataset.version, release=release)
    bundle = EvidenceBundle(
        item_id=candidate.item_id,
        lane=Lane.CROSS_CONTRACT,
        title=title,
        claim=_CLAIM,
        comparison_scope=_COMPARISON_SCOPE,
        reproduction_notes=notes,
        lines=tuple(lines),
        rejected=tuple(rejected),
        caveats=tuple(caveats),
        review_required=False,
        generated_at=datetime.now(UTC),
        unit_notes=tuple(unit_notes),
    )
    return bundle
