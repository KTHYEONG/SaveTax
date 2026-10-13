"""Agency evidence lane over delivery rows of the same contract and contract unit price."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation, localcontext

from src.savetax.audit import _arithmetic_precision
from src.savetax.competition import Fact, ItemDetail, ItemState
from src.savetax.config import AuditConfig, EvidenceConfig
from src.savetax.dataset import Dataset
from src.savetax.errors import ApiError, DataIntegrityError, DomainValidationError, SaveTaxError
from src.savetax.evidence import EvidenceBundle, Lane, LineProposal, PriceEvidence, RejectedFact, make_line
from src.savetax.listingrows import DeliveryPage, PageRow, delivery_page_url, parse_delivery_rows
from src.savetax.publicweb import PublicFetcher, visible_text
from src.savetax.rawapi import DeliveryRow, RawApiClient
from src.savetax.redaction import redact
from src.savetax.units import reconcile_units

logger = logging.getLogger(__name__)

__all__ = [
    "AgencyBenchmark",
    "AgencyCandidate",
    "agency_benchmark",
    "build_agency_bundle",
    "contract_order_listings",
    "find_agency_candidates",
    "request_values",
]

_CLAIM_TEMPLATE = (
    "같은 물품식별번호({item_id})를 같은 조달계약·같은 계약단가로 산 기관들 사이에 납품요구 단가가 다르다. "
    "이 납품요구보다 수량과 금액이 같거나 적은 다른 기관의 납품요구가 앞뒤 {window}일 안에 더 낮은 단가로 있었고 "
    "그 단가를 비교 단가로 삼았다. 시중 판매가와 견주는 비교가 아니라, 주최 측 「자료와 방법」"
    "(https://savetax.now/method)이 분석 유형으로 둔 「같은 물건, 다른 가격」 가운데 같은 제품인데 기관에 따라 값이 다른 경우에 해당한다. "
    "납품요구 기록 기준이며 검수·지급 완료나 특정 업체·기관의 위법·의도를 뜻하지 않는다."
)

_SCOPE_TEMPLATE = (
    "같은 물품식별번호 · 같은 조달계약 · 같은 계약단가(납품요구 기록의 계약단가 기준) · 같은 단위 · 다른 기관 · "
    "비교 납품요구의 수량과 금액이 이 납품요구 이하 · 요구일 이전과 이후 각 {window}일 안에 하나 이상 · "
    "비교 단가는 기관 {supporters}곳 이상이 그 이하로 산 단가이며 이전·이후 최저 단가보다 낮지 않음 · "
    "계약단가의 {floor}% 미만 단가는 제외 · 기관 {mass}곳 이상이 똑같이 쓴 할인 단가는 요구일 {near}일 이내 기록이 있을 때만 사용 · "
    "납품요구 단가가 비교 단가의 {threshold}배 이상."
)

_NOTES_TEMPLATE = (
    "[자료] 데이터셋 {version}과 원시 API의 납품요구 기록(listings/{{id}}/deliveries, 최종 차수), "
    "대회 물품 상세의 납품 사실. 비교 납품요구는 공개 등재 화면의 납품요구 기록 표에서 다시 확인하였다. "
    "[계산] 줄마다 예상액 = 납품요구 금액 − 비교 단가 × 수량(대회 산식). "  # noqa: RUF001
    "[따른 기준] ① 「자료와 방법」(https://savetax.now/method, 2026-10-13 확인) 7절 "
    "\"같은 물품식별번호의 같은 제품인데 업체나 기관에 따라 값이 두세 배 다릅니다\". "
    "② 같은 문서 2절 \"나라장터 납품요구 단가가 기준가의 1.2배 이상인 계약만 판정 목록에 넣습니다\". "
    "③ 대회 안내 \"같은 상품은 물품식별번호(item_id)로 판단합니다\"(https://hackathon.savetax.now/docs/README.md). "
    "④ 같은 문서 8절 \"금액은 납품요구 기록 기준입니다\". "
    "[한계] 2단계경쟁·할인행사 같은 주문별 가격 결정 방식과 한 요구서에 함께 담긴 다른 품목은 자료에 없어 조정하지 않았다. "
    "수량이나 금액이 더 큰 주문, 계약단가의 {floor}% 미만 단가, 다른 시기에만 쓰인 공통 할인 단가는 비교에서 제외하였다."
)


@dataclass(frozen=True)
class AgencyCandidate:
    """One item and its (contract_no, representative listing_id) keys worth fetching; amounts in KRW."""

    item_id: str
    keys: tuple[tuple[str, str], ...]
    order_amount_krw: int
    discount_mass_krw: int


@dataclass(frozen=True)
class AgencyBenchmark:
    """Benchmark unit price for one order row and the rows that support it."""

    price_krw: Decimal
    anchor: DeliveryRow
    before: DeliveryRow
    after: DeliveryRow
    supporter_agencies: int
    target_request_krw: Decimal | None = None
    anchor_request_krw: Decimal | None = None


def _to_decimal(value: object) -> Decimal | None:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value)
        except InvalidOperation:
            return None
    return None


def _floor_decimal(value: Decimal) -> int:
    from decimal import ROUND_FLOOR

    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def find_agency_candidates(
    dataset: Dataset, evidence: EvidenceConfig, audit: AuditConfig, *, items: Sequence[str] | None = None
) -> list[AgencyCandidate]:
    """Return items whose order keys show discounting, largest discount mass first."""
    wanted = list(dict.fromkeys(items)) if items is not None else None
    threshold = format(audit.min_unit_price_krw, "f")
    params: list[object] = [list(evidence.agency_excluded_categories), threshold]
    item_filter = ""
    if wanted is not None:
        if not wanted:
            return []
        item_filter = f"AND item_id IN ({', '.join(['?'] * len(wanted))})"
        params.extend(wanted)
    rows = dataset.connection.execute(
        "SELECT dlv_key, MIN(item_id), MIN(contract_no), MIN(listing_id), "  # noqa: S608
        "bool_or(detail_category IN (SELECT unnest(?))), "  # noqa: S608
        "MAX(dlv_lines), MAX(dlv_amount), MAX(dlv_qty), MAX(price) "  # noqa: S608
        "FROM listings WHERE dlv_key IS NOT NULL AND price >= CAST(? AS DECIMAL(38, 6)) "  # noqa: S608
        f"AND dlv_unit = unit AND dlv_qty > 0 AND dlv_amount > 0 {item_filter} "  # noqa: S608
        "GROUP BY dlv_key",  # noqa: S608
        params,
    ).fetchall()
    qualifying: dict[str, list[tuple[tuple[str, str], int, int]]] = {}
    fallback: dict[str, list[tuple[tuple[str, str], int, int]]] = {}
    for _dlv_key, item_id, contract_no, rep_listing, excluded, lines, amount, qty, pmax in rows:
        item = str(item_id)
        key = (str(contract_no), str(rep_listing))
        lines_n = int(lines) if isinstance(lines, int) and not isinstance(lines, bool) else None
        amount_d = _to_decimal(amount)
        qty_d = _to_decimal(qty)
        pmax_d = _to_decimal(pmax)
        if excluded:
            continue
        if lines_n is None or amount_d is None or qty_d is None or pmax_d is None or qty_d <= 0:
            continue
        with localcontext() as context:
            context.prec = _arithmetic_precision(pmax_d, qty_d, amount_d)
            listed_amount = pmax_d * qty_d
            mass = _floor_decimal(listed_amount - amount_d)
        if lines_n > 0:
            fallback.setdefault(item, []).append((key, int(amount_d), mass))
        if (
            lines_n >= evidence.agency_min_lines
            and amount_d >= evidence.agency_min_order_amount_krw
            and amount_d < listed_amount
        ):
            qualifying.setdefault(item, []).append((key, int(amount_d), mass))
    out: list[AgencyCandidate] = []
    for item, entries in qualifying.items():
        keys = tuple(sorted(key for key, _, _ in entries))
        out.append(
            AgencyCandidate(
                item_id=item,
                keys=keys,
                order_amount_krw=sum(a for _, a, _ in entries),
                discount_mass_krw=sum(m for _, _, m in entries),
            )
        )
    if wanted is not None:
        have = set(qualifying)
        for item in wanted:
            if item in have or item not in fallback:
                continue
            entries = fallback[item]
            keys = tuple(sorted(key for key, _, _ in entries))
            out.append(
                AgencyCandidate(
                    item_id=item,
                    keys=keys,
                    order_amount_krw=sum(a for _, a, _ in entries),
                    discount_mass_krw=sum(m for _, _, m in entries),
                )
            )
    out.sort(key=lambda c: (-c.discount_mass_krw, c.item_id))
    return out


def _is_valid_row(row: DeliveryRow) -> bool:
    values = (row.quantity, row.unit_price_krw, row.contract_unit_price_krw, row.amount_krw)
    if not all(value.is_finite() for value in values):
        return False
    if row.quantity <= 0 or row.unit_price_krw <= 0 or row.contract_unit_price_krw <= 0:
        return False
    if not row.agency_name.strip():
        return False
    with localcontext() as context:
        context.prec = _arithmetic_precision(row.unit_price_krw, row.quantity, row.amount_krw)
        return row.amount_krw == row.unit_price_krw * row.quantity


def _row_key(row: DeliveryRow) -> tuple[str, str]:
    return (row.request_no, row.line_no)


def contract_order_listings(dataset: Dataset, contract_no: str) -> tuple[str, ...]:
    """Return one listing id per ordered (contract, item) key of a contract, sorted ascending."""
    rows = dataset.connection.execute(
        "SELECT MIN(listing_id) FROM listings "  # noqa: S608
        "WHERE contract_no = ? AND dlv_key IS NOT NULL "  # noqa: S608
        "GROUP BY dlv_key",  # noqa: S608
        [contract_no],
    ).fetchall()
    return tuple(sorted(str(row[0]) for row in rows))


def request_values(row_sets: Iterable[Sequence[DeliveryRow]]) -> dict[str, Decimal]:
    """Return each request's size in KRW at contract unit prices, summed over every item of the contract."""
    seen: dict[tuple[str, str], tuple[Decimal, Decimal]] = {}
    totals: dict[str, Decimal] = {}
    unusable: set[str] = set()
    for row_set in row_sets:
        for row in row_set:
            key = (row.request_no, row.line_no)
            values = (row.quantity, row.unit_price_krw, row.contract_unit_price_krw, row.amount_krw)
            if not all(value.is_finite() for value in values):
                unusable.add(row.request_no)
                continue
            if row.quantity <= 0 or row.contract_unit_price_krw <= 0:
                unusable.add(row.request_no)
                continue
            size_inputs = (row.quantity, row.contract_unit_price_krw)
            if key in seen:
                if seen[key] != size_inputs:
                    unusable.add(row.request_no)
                continue
            seen[key] = size_inputs
            previous = totals.get(row.request_no, Decimal(0))
            with localcontext() as context:
                context.prec = _arithmetic_precision(previous, *size_inputs)
                totals[row.request_no] = previous + row.contract_unit_price_krw * row.quantity
    for request_no in unusable:
        totals.pop(request_no, None)
    return totals


def agency_benchmark(
    target: DeliveryRow, rows: Sequence[DeliveryRow], evidence: EvidenceConfig,
    request_value: Mapping[str, Decimal] | None = None,
) -> AgencyBenchmark | None:
    """Return the benchmark other agencies paid under the same contract unit price, or None."""
    if not _is_valid_row(target):
        return None
    if any(token in target.agency_name for token in evidence.agency_island_tokens if token):
        return None
    scoped = evidence.agency_request_scope
    target_request: Decimal | None = None
    scope_table: Mapping[str, Decimal] = {}
    if scoped:
        if request_value is None:
            return None
        target_request = request_value.get(target.request_no)
        if target_request is None:
            return None
        scope_table = request_value
    with localcontext() as context:
        context.prec = _arithmetic_precision(evidence.agency_price_floor_ratio, target.contract_unit_price_krw)
        floor = evidence.agency_price_floor_ratio * target.contract_unit_price_krw
    peers = [
        row
        for row in rows
        if _row_key(row) != _row_key(target)
        and _is_valid_row(row)
        and row.agency_name != target.agency_name
        and row.contract_unit_price_krw == target.contract_unit_price_krw
        and row.unit == target.unit
        and row.unit_price_krw < target.unit_price_krw
        and row.unit_price_krw >= floor
        and row.quantity <= target.quantity
        and row.amount_krw <= target.amount_krw
        and abs((row.request_date - target.request_date).days) <= evidence.agency_window_days
        and (not scoped or (
            (peer_value := scope_table.get(row.request_no)) is not None
            and target_request is not None
            and peer_value <= target_request
        ))
    ]
    before = [row for row in peers if row.request_date <= target.request_date]
    after = [row for row in peers if row.request_date >= target.request_date]
    if not before or not after:
        return None
    best: dict[str, Decimal] = {}
    for row in peers:
        if row.agency_name not in best or row.unit_price_krw < best[row.agency_name]:
            best[row.agency_name] = row.unit_price_krw
    if len(best) < evidence.agency_min_supporters:
        return None
    ordered = sorted(best.values())
    kth = ordered[evidence.agency_min_supporters - 1]
    price = max(kth, min(row.unit_price_krw for row in before), min(row.unit_price_krw for row in after))
    with localcontext() as context:
        context.prec = _arithmetic_precision(evidence.agency_markup_threshold, price)
        if target.unit_price_krw < evidence.agency_markup_threshold * price:
            return None
    supporters = [row for row in peers if row.unit_price_krw <= price]
    near = any(abs((row.request_date - target.request_date).days) <= evidence.agency_near_days for row in supporters)
    valid_same = [
        row
        for row in rows
        if _is_valid_row(row)
        and row.contract_unit_price_krw == target.contract_unit_price_krw
        and row.unit == target.unit
    ]
    for distinct in {row.unit_price_krw for row in supporters}:
        agencies = {row.agency_name for row in valid_same if row.unit_price_krw == distinct}
        if len(agencies) >= evidence.agency_mass_price_agencies and not near:
            return None

    def _anchor_key(row: DeliveryRow) -> tuple[int, date, str, str]:
        return (abs((row.request_date - target.request_date).days), row.request_date, row.request_no, row.line_no)

    anchor = min(
        [row for row in supporters if row.unit_price_krw == price], key=_anchor_key
    )
    supporter_keys = {_row_key(row) for row in supporters}
    before_here = [row for row in before if _row_key(row) in supporter_keys]
    after_here = [row for row in after if _row_key(row) in supporter_keys]
    before_pick = min(before_here, key=lambda r: ((target.request_date - r.request_date).days, r.request_no, r.line_no))
    after_pick = min(after_here, key=lambda r: ((r.request_date - target.request_date).days, r.request_no, r.line_no))
    agencies_support = {row.agency_name for row in supporters}
    if scoped:
        assert request_value is not None
        assert target_request is not None
        return AgencyBenchmark(
            price_krw=price, anchor=anchor, before=before_pick, after=after_pick,
            supporter_agencies=len(agencies_support),
            target_request_krw=target_request,
            anchor_request_krw=request_value.get(anchor.request_no),
        )
    return AgencyBenchmark(
        price_krw=price, anchor=anchor, before=before_pick, after=after_pick,
        supporter_agencies=len(agencies_support),
    )


def _format_threshold(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_won(value: Decimal) -> str:
    text = format(value, ",f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _cut_agency(name: str) -> str:
    return name[:40]


def _page_match(anchor: DeliveryRow, row: PageRow) -> bool:
    return (
        row.request_no == anchor.request_no
        and row.request_date == anchor.request_date
        and row.quantity == anchor.quantity
        and Decimal(row.unit_price_krw) == anchor.unit_price_krw
    )


def build_agency_bundle(
    candidate: AgencyCandidate, detail: ItemDetail, dataset: Dataset, raw: RawApiClient, fetcher: PublicFetcher,
    evidence: EvidenceConfig, audit: AuditConfig, *, today: date,
    allowed_states: frozenset[ItemState] = frozenset({ItemState.AVAILABLE}),
) -> EvidenceBundle | None:
    """Return the agency-lane bundle for one item, or None when no line is admissible."""
    _ = audit
    if detail.state not in allowed_states:
        return None
    if detail.item_id != candidate.item_id:
        raise DataIntegrityError("[DATA] item detail identity mismatch")
    rejected: list[RejectedFact] = []
    by_contract: dict[str, list[Fact]] = {}
    for fact in detail.facts:
        by_contract.setdefault(fact.contract_no, []).append(fact)
    key_listing: dict[str, str] = {}
    for contract_no, listing_id in candidate.keys:
        key_listing.setdefault(contract_no, listing_id)
    rows_by_contract: dict[str, tuple[DeliveryRow, ...] | None] = {}
    fetch_failed: set[str] = set()
    request_value_by_contract: dict[str, dict[str, Decimal]] = {}
    if evidence.agency_request_scope:
        fetched_cache: dict[str, tuple[DeliveryRow, ...] | None] = {}
        scope_failed: set[str] = set()
        for contract_no in by_contract:
            ids = contract_order_listings(dataset, contract_no)
            if not ids or len(ids) > evidence.agency_request_listings_max:
                scope_failed.add(contract_no)
                continue
            row_sets: list[tuple[DeliveryRow, ...]] = []
            failed = False
            for listing_id in ids:
                if listing_id not in fetched_cache:
                    try:
                        fetched = raw.list_deliveries(listing_id)
                    except (ApiError, DataIntegrityError):
                        fetched = None
                    fetched_cache[listing_id] = fetched
                else:
                    fetched = fetched_cache[listing_id]
                if fetched is None:
                    failed = True
                else:
                    row_sets.append(fetched)
            if failed:
                scope_failed.add(contract_no)
                continue
            key_id = key_listing.get(contract_no)
            if key_id is None:
                continue
            key_rows = fetched_cache.get(key_id)
            if key_id not in fetched_cache:
                try:
                    key_rows = raw.list_deliveries(key_id)
                except (ApiError, DataIntegrityError):
                    key_rows = None
                fetched_cache[key_id] = key_rows
                if key_rows is None:
                    scope_failed.add(contract_no)
                    continue
                row_sets.append(key_rows)
            rows_by_contract[contract_no] = key_rows
            request_value_by_contract[contract_no] = request_values(row_sets)
        if scope_failed:
            rejected.extend(
                RejectedFact(fact.fact_id, "request_scope_unavailable")
                for contract_no in scope_failed
                for fact in by_contract.get(contract_no, [])
            )
    else:
        for contract_no, listing_id in key_listing.items():
            if contract_no not in by_contract:
                continue
            try:
                fetched = raw.list_deliveries(listing_id)
            except (ApiError, DataIntegrityError):
                fetched = None
                fetch_failed.add(contract_no)
            if fetched is None:
                rows_by_contract[contract_no] = None
                fetch_failed.add(contract_no)
            else:
                rows_by_contract[contract_no] = fetched
        if fetch_failed:
            rejected.extend(
                RejectedFact(fact.fact_id, "rows_unavailable")
                for contract_no in fetch_failed
                for fact in by_contract.get(contract_no, [])
            )
    page_cache: dict[str, tuple[str | None, list[tuple[str, DeliveryPage | str]]]] = {}
    lines: list[LineProposal] = []
    cited_contracts: dict[str, int] = {}
    for fact in detail.facts:
        if any(item.fact_id == fact.fact_id for item in rejected):
            continue
        if not fact.scorable or fact.period_kind != "none":
            rejected.append(RejectedFact(fact.fact_id, "not_scorable"))
            continue
        if fact.request_date is None:
            rejected.append(RejectedFact(fact.fact_id, "no_request_date"))
            continue
        key_listing_id = key_listing.get(fact.contract_no)
        if key_listing_id is None or fact.contract_no not in rows_by_contract:
            rejected.append(RejectedFact(fact.fact_id, "no_agency_benchmark"))
            continue
        rows = rows_by_contract[fact.contract_no]
        assert rows is not None
        index = {_row_key(row): row for row in rows}
        row = index.get((fact.request_no, fact.line_no))
        if row is None:
            rejected.append(RejectedFact(fact.fact_id, "row_not_found"))
            continue
        if (
            row.unit_price_krw != fact.unit_price_krw
            or row.quantity != fact.quantity
            or row.request_date != fact.request_date
        ):
            rejected.append(RejectedFact(fact.fact_id, "row_fact_mismatch"))
            continue
        if not reconcile_units(fact.unit, row.unit, None).comparable:
            rejected.append(RejectedFact(fact.fact_id, "unit_mismatch"))
            continue
        benchmark = agency_benchmark(row, rows, evidence, request_value_by_contract.get(fact.contract_no))
        if benchmark is None:
            rejected.append(RejectedFact(fact.fact_id, "no_agency_benchmark"))
            continue
        if benchmark.price_krw != benchmark.price_krw.to_integral_value():
            rejected.append(RejectedFact(fact.fact_id, "benchmark_not_integral"))
            continue
        outcome = _cite_anchor(
            candidate.item_id, key_listing_id, benchmark, today, evidence, fetcher, page_cache,
        )
        if outcome is None:
            rejected.append(RejectedFact(fact.fact_id, "page_row_missing"))
            continue
        if isinstance(outcome, str):
            rejected.append(RejectedFact(fact.fact_id, outcome))
            continue
        source_url, price_evidence = outcome
        _ = source_url
        try:
            line = make_line(fact, benchmark.price_krw, [price_evidence])
        except DomainValidationError:
            rejected.append(RejectedFact(fact.fact_id, "no_agency_benchmark"))
            continue
        lines.append(line)
        cited_contracts[fact.contract_no] = cited_contracts.get(fact.contract_no, 0) + 1
    total = sum(line.expected_krw for line in lines)
    logger.info(
        "[EVIDENCE] item_id=%s lane=agency lines=%d rejected=%d expected_krw=%d",
        redact(candidate.item_id), len(lines), len(rejected), total,
    )
    if not lines:
        return None
    if total < evidence.min_item_expected_krw:
        return None
    ratios = [line.fact_unit_price_krw / line.benchmark_unit_price_krw for line in lines]
    title = f"같은 계약의 다른 기관 납품요구 단가 대비 {max(ratios):.2f}배"
    review_required = detail.name is not None and re.search(evidence.agency_review_name_pattern, detail.name) is not None
    window = evidence.agency_window_days
    floor_pct = int(evidence.agency_price_floor_ratio * 100)
    if evidence.agency_request_scope:
        claim = _CLAIM_TEMPLATE.format(item_id=candidate.item_id, window=window).replace(
            "이 납품요구보다 수량과 금액이 같거나 적은 다른 기관의 납품요구가",
            "이 납품요구보다 수량과 금액이 같거나 적고, 요구서 전체 규모도 같거나 작은 다른 기관의 납품요구가",
        )
        scope = _SCOPE_TEMPLATE.format(
            window=window, supporters=evidence.agency_min_supporters, floor=floor_pct,
            mass=evidence.agency_mass_price_agencies, near=evidence.agency_near_days,
            threshold=_format_threshold(evidence.agency_markup_threshold),
        ).replace(
            "비교 납품요구의 수량과 금액이 이 납품요구 이하 · ",
            "비교 납품요구의 수량과 금액이 이 납품요구 이하 · "
            "비교 납품요구가 속한 요구서 전체 규모(같은 계약의 모든 품목, 계약단가 × 수량 합)가 이 납품요구의 요구서 이하 · ",  # noqa: RUF001
        )
        notes = _NOTES_TEMPLATE.format(version=dataset.version, floor=floor_pct).replace(
            "2단계경쟁·할인행사 같은 주문별 가격 결정 방식과 한 요구서에 함께 담긴 다른 품목은 자료에 없어 조정하지 않았다. ",
            "2단계경쟁 실시 여부와 할인행사 같은 주문별 가격 결정 방식은 자료에 없어 직접 확인하지 못했다. "
            "대신 요구서 전체 규모가 더 큰 주문은 비교에서 제외하였다. ",
        )
    else:
        claim = _CLAIM_TEMPLATE.format(item_id=candidate.item_id, window=window)
        scope = _SCOPE_TEMPLATE.format(
            window=window, supporters=evidence.agency_min_supporters, floor=floor_pct,
            mass=evidence.agency_mass_price_agencies, near=evidence.agency_near_days,
            threshold=_format_threshold(evidence.agency_markup_threshold),
        )
        notes = _NOTES_TEMPLATE.format(version=dataset.version, floor=floor_pct)
    caveats = tuple(
        f"계약 {contract_no}: 비교 납품요구 {count}건 인용"
        for contract_no, count in sorted(cited_contracts.items())
    )
    return EvidenceBundle(
        item_id=candidate.item_id, lane=Lane.AGENCY, title=title, claim=claim,
        comparison_scope=scope, reproduction_notes=notes, lines=tuple(lines),
        rejected=tuple(rejected), caveats=caveats, review_required=review_required,
        generated_at=datetime.now(UTC), unit_notes=(),
    )


def _cite_anchor(
    item_id: str, listing_id: str, benchmark: AgencyBenchmark, today: date,
    evidence: EvidenceConfig, fetcher: PublicFetcher,
    page_cache: dict[str, tuple[str | None, list[tuple[str, DeliveryPage | str]]]],
) -> tuple[str, PriceEvidence] | str | None:
    """Cite the anchor row from the public delivery-row pages, or return a reason code."""
    if listing_id not in page_cache:
        from src.savetax.crosscontract import parse_listing_page

        first_url = delivery_page_url(evidence.listing_page_url_template, listing_id, 1)
        first = fetcher.get(first_url)
        if first.status != 200:
            page_cache[listing_id] = (None, [(first_url, "page_unreadable")])
            return "page_unreadable"
        try:
            listing = parse_listing_page(visible_text(first.text))
        except (DataIntegrityError, SaveTaxError, ValueError):
            page_cache[listing_id] = (None, [(first_url, "page_identity_mismatch")])
            return "page_identity_mismatch"
        if listing.listing_id != listing_id or listing.item_id != item_id:
            page_cache[listing_id] = (None, [(first_url, "page_identity_mismatch")])
            return "page_identity_mismatch"
        try:
            parsed = parse_delivery_rows(first.text)
        except (DataIntegrityError, ValueError):
            page_cache[listing_id] = (listing.name, [(first_url, "page_row_missing")])
            return "page_row_missing"
        page_cache[listing_id] = (listing.name, [(first_url, parsed)])
    name, cached = page_cache[listing_id]
    for url, cached_page in list(cached):
        if isinstance(cached_page, str):
            return cached_page
        for prow in cached_page.rows:
            if _page_match(benchmark.anchor, prow):
                return url, _price_evidence(item_id, listing_id, name, url, prow, benchmark, today)
    first_parsed = cached[0][1]
    assert isinstance(first_parsed, DeliveryPage)
    page_count = first_parsed.page_count
    for number in range(len(cached) + 1, page_count + 1):
        url = delivery_page_url(evidence.listing_page_url_template, listing_id, number)
        page = fetcher.get(url)
        if page.status != 200:
            cached.append((url, "page_unreadable"))
            return "page_unreadable"
        try:
            parsed = parse_delivery_rows(page.text)
        except (DataIntegrityError, ValueError):
            cached.append((url, "page_row_missing"))
            return "page_row_missing"
        cached.append((url, parsed))
        for prow in parsed.rows:
            if _page_match(benchmark.anchor, prow):
                return url, _price_evidence(item_id, listing_id, name, url, prow, benchmark, today)
    return None


def _price_evidence(
    item_id: str, listing_id: str, name: str | None, url: str, prow: PageRow,
    benchmark: AgencyBenchmark, today: date,
) -> PriceEvidence:
    head = f"{name} / " if name else ""
    quote = (
        f"{head}물품식별번호 {item_id} / 계약품목관리번호 {listing_id} / "
        f"납품요구 기록: {prow.text} / savetax.now 등재 화면, {today.isoformat()} 확인"
    )
    cup = benchmark.anchor.contract_unit_price_krw

    def _seg(row: DeliveryRow) -> str:
        qty = _format_decimal(row.quantity)
        unit = row.unit or ""
        return (
            f"{row.request_date.isoformat()} {_cut_agency(row.agency_name)} "
            f"{qty}{unit} {_format_won(row.unit_price_krw)}원 {row.request_no}"
        )

    note = (
        f"같은 계약·같은 계약단가 {_format_won(cup)}원의 다른 기관 납품요구. "
        f"이전: {_seg(benchmark.before)} / 이후: {_seg(benchmark.after)} / "
        f"이 단가 이하 기관 {benchmark.supporter_agencies}곳"
    )
    if benchmark.target_request_krw is not None and benchmark.anchor_request_krw is not None:
        note += (
            f" / 요구서 규모(계약단가 기준): 이 납품요구 {_format_won(benchmark.target_request_krw)}원, "
            f"비교 {_format_won(benchmark.anchor_request_krw)}원"
        )
    if len(note) > 500:
        note = note[:500]
    return PriceEvidence(
        source_url=url, quote=quote, observed_price_krw=int(benchmark.price_krw),
        observed_at=today, vat_included=True, shipping_included=True, note=note,
    )
