"""Candidate screening over the local dataset."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from src.savetax.audit import amount_lower_bound_krw, is_anomaly
from src.savetax.competition import CompetitionClient, ItemState
from src.savetax.config import AuditConfig
from src.savetax.dataset import Dataset
from src.savetax.errors import ConfigError, CredentialError
from src.savetax.rawapi import RawApiClient
from src.savetax.redaction import redact

logger = logging.getLogger(__name__)

__all__ = [
    "Benchmark",
    "Candidate",
    "annotate_availability",
    "pool_size",
    "screen_batch",
    "shuffle_key",
    "verify_benchmark_conditions",
    "write_candidates",
]


@dataclass(frozen=True)
class Benchmark:
    """A cheaper listing of the same item under another contract."""

    listing_id: str
    contract_no: str
    vendor_name: str | None
    unit_price_krw: Decimal
    contract_begin: date | None
    contract_end: date | None
    conditions_verified: bool = False
    condition_note: str | None = None


@dataclass(frozen=True)
class Candidate:
    """One item queued for price verification."""

    item_id: str
    name: str
    detail_category: str
    category_l_name: str
    manufacturer: str
    model: str
    unit: str | None
    order_amount_krw: int
    order_qty: int | None
    avg_order_unit_price_krw: Decimal | None
    key_count: int
    first_order_date: date | None
    last_order_date: date | None
    order_listing_ids: tuple[str, ...]
    shuffle_position: int
    batch_index: int
    benchmark: Benchmark | None = None
    internal_gap_krw: int | None = None
    availability: ItemState | None = None


def _check_handle(handle: str) -> None:
    if not handle:
        raise ConfigError("[SYS] invalid handle: must be non-empty")
    if redact(handle) != handle:
        raise CredentialError("[SYS] token must never seed ordering")


def shuffle_key(handle: str, dataset_version: str, item_id: str) -> str:
    """Return the participant-seeded ordering key for an item."""
    _check_handle(handle)
    return hashlib.md5(f"{handle}:{dataset_version}:{item_id}".encode(), usedforsecurity=False).hexdigest()


_POOL_SQL = """
    SELECT * EXCLUDE (model_code) FROM (
        SELECT *, regexp_replace(model, '^\\([A-Za-z]{2}\\)', '') AS model_code
        FROM item_orders
    ) WHERE order_amount_krw >= ? AND list_contains(?, category_l_name)
        AND manufacturer IS NOT NULL AND trim(manufacturer) <> ''
        AND NOT list_contains(?, manufacturer)
        AND (NOT ? OR reseller)
        AND NOT regexp_matches(model_code, '\\s')
        AND regexp_matches(model_code, '[A-Za-z]') AND regexp_matches(model_code, '[0-9]')
        AND length(regexp_replace(model_code, '[^A-Za-z0-9]', '', 'g')) >= ?
"""


def _pool_params(audit: AuditConfig) -> list[object]:
    return [
        audit.min_item_order_amount_krw,
        list(audit.comparable_large_categories),
        list(audit.generic_manufacturers),
        audit.require_reseller,
        audit.min_model_alnum_length,
    ]


def pool_size(dataset: Dataset, audit: AuditConfig) -> int:
    """Return how many items pass the comparability filters."""
    row = dataset.connection.execute("SELECT count(*) FROM (" + _POOL_SQL + ")", _pool_params(audit)).fetchone()  # noqa: S608
    assert row is not None
    return int(row[0])


def _unit_price(amount: int, quantity: int) -> Decimal:
    with localcontext() as context:
        context.prec = max(28, len(str(amount)) + len(str(quantity)) + 12)
        return Decimal(amount) / Decimal(quantity)


def _benchmarks_for_batch(
    dataset: Dataset, audit: AuditConfig, candidates: Sequence[Candidate]
) -> dict[str, tuple[Benchmark, int]]:
    ids = [candidate.item_id for candidate in candidates if candidate.unit is not None]
    rows = dataset.connection.execute(
        """
        SELECT k.item_id, k.order_amount_krw, k.order_qty,
            l.listing_id, l.contract_no, l.vendor_name, l.price, l.contract_begin, l.contract_end
        FROM order_keys k JOIN listings l ON k.item_id = l.item_id AND k.unit = l.unit
            AND k.contract_no <> l.contract_no AND l.price >= ?
            AND l.contract_begin <= k.first_order_date AND l.contract_end >= k.last_order_date
        WHERE list_contains(?, k.item_id)
        QUALIFY row_number() OVER (PARTITION BY k.dlv_key ORDER BY l.price, l.listing_id) = 1
        ORDER BY k.item_id, k.dlv_key
        """,
        [str(audit.min_unit_price_krw), ids],
    ).fetchall()
    result: dict[str, tuple[Benchmark, int]] = {}
    for item_id, amount, qty, listing_id, contract_no, vendor, price, begin, end in rows:
        if not is_anomaly(_unit_price(int(amount), int(qty)), price, audit):
            continue
        benchmark = Benchmark(listing_id, contract_no, vendor, price, begin, end)
        gap = amount_lower_bound_krw(int(amount), int(qty), price)
        previous = result.get(item_id)
        if previous is not None:
            best, previous_gap = previous
            gap += previous_gap
            if (best.unit_price_krw, best.listing_id) < (price, listing_id):
                benchmark = best
        result[item_id] = benchmark, gap
    return result


def screen_batch(dataset: Dataset, audit: AuditConfig, handle: str, batch_index: int) -> list[Candidate]:
    """Return one seeded batch of candidates sorted by order amount."""
    _check_handle(handle)
    if batch_index < 0:
        raise ConfigError("[SYS] invalid batch_index: must be >= 0")
    pool_count = pool_size(dataset, audit)
    start = batch_index * audit.batch_size
    cursor = dataset.connection.execute(
        "WITH pool AS (" + _POOL_SQL + "), ordered AS ("  # noqa: S608
        "SELECT *, row_number() OVER (ORDER BY md5(? || ':' || ? || ':' || item_id), item_id) - 1 "
        "AS shuffle_position FROM pool) SELECT * FROM ordered WHERE shuffle_position >= ? "
        "AND shuffle_position < ? ORDER BY order_amount_krw DESC, item_id",
        [*_pool_params(audit), handle, dataset.version, start, start + audit.batch_size],
    )
    cols = [column[0] for column in cursor.description]
    selected = [dict(zip(cols, row, strict=True)) for row in cursor.fetchall()]
    if not selected:
        logger.info(
            "[AUDIT] batch=%d pool=%d candidates=0 with_benchmark=0 top_order_amount_krw=0",
            batch_index,
            pool_count,
        )
        return []
    candidates: list[Candidate] = []
    for rec in selected:
        item_id = str(rec["item_id"])
        pos = rec["shuffle_position"]
        unit = rec["unit"]
        order_amount = int(rec["order_amount_krw"])
        raw_qty = rec["order_qty"]
        order_qty: int | None = None if raw_qty is None else int(raw_qty)
        avg: Decimal | None = None
        if unit is not None and order_qty is not None:
            avg = _unit_price(order_amount, order_qty)
        else:
            order_qty = None
            avg = None
        raw_ids = rec["order_listing_ids"]
        ids = tuple(str(v) for v in (list(raw_ids) if raw_ids is not None else []))
        candidates.append(
            Candidate(
                item_id=item_id,
                name=str(rec["name"]),
                detail_category=str(rec["detail_category"]),
                category_l_name=str(rec["category_l_name"]),
                manufacturer=str(rec["manufacturer"]),
                model=str(rec["model"]),
                unit=None if unit is None else str(unit),
                order_amount_krw=order_amount,
                order_qty=order_qty,
                avg_order_unit_price_krw=avg,
                key_count=int(rec["key_count"]),
                first_order_date=rec["first_order_date"],
                last_order_date=rec["last_order_date"],
                order_listing_ids=ids,
                shuffle_position=pos,
                batch_index=batch_index,
            )
        )
    benchmarks = _benchmarks_for_batch(dataset, audit, candidates)
    candidates = [
        replace(
            candidate, benchmark=benchmarks[candidate.item_id][0], internal_gap_krw=benchmarks[candidate.item_id][1]
        )
        if candidate.item_id in benchmarks
        else candidate
        for candidate in candidates
    ]
    with_benchmark = sum(1 for c in candidates if c.benchmark is not None)
    top_amount = max((c.order_amount_krw for c in candidates), default=0)
    logger.info(
        "[AUDIT] batch=%d pool=%d candidates=%d with_benchmark=%d top_order_amount_krw=%d",
        batch_index,
        pool_count,
        len(candidates),
        with_benchmark,
        top_amount,
    )
    return candidates


def _with_benchmark(candidate: Candidate, verified: bool, note: str | None) -> Candidate:
    assert candidate.benchmark is not None
    return replace(
        candidate,
        benchmark=replace(candidate.benchmark, conditions_verified=verified, condition_note=note),
    )


def verify_benchmark_conditions(candidate: Candidate, raw: RawApiClient, audit: AuditConfig) -> Candidate:
    """Return the candidate with its benchmark marked verified only when conditions match."""
    bench = candidate.benchmark
    if bench is None:
        return candidate
    if len(candidate.order_listing_ids) > audit.max_condition_listings:
        result = _with_benchmark(candidate, False, "too_many_order_listings")
        logger.info(
            "[EVIDENCE] item_id=%s benchmark_listing=%s verified=%s note=%s",
            redact(candidate.item_id),
            redact(bench.listing_id),
            False,
            "too_many_order_listings",
        )
        return result
    bench_detail = raw.get_listing(bench.listing_id)
    order_details = [raw.get_listing(lid) for lid in candidate.order_listing_ids]
    note: str | None = None
    verified = False
    if bench_detail is None or any(d is None for d in order_details):
        note = "listing_not_served"
    elif bench_detail.price_kind == "catalog_reference":
        note = "benchmark_is_catalog_reference"
    elif bench_detail.price_krw != bench.unit_price_krw:
        note = "benchmark_price_changed"
    elif (
        not candidate.order_listing_ids
        or bench_detail.delivery_terms is None
        or bench_detail.supply_region is None
        or not bench_detail.delivery_terms.strip()
        or not bench_detail.supply_region.strip()
    ):
        note = "conditions_missing"
    else:
        bench_terms = str(bench_detail.delivery_terms).strip()
        bench_region = str(bench_detail.supply_region).strip()
        terms_ok = True
        region_ok = True
        for det in order_details:
            assert det is not None
            o_terms = det.delivery_terms
            o_region = det.supply_region
            if o_terms is None or str(o_terms).strip() != bench_terms:
                terms_ok = False
                break
        if not terms_ok:
            note = "delivery_terms_mismatch"
        else:
            for det in order_details:
                assert det is not None
                o_region = det.supply_region
                if o_region is None or str(o_region).strip() != bench_region:
                    region_ok = False
                    break
            if not region_ok:
                note = "supply_region_mismatch"
            else:
                verified = True
                note = None
    result = _with_benchmark(candidate, verified, note)
    logger.info(
        "[EVIDENCE] item_id=%s benchmark_listing=%s verified=%s note=%s",
        redact(candidate.item_id),
        redact(bench.listing_id),
        verified,
        note,
    )
    return result


def annotate_availability(candidates: Sequence[Candidate], competition: CompetitionClient) -> list[Candidate]:
    """Return candidates with availability filled from the pre-registration lookup."""
    if not candidates:
        return []
    distinct: list[str] = list(dict.fromkeys(c.item_id for c in candidates))
    results = competition.lookup_items(distinct)
    by_input = {r.input: r.state for r in results}
    return [replace(c, availability=by_input[c.item_id]) for c in candidates]


def _format_decimal(value: Decimal) -> str:
    return format(value, "f")


def _bench_to_json(bench: Benchmark) -> dict[str, Any]:
    return {
        "listing_id": bench.listing_id,
        "contract_no": bench.contract_no,
        "vendor_name": bench.vendor_name,
        "unit_price_krw": _format_decimal(bench.unit_price_krw),
        "contract_begin": bench.contract_begin.isoformat() if bench.contract_begin is not None else None,
        "contract_end": bench.contract_end.isoformat() if bench.contract_end is not None else None,
        "conditions_verified": bench.conditions_verified,
        "condition_note": bench.condition_note,
    }


def _candidate_to_json(candidate: Candidate) -> dict[str, Any]:
    return {
        "item_id": candidate.item_id,
        "name": candidate.name,
        "detail_category": candidate.detail_category,
        "category_l_name": candidate.category_l_name,
        "manufacturer": candidate.manufacturer,
        "model": candidate.model,
        "unit": candidate.unit,
        "order_amount_krw": candidate.order_amount_krw,
        "order_qty": candidate.order_qty,
        "avg_order_unit_price_krw": (
            _format_decimal(candidate.avg_order_unit_price_krw)
            if candidate.avg_order_unit_price_krw is not None
            else None
        ),
        "key_count": candidate.key_count,
        "first_order_date": candidate.first_order_date.isoformat() if candidate.first_order_date is not None else None,
        "last_order_date": candidate.last_order_date.isoformat() if candidate.last_order_date is not None else None,
        "order_listing_ids": list(candidate.order_listing_ids),
        "shuffle_position": candidate.shuffle_position,
        "batch_index": candidate.batch_index,
        "benchmark": _bench_to_json(candidate.benchmark) if candidate.benchmark is not None else None,
        "internal_gap_krw": candidate.internal_gap_krw,
        "availability": candidate.availability.value if candidate.availability is not None else None,
    }


def write_candidates(candidates: Sequence[Candidate], path: Path) -> Path:
    """Write candidates as JSON Lines to path atomically and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as fh:
        tmp_path = Path(fh.name)
        try:
            for candidate in candidates:
                fh.write(json.dumps(_candidate_to_json(candidate), ensure_ascii=False) + "\n")
            fh.close()
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return path
