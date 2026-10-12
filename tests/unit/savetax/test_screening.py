"""Invariant guards for candidate screening."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from decimal import localcontext
from pathlib import Path
from typing import Any

import duckdb
import pytest

from src.savetax.competition import ItemState, LookupResult
from src.savetax.config import AuditConfig, SaveTaxConfig
from src.savetax.dataset import Dataset
from src.savetax.errors import ApiError, ConfigError, CredentialError
from src.savetax.rawapi import ListingDetail
from src.savetax.screening import (
    Benchmark,
    Candidate,
    annotate_availability,
    pool_size,
    screen_batch,
    shuffle_key,
    verify_benchmark_conditions,
    write_candidates,
)

_COLUMNS = (
    "listing_id",
    "item_id",
    "name",
    "detail_category",
    "category_l_name",
    "manufacturer",
    "model",
    "unit",
    "price",
    "contract_begin",
    "contract_end",
    "vendor_name",
    "dlv_key",
    "dlv_lines",
    "dlv_qty",
    "dlv_unit",
    "dlv_amount",
    "dlv_first_date",
    "dlv_last_date",
)


def _base_row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "listing_id": "111111111_0000000001",
        "item_id": "11111111",
        "name": "레이저 프린터",
        "detail_category": "프린터",
        "category_l_name": "전자/정보/통신/영상",
        "manufacturer": "Epson",
        "model": "EB-PU2216B",
        "unit": "대",
        "price": 2736000,
        "contract_begin": "2025-01-01",
        "contract_end": "2026-01-01",
        "vendor_name": "주식회사 트리엠",
        "dlv_key": 1,
        "dlv_lines": 1,
        "dlv_qty": 10,
        "dlv_unit": "대",
        "dlv_amount": 27360000,
        "dlv_first_date": "2025-10-11",
        "dlv_last_date": "2025-10-12",
    }
    row.update(over)
    return row


def _write(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    dataset_dir = tmp_path / "savetax-products-test"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    parquet = dataset_dir / "listings.parquet"
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE tmp(listing_id VARCHAR, item_id VARCHAR, name VARCHAR, "
            "detail_category VARCHAR, category_l_name VARCHAR, manufacturer VARCHAR, "
            "model VARCHAR, unit VARCHAR, price DECIMAL(18, 3), contract_begin DATE, "
            "contract_end DATE, vendor_name VARCHAR, dlv_key INTEGER, dlv_lines INTEGER, "
            "dlv_qty BIGINT, dlv_unit VARCHAR, dlv_amount BIGINT, "
            "dlv_first_date DATE, dlv_last_date DATE)"
        )
        for row in rows:
            values = [row.get(c) for c in _COLUMNS]
            con.execute("INSERT INTO tmp VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        proj = ", ".join(_COLUMNS)
        con.execute(f"COPY (SELECT {proj} FROM tmp) TO '{parquet}' (FORMAT PARQUET)")  # noqa: S608
    finally:
        con.close()
    return dataset_dir


def _config(dataset_dir: Path) -> SaveTaxConfig:
    return SaveTaxConfig(dataset_dir=dataset_dir)


def _audit(**over: Any) -> AuditConfig:
    return AuditConfig(**over)


def _listing(
    listing_id: str,
    price: Decimal,
    terms: str | None = "현장설치도",
    region: str | None = "서울",
    kind: str = "contract",
) -> ListingDetail:
    return ListingDetail(
        listing_id=listing_id,
        item_id="11111111",
        unit="대",
        price_krw=price,
        price_kind=kind,
        contract_no="c1",
        contract_type=None,
        contract_begin=None,
        contract_end=None,
        vendor_name=None,
        manufacturer=None,
        model=None,
        category=None,
        delivery_terms=terms,
        supply_region=region,
        release=None,
    )


class _FakeRaw:
    def __init__(self, mapping: dict[str, ListingDetail | None], error: ApiError | None = None) -> None:
        self.mapping = mapping
        self.calls: list[str] = []
        self.error = error

    def get_listing(self, listing_id: str) -> ListingDetail | None:
        self.calls.append(listing_id)
        if self.error is not None:
            raise self.error
        return self.mapping.get(listing_id)


class _FakeComp:
    def __init__(self, states: dict[str, ItemState]) -> None:
        self.states = states
        self.calls: list[list[str]] = []

    def lookup_items(self, ids: list[str]) -> list[LookupResult]:
        self.calls.append(list(ids))
        return [
            LookupResult(input=i, kind="item", item_id=i, state=self.states.get(i, ItemState.AVAILABLE)) for i in ids
        ]


def _candidate(**over: Any) -> Candidate:
    base: dict[str, Any] = {
        "item_id": "11111111",
        "name": "n",
        "detail_category": "d",
        "category_l_name": "전자/정보/통신/영상",
        "manufacturer": "Epson",
        "model": "EB-PU2216B",
        "unit": "대",
        "order_amount_krw": 27360000,
        "order_qty": 10,
        "avg_order_unit_price_krw": Decimal("2736000"),
        "key_count": 1,
        "first_order_date": date(2025, 10, 11),
        "last_order_date": date(2025, 10, 12),
        "order_listing_ids": ("111111111_0000000001",),
        "shuffle_position": 0,
        "batch_index": 0,
    }
    base.update(over)
    return Candidate(**base)  # type: ignore[arg-type]


def test_shuffle_key_matches_documented_formula() -> None:
    key = shuffle_key("_kth_", "20261010-r5", "24384969")
    assert key == hashlib.md5(b"_kth_:20261010-r5:24384969").hexdigest()  # noqa: S324
    con = duckdb.connect()
    try:
        row = con.execute("SELECT md5('_kth_:20261010-r5:24384969')").fetchone()
        assert row is not None
        assert row[0] == key
    finally:
        con.close()


def test_token_can_never_seed_order() -> None:
    with pytest.raises(CredentialError):
        shuffle_key("savetax_ab-cd", "20261010-r5", "24384969")
    with pytest.raises(ConfigError):
        shuffle_key("", "20261010-r5", "24384969")


def test_same_participant_same_order(tmp_path: Path) -> None:
    rows = [_base_row(item_id=f"{10000000 + i}", listing_id=f"11111111{i}_0000000001", dlv_key=i + 1) for i in range(5)]
    d = _write(tmp_path, rows)
    with Dataset(_config(d), _audit(batch_size=10)) as ds:
        first = screen_batch(ds, _audit(batch_size=10), "a", 0)
        second = screen_batch(ds, _audit(batch_size=10), "a", 0)
        assert [c.item_id for c in first] == [c.item_id for c in second]
        assert [c.shuffle_position for c in first] == [c.shuffle_position for c in second]


def test_different_participants_different_orders(tmp_path: Path) -> None:
    rows = [
        _base_row(item_id=f"{10000000 + i}", listing_id=f"11111111{i}_0000000001", dlv_key=i + 1) for i in range(40)
    ]
    d = _write(tmp_path, rows)
    with Dataset(_config(d), _audit(batch_size=10)) as ds:
        a = {c.item_id for c in screen_batch(ds, _audit(batch_size=10), "a", 0)}
        b = {c.item_id for c in screen_batch(ds, _audit(batch_size=10), "b", 0)}
        assert a != b


def test_batches_partition_pool(tmp_path: Path) -> None:
    rows = [
        _base_row(item_id=f"{10000000 + i}", listing_id=f"11111111{i}_0000000001", dlv_key=i + 1) for i in range(25)
    ]
    d = _write(tmp_path, rows)
    audit = _audit(batch_size=10)
    with Dataset(_config(d), audit) as ds:
        assert pool_size(ds, audit) == 25
        b0 = screen_batch(ds, audit, "h", 0)
        b1 = screen_batch(ds, audit, "h", 1)
        b2 = screen_batch(ds, audit, "h", 2)
        b3 = screen_batch(ds, audit, "h", 3)
        assert (len(b0), len(b1), len(b2), len(b3)) == (10, 10, 5, 0)
        ids = [c.item_id for c in [*b0, *b1, *b2]]
        assert len(set(ids)) == 25
        with pytest.raises(ConfigError):
            screen_batch(ds, audit, "h", -1)


def test_batch_ranked_by_order_amount(tmp_path: Path) -> None:
    rows = [
        _base_row(item_id="11111111", listing_id="111111111_0000000001", dlv_key=1, dlv_amount=30000000),
        _base_row(item_id="22222222", listing_id="222222222_0000000001", dlv_key=2, dlv_amount=90000000),
        _base_row(item_id="33333333", listing_id="333333333_0000000001", dlv_key=3, dlv_amount=50000000),
    ]
    d = _write(tmp_path, rows)
    with Dataset(_config(d), _audit(batch_size=10)) as ds:
        batch = screen_batch(ds, _audit(batch_size=10), "h", 0)
        assert [c.order_amount_krw for c in batch] == [90000000, 50000000, 30000000]


def test_small_orders_filtered(tmp_path: Path) -> None:
    rows = [
        _base_row(item_id="11111111", listing_id="111111111_0000000001", dlv_key=1, dlv_amount=9999999),
        _base_row(item_id="22222222", listing_id="222222222_0000000001", dlv_key=2, dlv_amount=10000000),
    ]
    d = _write(tmp_path, rows)
    with Dataset(_config(d), _audit()) as ds:
        assert pool_size(ds, _audit()) == 1
        batch = screen_batch(ds, _audit(), "h", 0)
        assert [c.item_id for c in batch] == ["22222222"]


def test_non_comparable_categories_filtered(tmp_path: Path) -> None:
    rows = [
        _base_row(item_id="11111111", listing_id="111111111_0000000001", dlv_key=1, category_l_name="토목/건축/자재"),
        _base_row(
            item_id="22222222",
            listing_id="222222222_0000000001",
            dlv_key=2,
            category_l_name="전자/정보/통신/영상",
        ),
    ]
    d = _write(tmp_path, rows)
    with Dataset(_config(d), _audit()) as ds:
        assert pool_size(ds, _audit()) == 1


def test_generic_and_missing_manufacturers_filtered(tmp_path: Path) -> None:
    rows = [
        _base_row(item_id="11111111", listing_id="111111111_0000000001", dlv_key=1, manufacturer="조합공통품목"),
        _base_row(item_id="22222222", listing_id="222222222_0000000001", dlv_key=2, manufacturer=""),
        _base_row(item_id="33333333", listing_id="333333333_0000000001", dlv_key=3, manufacturer="Epson"),
    ]
    d = _write(tmp_path, rows)
    with Dataset(_config(d), _audit()) as ds:
        batch = screen_batch(ds, _audit(), "h", 0)
        assert [c.item_id for c in batch] == ["33333333"]


def test_short_or_non_code_models_filtered(tmp_path: Path) -> None:
    cases = [
        ("11111111", "S657"),
        ("22222222", "50mm"),
        ("33333333", "Windows 11 Pro"),
        ("44444444", "(CN)27BR400"),
        ("55555555", "EB-PU2216B"),
        ("66666666", "12345678"),
        ("77777777", "ABCDEFGH"),
        ("88888888", None),
    ]
    rows = [
        _base_row(item_id=item, listing_id=f"{item[:8]}111_0000000001", dlv_key=i + 1, model=model)
        for i, (item, model) in enumerate(cases)
    ]
    d = _write(tmp_path, rows)
    with Dataset(_config(d), _audit()) as ds:
        batch = screen_batch(ds, _audit(), "h", 0)
        assert {c.item_id for c in batch} == {"44444444", "55555555"}


def test_manufacturer_direct_filtered_when_resellers_required(tmp_path: Path) -> None:
    rows = [_base_row(vendor_name="Epson", manufacturer="Epson")]
    d = _write(tmp_path, rows)
    with Dataset(_config(d), _audit(require_reseller=True)) as ds:
        assert pool_size(ds, _audit(require_reseller=True)) == 0
    with Dataset(_config(d), _audit(require_reseller=False)) as ds:
        assert pool_size(ds, _audit(require_reseller=False)) == 1


def test_same_contract_cheaper_never_benchmark(tmp_path: Path) -> None:
    rows = [
        _base_row(
            listing_id="111111111_0000000001",
            item_id="11111111",
            price=218000,
            dlv_key=1,
            dlv_qty=10,
            dlv_amount=1890000,
            unit="개",
            dlv_unit="개",
        ),
        _base_row(
            listing_id="111111111_0000000002",
            item_id="11111111",
            price=132950,
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            unit="개",
        ),
    ]
    d = _write(tmp_path, rows)
    audit = _audit(min_item_order_amount_krw=0)
    with Dataset(_config(d), audit) as ds:
        (cand,) = screen_batch(ds, audit, "h", 0)
        assert cand.benchmark is None
        assert cand.internal_gap_krw is None


def test_other_contract_cheaper_becomes_benchmark(tmp_path: Path) -> None:
    rows = [
        _base_row(
            listing_id="111111111_0000000001",
            item_id="11111111",
            price=2736000,
            dlv_key=1,
            dlv_qty=10,
            dlv_amount=27360000,
            contract_begin="2025-01-01",
            contract_end="2026-12-31",
        ),
        _base_row(
            listing_id="222222222_0000000001",
            item_id="11111111",
            price=2508000,
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            contract_begin="2025-01-01",
            contract_end="2026-12-31",
            vendor_name="다른업체",
        ),
    ]
    d = _write(tmp_path, rows)
    audit = _audit(markup_threshold=Decimal("1.05"))
    with Dataset(_config(d), audit) as ds:
        (cand,) = screen_batch(ds, audit, "h", 0)
        assert cand.benchmark is not None
        assert cand.benchmark.listing_id == "222222222_0000000001"
        assert cand.benchmark.conditions_verified is False
        assert cand.internal_gap_krw == 2280000


def test_gap_below_threshold_ignored(tmp_path: Path) -> None:
    rows = [
        _base_row(
            listing_id="111111111_0000000001",
            price=2736000,
            dlv_key=1,
            dlv_qty=10,
            dlv_amount=27360000,
        ),
        _base_row(
            listing_id="222222222_0000000001",
            price=2508000,
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            vendor_name="다른업체",
        ),
    ]
    d = _write(tmp_path, rows)
    audit = _audit(markup_threshold=Decimal("1.2"))
    with Dataset(_config(d), audit) as ds:
        (cand,) = screen_batch(ds, audit, "h", 0)
        assert cand.benchmark is None


def test_benchmark_contract_must_cover_window(tmp_path: Path) -> None:
    rows = [
        _base_row(listing_id="111111111_0000000001", price=2736000, dlv_key=1, dlv_qty=10, dlv_amount=27360000),
        _base_row(
            listing_id="222222222_0000000001",
            price=100,
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            contract_begin="2025-01-01",
            contract_end="2025-10-11",
            vendor_name="다른업체",
        ),
    ]
    d = _write(tmp_path, rows)
    with Dataset(_config(d), _audit(markup_threshold=Decimal("1.05"))) as ds:
        (cand,) = screen_batch(ds, _audit(markup_threshold=Decimal("1.05")), "h", 0)
        assert cand.benchmark is None


def test_different_unit_never_compared(tmp_path: Path) -> None:
    rows = [
        _base_row(listing_id="111111111_0000000001", price=2736000, dlv_key=1, dlv_qty=10, dlv_amount=27360000),
        _base_row(
            listing_id="222222222_0000000001",
            price=100,
            unit="조",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            vendor_name="다른업체",
        ),
    ]
    d = _write(tmp_path, rows)
    with Dataset(_config(d), _audit(markup_threshold=Decimal("1.05"))) as ds:
        (cand,) = screen_batch(ds, _audit(markup_threshold=Decimal("1.05")), "h", 0)
        assert cand.benchmark is None


def test_mixed_unit_carries_no_unit_price(tmp_path: Path) -> None:
    rows = [
        _base_row(
            listing_id="111111111_0000000001", dlv_key=1, unit="대", dlv_unit="대", dlv_amount=30000000, dlv_qty=10
        ),
        _base_row(
            listing_id="222222222_0000000001", dlv_key=2, unit="조", dlv_unit="조", dlv_amount=20000000, dlv_qty=5
        ),
    ]
    d = _write(tmp_path, rows)
    with Dataset(_config(d), _audit()) as ds:
        (cand,) = screen_batch(ds, _audit(), "h", 0)
        assert cand.order_qty is None
        assert cand.avg_order_unit_price_krw is None
        assert cand.benchmark is None


def _bench_candidate() -> Candidate:
    return _candidate(
        internal_gap_krw=2280000,
        benchmark=Benchmark(
            listing_id="B1",
            contract_no="cB",
            vendor_name="v",
            unit_price_krw=Decimal("2508000"),
            contract_begin=None,
            contract_end=None,
        ),
    )


def test_matching_delivery_conditions_verify() -> None:
    cand = _bench_candidate()
    raw = _FakeRaw(
        {
            "B1": _listing("B1", Decimal("2508000"), "현장설치도", "서울"),
            "111111111_0000000001": _listing("111111111_0000000001", Decimal("2736000"), "현장설치도", "서울"),
        }
    )
    out = verify_benchmark_conditions(cand, raw, _audit())  # type: ignore[arg-type]
    assert out.benchmark is not None
    assert out.benchmark.conditions_verified is True
    assert out.benchmark.condition_note is None
    assert out.internal_gap_krw == cand.internal_gap_krw


def test_dropoff_vs_installed_rejected() -> None:
    cand = _bench_candidate()
    raw = _FakeRaw(
        {
            "B1": _listing("B1", Decimal("2508000"), "납품장소 하차도", "서울"),
            "111111111_0000000001": _listing("111111111_0000000001", Decimal("2736000"), "현장설치도", "서울"),
        }
    )
    out = verify_benchmark_conditions(cand, raw, _audit())  # type: ignore[arg-type]
    assert out.benchmark is not None
    assert out.benchmark.conditions_verified is False
    assert out.benchmark.condition_note == "delivery_terms_mismatch"


def test_region_mismatch_rejected() -> None:
    cand = _bench_candidate()
    raw = _FakeRaw(
        {
            "B1": _listing("B1", Decimal("2508000"), "현장설치도", "서울"),
            "111111111_0000000001": _listing("111111111_0000000001", Decimal("2736000"), "현장설치도", "부산"),
        }
    )
    out = verify_benchmark_conditions(cand, raw, _audit())  # type: ignore[arg-type]
    assert out.benchmark is not None
    assert out.benchmark.condition_note == "supply_region_mismatch"


def test_unserved_listing_blocks() -> None:
    cand = _bench_candidate()
    raw = _FakeRaw({"111111111_0000000001": _listing("111111111_0000000001", Decimal("1"), "현장설치도", "서울")})
    out = verify_benchmark_conditions(cand, raw, _audit())  # type: ignore[arg-type]
    assert out.benchmark is not None
    assert out.benchmark.condition_note == "listing_not_served"


def test_missing_condition_text_blocks() -> None:
    cand = _bench_candidate()
    raw = _FakeRaw(
        {
            "B1": _listing("B1", Decimal("2508000"), None, "서울"),
            "111111111_0000000001": _listing("111111111_0000000001", Decimal("1"), "현장설치도", "서울"),
        }
    )
    out = verify_benchmark_conditions(cand, raw, _audit())  # type: ignore[arg-type]
    assert out.benchmark is not None
    assert out.benchmark.condition_note == "conditions_missing"


def test_catalog_reference_not_benchmark() -> None:
    cand = _bench_candidate()
    raw = _FakeRaw(
        {
            "B1": _listing("B1", Decimal("2508000"), "현장설치도", "서울", kind="catalog_reference"),
            "111111111_0000000001": _listing("111111111_0000000001", Decimal("1"), "현장설치도", "서울"),
        }
    )
    out = verify_benchmark_conditions(cand, raw, _audit())  # type: ignore[arg-type]
    assert out.benchmark is not None
    assert out.benchmark.condition_note == "benchmark_is_catalog_reference"


def test_stale_benchmark_price_rejected() -> None:
    cand = _bench_candidate()
    raw = _FakeRaw(
        {
            "B1": _listing("B1", Decimal("999"), "현장설치도", "서울"),
            "111111111_0000000001": _listing("111111111_0000000001", Decimal("1"), "현장설치도", "서울"),
        }
    )
    out = verify_benchmark_conditions(cand, raw, _audit())  # type: ignore[arg-type]
    assert out.benchmark is not None
    assert out.benchmark.condition_note == "benchmark_price_changed"


def test_too_many_order_listings_skip_network() -> None:
    cand = _candidate(
        order_listing_ids=tuple(f"L{i}" for i in range(9)),
        benchmark=Benchmark(
            listing_id="B1",
            contract_no="cB",
            vendor_name="v",
            unit_price_krw=Decimal("1"),
            contract_begin=None,
            contract_end=None,
        ),
    )
    raw = _FakeRaw({})
    out = verify_benchmark_conditions(cand, raw, _audit(max_condition_listings=8))  # type: ignore[arg-type]
    assert out.benchmark is not None
    assert out.benchmark.condition_note == "too_many_order_listings"
    assert raw.calls == []


def test_candidate_without_benchmark_no_request() -> None:
    cand = _candidate(benchmark=None)
    raw = _FakeRaw({})
    out = verify_benchmark_conditions(cand, raw, _audit())  # type: ignore[arg-type]
    assert out == cand
    assert raw.calls == []


def test_availability_attached_in_order() -> None:
    cands = [_candidate(item_id="1"), _candidate(item_id="2"), _candidate(item_id="3")]
    comp = _FakeComp({"1": ItemState.CLAIMED, "2": ItemState.AVAILABLE, "3": ItemState.REFORM_FOUND})
    out = annotate_availability(cands, comp)  # type: ignore[arg-type]
    assert [c.availability for c in out] == [ItemState.CLAIMED, ItemState.AVAILABLE, ItemState.REFORM_FOUND]
    assert annotate_availability([], comp) == []  # type: ignore[arg-type]
    assert comp.calls
    assert comp.calls[0] == ["1", "2", "3"]


def test_output_preserves_exact_amounts(tmp_path: Path) -> None:
    cand = _candidate(name="프린터 한글이름", avg_order_unit_price_krw=Decimal("189000.123456"), order_qty=None)
    cand = Candidate(
        **{**cand.__dict__, "order_qty": None, "avg_order_unit_price_krw": Decimal("189000.123456"), "benchmark": None}
    )
    path = tmp_path / "out.jsonl"
    write_candidates([cand], path)
    text = path.read_text(encoding="utf-8")
    assert '"189000.123456"' in text
    assert "프린터 한글이름" in text
    assert "null" in text
    assert text.strip().count("\n") == 0
    raw = json.loads(text.strip())
    assert raw["avg_order_unit_price_krw"] == "189000.123456"


def test_empty_batch_writes_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "empty.jsonl"
    write_candidates([], path)
    assert path.exists()
    assert path.stat().st_size == 0


def test_whitespace_conditions_match_after_strip() -> None:
    cand = _bench_candidate()
    raw = _FakeRaw(
        {
            "B1": _listing("B1", Decimal("2508000"), "현장설치도", "서울"),
            "111111111_0000000001": _listing("111111111_0000000001", Decimal("1"), "  현장설치도  ", "  서울 "),
        }
    )
    out = verify_benchmark_conditions(cand, raw, _audit())  # type: ignore[arg-type]
    assert out.benchmark is not None
    assert out.benchmark.conditions_verified is True


def test_gaps_sum_deduplicated_keys_with_cheapest_valid_benchmark(tmp_path: Path) -> None:
    rows = [
        _base_row(
            listing_id=f"{contract}_0000000001", dlv_key=key, dlv_amount=30000000, dlv_first_date=day, dlv_last_date=day
        )
        for contract, key, day in [
            ("111111111", 1, "2025-01-15"),
            ("333333333", 2, "2025-06-15"),
            ("555555555", 3, "2025-11-15"),
        ]
    ]
    rows.append({**rows[0], "listing_id": "111111111_0000000002"})
    for listing_id, price, begin, end in [
        ("222222222_0000000002", 1000000, "2025-01-01", "2026-01-01"),
        ("222222222_0000000001", 1000000, "2025-01-01", "2026-01-01"),
        ("444444444_0000000001", 500000, "2025-06-01", "2025-07-01"),
    ]:
        rows.append(
            _base_row(
                listing_id=listing_id,
                price=price,
                contract_begin=begin,
                contract_end=end,
                dlv_key=None,
                dlv_lines=None,
                dlv_qty=None,
                dlv_amount=None,
            )
        )
    with Dataset(_config(_write(tmp_path, rows)), _audit()) as ds:
        (candidate,) = screen_batch(ds, _audit(), "h", 0)
        assert candidate.order_amount_krw == 90000000
        assert candidate.key_count == 3
        assert candidate.internal_gap_krw == 65000000
        assert candidate.benchmark is not None
        assert candidate.benchmark.listing_id == "444444444_0000000001"


def test_unit_price_and_anomaly_ignore_ambient_decimal_precision(tmp_path: Path) -> None:
    rows = [
        _base_row(dlv_amount=12000001, dlv_qty=10),
        _base_row(
            listing_id="222222222_0000000001",
            price=1000000,
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_amount=None,
        ),
    ]
    with Dataset(_config(_write(tmp_path, rows)), _audit()) as ds, localcontext() as context:
        context.prec = 3
        (candidate,) = screen_batch(ds, _audit(), "h", 0)
        assert candidate.avg_order_unit_price_krw == Decimal("1200000.1")
        assert candidate.internal_gap_krw == 2000001
        assert context.prec == 3


@pytest.mark.parametrize(
    ("ids", "terms", "region"),
    [
        ((), "현장설치도", "서울"),
        (("111111111_0000000001",), " ", "서울"),
        (("111111111_0000000001",), "현장설치도", " "),
    ],
)
def test_empty_evidence_cannot_verify(ids: tuple[str, ...], terms: str, region: str) -> None:
    from dataclasses import replace

    candidate = replace(_bench_candidate(), order_listing_ids=ids)
    raw = _FakeRaw(
        {
            "B1": _listing("B1", Decimal("2508000"), terms, region),
            "111111111_0000000001": _listing("111111111_0000000001", Decimal("1")),
        }
    )
    result = verify_benchmark_conditions(candidate, raw, _audit())  # type: ignore[arg-type]
    assert result.benchmark is not None
    assert result.benchmark.conditions_verified is False
    assert result.benchmark.condition_note == "conditions_missing"


def test_atomic_output_failure_preserves_existing_file(tmp_path: Path, monkeypatch: Any) -> None:
    import src.savetax.screening as screening

    path = tmp_path / "batch.jsonl"
    path.write_text("existing\n", encoding="utf-8")

    def fail_replace(*_args: Any) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(screening.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_candidates([_candidate()], path)
    assert path.read_text(encoding="utf-8") == "existing\n"
    assert list(tmp_path.iterdir()) == [path]


def test_benchmark_price_ties_choose_smallest_listing_id(tmp_path: Path) -> None:
    rows = [_base_row(dlv_amount=30000000)]
    rows.extend(
        _base_row(listing_id=listing_id, price=1000000, dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_amount=None)
        for listing_id in ("222222222_0000000002", "222222222_0000000001")
    )
    with Dataset(_config(_write(tmp_path, rows)), _audit()) as ds:
        (candidate,) = screen_batch(ds, _audit(), "h", 0)
    assert candidate.benchmark is not None
    assert candidate.benchmark.listing_id == "222222222_0000000001"
    assert candidate.internal_gap_krw == 20000000


def test_availability_deduplicates_requests_preserving_candidate_order() -> None:
    candidates = [_candidate(item_id="2"), _candidate(item_id="1"), _candidate(item_id="2")]
    competition = _FakeComp({"1": ItemState.CLAIMED, "2": ItemState.AVAILABLE})
    result = annotate_availability(candidates, competition)  # type: ignore[arg-type]
    assert competition.calls == [["2", "1"]]
    assert [candidate.availability for candidate in result] == [
        ItemState.AVAILABLE,
        ItemState.CLAIMED,
        ItemState.AVAILABLE,
    ]
