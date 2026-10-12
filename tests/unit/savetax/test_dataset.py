"""Invariant guards for dataset access and audit views."""

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pytest

from src.savetax.config import AuditConfig, SaveTaxConfig
from src.savetax.dataset import Dataset, REQUIRED_COLUMNS
from src.savetax.errors import DataAccessPending, DataIntegrityError

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
        "listing_id": "232130948_1280000462",
        "item_id": "11111111",
        "name": "품명, 제조사, MODEL-123456",
        "detail_category": "책상",
        "category_l_name": "사무",
        "manufacturer": "제조사",
        "model": "MODEL-123456",
        "unit": "개",
        "price": 218000,
        "contract_begin": "2025-01-01",
        "contract_end": "2026-01-01",
        "vendor_name": "업체",
        "dlv_key": 1,
        "dlv_lines": 1,
        "dlv_qty": 10,
        "dlv_unit": "개",
        "dlv_amount": 1000,
        "dlv_first_date": "2025-10-11",
        "dlv_last_date": "2025-10-12",
    }
    row.update(over)
    return row


def _write(tmp_path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> Path:
    dataset_dir = tmp_path / "savetax-products-test"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    parquet = dataset_dir / "listings.parquet"
    cols = list(columns) if columns is not None else list(_COLUMNS)
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
        proj = ", ".join(cols)
        con.execute(f"COPY (SELECT {proj} FROM tmp) TO '{parquet}' (FORMAT PARQUET)")  # noqa: S608
    finally:
        con.close()
    return dataset_dir


def _config(dataset_dir: Path) -> SaveTaxConfig:
    return SaveTaxConfig(dataset_dir=dataset_dir)


def _audit(**over: Any) -> AuditConfig:
    return AuditConfig(**over)


def test_required_columns_match_spec() -> None:
    assert frozenset(_COLUMNS) == REQUIRED_COLUMNS


def test_shared_order_key_counted_once(tmp_path: Path) -> None:
    rows = [
        _base_row(listing_id="111111111_0000000001", dlv_key=1, dlv_amount=1000),
        _base_row(listing_id="111111111_0000000002", dlv_key=1, dlv_amount=1000),
        _base_row(listing_id="222222222_0000000001", item_id="22222222", dlv_key=2, dlv_amount=500),
    ]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        summary = dataset.summary()
        assert summary.deduped_order_amount_krw == 1500
        assert dataset.connection.execute("SELECT COUNT(dlv_key) FROM order_keys").fetchone()[0] == 2


def test_unordered_listings_stay_null(tmp_path: Path) -> None:
    rows = [_base_row(dlv_key=None, dlv_lines=None, dlv_qty=None, dlv_unit=None, dlv_amount=None)]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        assert dataset.connection.execute("SELECT COUNT(dlv_key) FROM order_keys").fetchone()[0] == 0
        assert dataset.connection.execute("SELECT COUNT(item_id) FROM item_orders").fetchone()[0] == 0
        assert dataset.summary().order_keys == 0


def test_cancelled_key_excluded_but_counted(tmp_path: Path) -> None:
    rows = [_base_row(dlv_lines=0, dlv_amount=0)]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        assert dataset.connection.execute("SELECT COUNT(dlv_key) FROM order_keys").fetchone()[0] == 0
        assert dataset.summary().order_keys == 1


def test_unit_unresolved_key_excluded(tmp_path: Path) -> None:
    rows = [_base_row(dlv_qty=None, dlv_amount=500)]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        assert dataset.connection.execute("SELECT COUNT(dlv_key) FROM order_keys").fetchone()[0] == 0


def test_order_unit_must_equal_listing_unit(tmp_path: Path) -> None:
    rows = [_base_row(unit="개", dlv_unit="식")]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        assert dataset.connection.execute("SELECT COUNT(dlv_key) FROM order_keys").fetchone()[0] == 0


def test_absurd_unit_price_filtered(tmp_path: Path) -> None:
    rows = [_base_row(price=1)]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit(min_unit_price_krw=Decimal("100"))) as dataset:
        assert dataset.connection.execute("SELECT COUNT(dlv_key) FROM order_keys").fetchone()[0] == 0


def test_average_order_price_from_order(tmp_path: Path) -> None:
    rows = [_base_row(price=218000, dlv_amount=1890000, dlv_qty=10)]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        value = dataset.connection.execute("SELECT avg_order_unit_price_krw FROM order_keys").fetchone()[0]
        assert isinstance(value, Decimal)
        assert value == Decimal("189000")


def test_contract_number_both_id_forms(tmp_path: Path) -> None:
    rows = [
        _base_row(listing_id="232130948_1280000462", dlv_key=None, dlv_amount=None, dlv_qty=None, dlv_unit=None),
        _base_row(
            listing_id="R25TA00622517010000003",
            item_id="22222222",
            dlv_key=None,
            dlv_amount=None,
            dlv_qty=None,
            dlv_unit=None,
        ),
    ]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        found = {
            row[0]: row[1]
            for row in dataset.connection.execute("SELECT listing_id, contract_no FROM listings").fetchall()
        }
        assert found["232130948_1280000462"] == "232130948_1"
        assert found["R25TA00622517010000003"] == "R25TA00622517"


def test_item_aggregates_across_contracts(tmp_path: Path) -> None:
    rows = [
        _base_row(listing_id="232130948_1280000462", dlv_key=1, dlv_amount=300, dlv_qty=3),
        _base_row(listing_id="232130949_1280000463", dlv_key=2, dlv_amount=700, dlv_qty=7),
    ]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        row = dataset.connection.execute("SELECT order_amount_krw, order_qty, key_count FROM item_orders").fetchone()
        assert row[0] == 1000
        assert row[1] == 10
        assert row[2] == 2


def test_mixed_units_leave_quantity_null(tmp_path: Path) -> None:
    rows = [
        _base_row(listing_id="111111111_0000000001", dlv_key=1, unit="대", dlv_unit="대", dlv_amount=300, dlv_qty=3),
        _base_row(listing_id="222222222_0000000001", dlv_key=2, unit="조", dlv_unit="조", dlv_amount=700, dlv_qty=7),
    ]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        row = dataset.connection.execute("SELECT unit, order_qty, order_amount_krw FROM item_orders").fetchone()
        assert row[0] is None
        assert row[1] is None
        assert row[2] == 1000


def test_representative_attributes_deterministic(tmp_path: Path) -> None:
    rows = [
        _base_row(listing_id="200000000_0000000002", dlv_key=2, name="두번째"),
        _base_row(listing_id="100000000_0000000001", dlv_key=1, name="첫번째"),
    ]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        first = dataset.connection.execute("SELECT name FROM item_orders").fetchone()[0]
        second = dataset.connection.execute("SELECT name FROM item_orders").fetchone()[0]
        assert first == "첫번째"
        assert second == "첫번째"
        ids = dataset.connection.execute("SELECT order_listing_ids FROM item_orders").fetchone()[0]
        assert list(ids) == sorted(ids)


def test_reseller_ignores_corporate_suffixes(tmp_path: Path) -> None:
    rows = [_base_row(vendor_name="(주)덕인금속", manufacturer="덕인금속주식회사")]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        assert dataset.connection.execute("SELECT reseller FROM item_orders").fetchone()[0] is False


def test_reseller_true_for_distinct_maker(tmp_path: Path) -> None:
    rows = [_base_row(vendor_name="주식회사 트리엠", manufacturer="Epson")]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        assert dataset.connection.execute("SELECT reseller FROM item_orders").fetchone()[0] is True


def test_reseller_false_for_empty_manufacturer(tmp_path: Path) -> None:
    rows = [_base_row(manufacturer="")]
    dataset_dir = _write(tmp_path, rows)
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        assert dataset.connection.execute("SELECT reseller FROM item_orders").fetchone()[0] is False


def test_key_spanning_two_items_fails_closed(tmp_path: Path) -> None:
    rows = [
        _base_row(listing_id="111111111_0000000001", item_id="11111111", dlv_key=1),
        _base_row(listing_id="111111111_0000000002", item_id="22222222", dlv_key=1),
    ]
    dataset_dir = _write(tmp_path, rows)
    with pytest.raises(DataIntegrityError):
        Dataset(_config(dataset_dir), _audit())


def test_inconsistent_key_amount_fails_closed(tmp_path: Path) -> None:
    rows = [
        _base_row(listing_id="111111111_0000000001", dlv_key=1, dlv_amount=100),
        _base_row(listing_id="111111111_0000000002", dlv_key=1, dlv_amount=200),
    ]
    dataset_dir = _write(tmp_path, rows)
    with pytest.raises(DataIntegrityError):
        Dataset(_config(dataset_dir), _audit())


def test_short_listing_id_fails_closed(tmp_path: Path) -> None:
    rows = [_base_row(listing_id="short")]
    dataset_dir = _write(tmp_path, rows)
    with pytest.raises(DataIntegrityError):
        Dataset(_config(dataset_dir), _audit())


def test_missing_column_fails_closed(tmp_path: Path) -> None:
    rows = [_base_row()]
    cols = [c for c in _COLUMNS if c != "dlv_unit"]
    dataset_dir = _write(tmp_path, rows, columns=cols)
    with pytest.raises(DataIntegrityError, match="dlv_unit"):
        Dataset(_config(dataset_dir), _audit())


def test_missing_file_pauses(tmp_path: Path) -> None:
    missing = tmp_path / "savetax-products-test" / "listings.parquet"
    with pytest.raises(DataAccessPending, match="자료 접근 대기"):
        Dataset(_config(tmp_path / "savetax-products-test"), _audit())


def test_empty_file_fails_closed(tmp_path: Path) -> None:
    dataset_dir = _write(tmp_path, [])
    with pytest.raises(DataIntegrityError):
        Dataset(_config(dataset_dir), _audit())


def test_unreadable_file_fails_closed(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "savetax-products-test"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "listings.parquet").write_text("not a parquet file", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        Dataset(_config(dataset_dir), _audit())


def test_handle_released(tmp_path: Path) -> None:
    dataset_dir = _write(tmp_path, [_base_row()])
    dataset = Dataset(_config(dataset_dir), _audit())
    with dataset:
        assert dataset.version == "test"
    with pytest.raises(DataAccessPending):
        _ = dataset.connection
    dataset.close()


@pytest.mark.parametrize("field", ["item_id", "dlv_amount", "listing_id"])
def test_null_key_mapping_fails_closed(tmp_path: Path, field: str) -> None:
    rows = [_base_row(), _base_row(**{field: None})]
    dataset_dir = _write(tmp_path, rows)
    with pytest.raises(DataIntegrityError):
        Dataset(_config(dataset_dir), _audit())


def test_key_spanning_contracts_fails_closed(tmp_path: Path) -> None:
    dataset_dir = _write(tmp_path, [_base_row(), _base_row(listing_id="333333333_0000000001")])
    with pytest.raises(DataIntegrityError, match="contract_no"):
        Dataset(_config(dataset_dir), _audit())


def test_representative_retains_null_attributes(tmp_path: Path) -> None:
    dataset_dir = _write(
        tmp_path,
        [
            _base_row(listing_id="100000000_0000000001", dlv_key=1, manufacturer=None, model=None),
            _base_row(listing_id="200000000_0000000001", dlv_key=2),
        ],
    )
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        assert dataset.connection.execute("SELECT manufacturer, model FROM item_orders").fetchone() == (None, None)


def test_missing_listing_id_names_column(tmp_path: Path) -> None:
    dataset_dir = _write(tmp_path, [_base_row()], [c for c in _COLUMNS if c != "listing_id"])
    with pytest.raises(DataIntegrityError, match="listing_id"):
        Dataset(_config(dataset_dir), _audit())


def test_invalid_numeric_schema_fails_closed(tmp_path: Path) -> None:
    dataset_dir = _write(tmp_path, [_base_row()])
    parquet = dataset_dir / "listings.parquet"
    with duckdb.connect() as conn:
        conn.read_parquet(str(parquet)).create("bad")
        conn.execute("ALTER TABLE bad ALTER COLUMN price TYPE VARCHAR")
        conn.execute("UPDATE bad SET price = 'invalid'")
        conn.table("bad").write_parquet(str(parquet))
    with pytest.raises(DataIntegrityError):
        Dataset(_config(dataset_dir), _audit())


def test_large_order_average_avoids_binary_float(tmp_path: Path) -> None:
    dataset_dir = _write(tmp_path, [_base_row(dlv_amount=9007199254740993, dlv_qty=1)])
    with Dataset(_config(dataset_dir), _audit()) as dataset:
        assert dataset.connection.execute("SELECT avg_order_unit_price_krw FROM order_keys").fetchone()[0] == Decimal(
            "9007199254740993"
        )


def test_exception_in_context_releases_handle(tmp_path: Path) -> None:
    dataset_dir = _write(tmp_path, [_base_row()])
    dataset = Dataset(_config(dataset_dir), _audit())
    with pytest.raises(ValueError, match="body failure"), dataset:
        raise ValueError("body failure")
    with pytest.raises(DataAccessPending):
        _ = dataset.connection


def test_minimum_price_threshold_is_not_rounded(tmp_path: Path) -> None:
    dataset_dir = _write(tmp_path, [_base_row(price=100)])
    with Dataset(_config(dataset_dir), _audit(min_unit_price_krw=Decimal("100.0000001"))) as dataset:
        assert dataset.summary().eligible_order_keys == 0
