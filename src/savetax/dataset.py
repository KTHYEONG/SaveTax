"""Local dataset access over the listings Parquet."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import duckdb

from src.savetax.config import AuditConfig, SaveTaxConfig
from src.savetax.errors import DataAccessPending, DataIntegrityError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
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
    }
)

_PROJECTED_COLUMNS = (
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

__all__ = ["REQUIRED_COLUMNS", "Dataset", "DatasetSummary"]


@dataclass(frozen=True)
class DatasetSummary:
    """Row counts and the deduplicated order total in KRW."""

    listings: int
    items: int
    order_keys: int
    eligible_order_keys: int
    deduped_order_amount_krw: int


def _norm_sql(column: str) -> str:
    inner = f"COALESCE({column}, '')"
    for token in ("(주)", "주식회사", "㈜", "(유)", "유한회사", "(사)"):
        inner = f"replace({inner}, '{token}', '')"
    return f"regexp_replace({inner}, '\\s+', '', 'g')"


_RESELLER_EXPR = (
    f"TRIM(COALESCE(manufacturer, '')) <> '' "
    f"AND {_norm_sql('manufacturer')} <> {_norm_sql('vendor_name')} "
    f"AND NOT contains({_norm_sql('vendor_name')}, {_norm_sql('manufacturer')}) "
    f"AND NOT contains({_norm_sql('manufacturer')}, {_norm_sql('vendor_name')})"
)


class Dataset:
    """Open DuckDB handle over the listings Parquet exposing the canonical relations."""

    def __init__(self, config: SaveTaxConfig, audit: AuditConfig) -> None:
        self._config = config
        self._audit = audit
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._closed = False
        parquet = config.dataset_parquet_path
        if not parquet.is_file():
            raise DataAccessPending(f"[DATA] dataset file missing: 자료 접근 대기 (resume: {parquet})")
        version = config.dataset_version
        self._conn = duckdb.connect()
        try:
            self._build(parquet)
            summary = self.summary()
        except duckdb.Error as exc:
            self.close()
            raise DataIntegrityError(f"[DATA] malformed dataset: {exc}") from None
        except Exception:
            self.close()
            raise
        logger.info(
            "[DATA] dataset version=%s listings=%d items=%d eligible_order_keys=%d deduped_order_amount_krw=%d",
            version,
            summary.listings,
            summary.items,
            summary.eligible_order_keys,
            summary.deduped_order_amount_krw,
        )

    def _quoted_path(self, parquet: Path) -> str:
        return str(parquet).replace("'", "''")

    def _build(self, parquet: Path) -> None:
        conn = self._require_open()
        quoted = self._quoted_path(parquet)
        try:
            names = [row[0] for row in conn.execute(f"SELECT name FROM parquet_schema('{quoted}')").fetchall()]  # noqa: S608
        except Exception as exc:
            raise DataIntegrityError(f"[DATA] unreadable parquet: {exc}") from None
        missing = sorted(REQUIRED_COLUMNS - set(names))
        if missing:
            raise DataIntegrityError(f"[DATA] missing column {missing[0]}")
        total = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{quoted}')").fetchone()  # noqa: S608
        if total is None or int(total[0]) == 0:
            raise DataIntegrityError("[DATA] empty dataset")
        cols = ", ".join(_PROJECTED_COLUMNS)
        conn.execute(
            "CREATE OR REPLACE VIEW listings AS "  # noqa: S608
            f"SELECT {cols}, "  # noqa: S608
            "CASE WHEN starts_with(listing_id, 'R') THEN substr(listing_id, 1, 13) "
            "ELSE substr(listing_id, 1, 11) END AS contract_no "
            f"FROM read_parquet('{quoted}')"  # noqa: S608
        )
        short = conn.execute(
            "SELECT COUNT(*) FROM listings WHERE listing_id IS NULL "
            "OR (starts_with(listing_id, 'R') AND length(listing_id) < 13) "
            "OR (NOT starts_with(listing_id, 'R') AND length(listing_id) < 11)"
        ).fetchone()
        if short is not None and int(short[0]) > 0:
            raise DataIntegrityError("[DATA] malformed listing_id")
        for label, distinct in (
            ("item_id", "item_id"),
            ("contract_no", "contract_no"),
            ("dlv_amount", "dlv_amount"),
        ):
            row = conn.execute(
                "SELECT dlv_key FROM listings WHERE dlv_key IS NOT NULL "  # noqa: S608
                f"GROUP BY dlv_key HAVING COUNT(DISTINCT {distinct}) <> 1 "  # noqa: S608
                f"OR COUNT({distinct}) <> COUNT(*) LIMIT 1"
            ).fetchone()
            if row is not None:
                raise DataIntegrityError(f"[DATA] inconsistent {label} for dlv_key")
        threshold = format(self._audit.min_unit_price_krw, "f")
        conn.execute(
            "CREATE OR REPLACE VIEW order_keys AS "  # noqa: S608
            "SELECT dlv_key, "
            "MIN(item_id) AS item_id, "
            "MIN(contract_no) AS contract_no, "
            "MIN(unit) AS unit, "
            "MAX(dlv_amount) AS order_amount_krw, "
            "MAX(dlv_qty) AS order_qty, "
            "CAST((CAST(MAX(dlv_amount) AS HUGEINT) * 1000000) // MAX(dlv_qty) "
            "AS DECIMAL(38, 0)) * CAST('0.000001' AS DECIMAL(7, 6)) "
            "AS avg_order_unit_price_krw, "
            "MIN(dlv_first_date) AS first_order_date, "
            "MAX(dlv_last_date) AS last_order_date, "
            "MIN(vendor_name) AS vendor_name "
            "FROM listings WHERE dlv_key IS NOT NULL "
            "AND dlv_lines > 0 AND dlv_amount > 0 AND dlv_qty > 0 "
            f"AND dlv_unit = unit AND price >= {threshold} "  # noqa: S608
            "GROUP BY dlv_key"
        )
        conn.execute(
            "CREATE OR REPLACE VIEW item_orders AS "  # noqa: S608
            "WITH eligible AS ("
            "SELECT l.listing_id, l.name, l.detail_category, l.category_l_name, "
            "l.manufacturer, l.model, l.vendor_name, k.item_id "
            "FROM listings l JOIN order_keys k ON l.dlv_key = k.dlv_key"
            "), agg AS ("
            "SELECT item_id, SUM(order_amount_krw) AS order_amount_krw, "
            "COUNT(dlv_key) AS key_count, "
            "CASE WHEN COUNT(DISTINCT unit) = 1 THEN MIN(unit) ELSE NULL END AS unit, "
            "CASE WHEN COUNT(DISTINCT unit) = 1 THEN SUM(order_qty) ELSE NULL END AS order_qty, "
            "MIN(first_order_date) AS first_order_date, "
            "MAX(last_order_date) AS last_order_date "
            "FROM order_keys GROUP BY item_id"
            "), descr AS ("
            "SELECT item_id, "
            "first(name ORDER BY listing_id) AS name, "
            "first(detail_category ORDER BY listing_id) AS detail_category, "
            "first(category_l_name ORDER BY listing_id) AS category_l_name, "
            "first(manufacturer ORDER BY listing_id) AS manufacturer, "
            "first(model ORDER BY listing_id) AS model, "
            "list_sort(list(listing_id)) AS order_listing_ids, "
            f"COALESCE(bool_or({_RESELLER_EXPR}), false) AS reseller "
            "FROM eligible GROUP BY item_id"
            ") SELECT agg.item_id, agg.order_amount_krw, agg.key_count, agg.unit, agg.order_qty, "
            "agg.first_order_date, agg.last_order_date, "
            "descr.name, descr.detail_category, descr.category_l_name, "
            "descr.manufacturer, descr.model, descr.reseller, descr.order_listing_ids "
            "FROM agg JOIN descr ON agg.item_id = descr.item_id"
        )

    def _require_open(self) -> duckdb.DuckDBPyConnection:
        if self._closed or self._conn is None:
            raise DataAccessPending("[DATA] connection closed: 자료 접근 대기")
        return self._conn

    def __enter__(self) -> Dataset:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Return the open DuckDB handle."""
        return self._require_open()

    @property
    def version(self) -> str:
        """Return the dataset release suffix."""
        return self._config.dataset_version

    def summary(self) -> DatasetSummary:
        """Row counts and the deduplicated order total in KRW."""
        conn = self._require_open()
        listings_row = conn.execute("SELECT COUNT(listing_id) FROM listings").fetchone()
        items_row = conn.execute("SELECT COUNT(DISTINCT item_id) FROM listings").fetchone()
        keys_row = conn.execute("SELECT COUNT(DISTINCT dlv_key) FROM listings WHERE dlv_key IS NOT NULL").fetchone()
        eligible_row = conn.execute("SELECT COUNT(dlv_key) FROM order_keys").fetchone()
        row = conn.execute(
            "SELECT COALESCE(SUM(a), 0) FROM "
            "(SELECT MAX(dlv_amount) AS a FROM listings WHERE dlv_key IS NOT NULL GROUP BY dlv_key)"
        ).fetchone()
        assert listings_row is not None
        assert items_row is not None
        assert keys_row is not None
        assert eligible_row is not None
        assert row is not None
        return DatasetSummary(
            listings=int(listings_row[0]),
            items=int(items_row[0]),
            order_keys=int(keys_row[0]),
            eligible_order_keys=int(eligible_row[0]),
            deduped_order_amount_krw=int(row[0]),
        )

    def close(self) -> None:
        """Release the DuckDB handle."""
        if self._closed:
            return
        self._closed = True
        conn, self._conn = self._conn, None
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
