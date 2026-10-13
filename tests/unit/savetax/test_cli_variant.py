"""Invariant guards for the variant policy CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from src.savetax.cli import main
from src.savetax.http import HttpResponse

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

TOKEN = "savetax_test-token_1"
_ITEM = "99999999"


def _row(lid: str, price: int, dlv_key: Any = None, amount: Any = None, qty: Any = None) -> dict[str, Any]:
    return {
        "listing_id": lid,
        "item_id": _ITEM,
        "name": "물탱크",
        "detail_category": "물탱크",
        "category_l_name": "기계",
        "manufacturer": "M",
        "model": "T",
        "unit": "대",
        "price": price,
        "contract_begin": "2025-01-01",
        "contract_end": "2026-12-31",
        "vendor_name": "업체",
        "dlv_key": dlv_key,
        "dlv_lines": 1 if dlv_key is not None else None,
        "dlv_qty": qty,
        "dlv_unit": "대" if dlv_key is not None else None,
        "dlv_amount": amount,
        "dlv_first_date": "2025-10-11" if dlv_key is not None else None,
        "dlv_last_date": "2025-10-12" if dlv_key is not None else None,
    }


def _write(tmp_path: Path) -> Path:
    data_dir = tmp_path / "savetax-products-test"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet = data_dir / "listings.parquet"
    rows = [
        _row("111111111_0000000001", 4225000),
        _row("111111111_0000000002", 5200000),
        _row("222222222_0000000001", 2100000),
        _row("222222222_0000000002", 4510000),
        _row("333333333_0000000001", 3000000),
        _row("444444444_0000000001", 5200000, dlv_key=7, amount=10400000, qty=2),
    ]
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
            con.execute(
                "INSERT INTO tmp VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [row.get(c) for c in _COLUMNS],
            )
        proj = ", ".join(_COLUMNS)
        con.execute(f"COPY (SELECT {proj} FROM tmp) TO '{parquet}' (FORMAT PARQUET)")  # noqa: S608
    finally:
        con.close()
    return data_dir


def _resp(status: int, payload: object) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(status=status, headers={}, body=body, elapsed_seconds=0.01)


def _listing_body(lid: str, price: str, contract: str) -> dict:
    return {
        "data": {
            "id": lid,
            "itemId": _ITEM,
            "unit": "대",
            "price": price,
            "priceKind": "contract",
            "deliveryTerms": "서울",
            "supplyRegion": "서울",
            "vendor": {"name": "v"},
            "contract": {"no": contract, "beginYmd": "20250101", "endYmd": "20261231"},
        },
        "meta": {"release": "rtest"},
    }


def _page(lid: str, price: str) -> bytes:
    return (
        f"단위 대 물품식별번호 {_ITEM} 계약품목관리번호 {lid} "
        f"계약 단가 {price}원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P "
        f"납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음"
    ).encode()


def _transport(calls: list, fact_price: str = "5200000") -> Any:
    def run(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        if url.endswith("/events/current/me"):
            return _resp(200, {"handle": "h", "registered": True, "claims_used": 0, "claims_today": 0,
                               "remaining": {"claims_total": 100, "claims_per_day": 50}})
        if url.endswith("/events/current"):
            return _resp(200, {"dataset_release_id": "r", "ruleset_version": "v", "phase": "open",
                               "opens_at": "2026-10-01T00:00:00Z", "closes_at": "2026-11-01T00:00:00Z",
                               "evidence_format": "v2", "item_catalog": "c"})
        if "/items/lookup" in url:
            return _resp(200, {"items": [{"input": _ITEM, "kind": "item", "item_id": _ITEM, "state": "available"}]})
        if "/events/current/items/" in url:
            return _resp(200, {
                "item_id": _ITEM,
                "availability": {"state": "available"},
                "snapshot": {"name": "T", "model": "T", "manufacturer": "M", "category": "C",
                             "source_release": "rtest", "facts_total": 1, "facts_truncated": False},
                "facts": [{
                    "fact_id": "F1", "listing_id": "111111111_0000000002", "contract_no": "111111111_0",
                    "request_no": "R1", "line_no": "1", "request_date": "2025-10-12", "unit": "대",
                    "tax_basis": "vat_included", "basis_id": "basis:대:vat_included", "period_kind": "none",
                    "unit_price_krw": fact_price, "quantity": "2", "comparable_amount_krw": str(int(fact_price) * 2),
                    "scorable": True, "exclusion_reasons": [],
                }],
            })
        if "/listings/111111111_0000000001" in url:
            return _resp(200, _listing_body("111111111_0000000001", "4225000", "111111111_0"))
        if "/listings/111111111_0000000002" in url:
            return _resp(200, _listing_body("111111111_0000000002", "5200000", "111111111_0"))
        if "/listings/222222222_0000000001" in url:
            return _resp(200, _listing_body("222222222_0000000001", "2100000", "222222222_0"))
        if "/listings/222222222_0000000002" in url:
            return _resp(200, _listing_body("222222222_0000000002", "4510000", "222222222_0"))
        if "/listings/333333333_0000000001" in url:
            return _resp(200, _listing_body("333333333_0000000001", "3000000", "333333333_0"))
        if "savetax.now/listing/222222222_0000000002" in url:
            return HttpResponse(status=200, headers={}, body=_page("222222222_0000000002", "4,510,000"), elapsed_seconds=0.01)
        if "savetax.now/listing/222222222_0000000001" in url:
            return HttpResponse(status=200, headers={}, body=_page("222222222_0000000001", "2,100,000"), elapsed_seconds=0.01)
        if "savetax.now/listing/" in url:
            return HttpResponse(status=404, headers={}, body=b"", elapsed_seconds=0.01)
        raise AssertionError(f"unexpected {url}")

    return run


def test_variant_policy_off_cites_cheaper_sibling(tmp_path: Path) -> None:
    data_dir = _write(tmp_path)
    for policy, want in (("off", "2100000"), ("strict", "4510000")):
        out = tmp_path / f"ev-{policy}"
        calls: list = []
        code = main(
            ["evidence", "--out-dir", str(out), "--data-dir", str(data_dir), "--variant-policy", policy],
            environ={"SAVETAX_TOKEN": TOKEN},
            transport=_transport(calls),  # type: ignore[arg-type]
        )
        assert code == 0
        lines = (out / "cross-bundles.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        body = json.loads(lines[0])
        assert body["lines"][0]["benchmark_unit_price_krw"] == want
        draft = json.loads((out / "drafts" / f"{_ITEM}.json").read_text(encoding="utf-8"))
        assert draft["lines"][0]["benchmark_unit_price_krw"] == want


def test_strict_summary_keeps_discovery_filtered_sole_rejection(tmp_path: Path, caplog: Any) -> None:
    data_dir = _write(tmp_path)
    out = tmp_path / "ev-strict-low"
    with caplog.at_level("INFO"):
        code = main(
            ["evidence", "--out-dir", str(out), "--data-dir", str(data_dir),
             "--variant-policy", "strict"],
            environ={"SAVETAX_TOKEN": TOKEN}, transport=_transport([], fact_price="4225000"),
        )
    assert code == 0
    assert "variant_unresolved=1 variant_bearing_items=1 policy=strict" in caplog.text
    body = json.loads((out / "cross-bundles.jsonl").read_text())
    assert body["lines"] == []
    assert body["rejected"] == [{"fact_id": "F1", "reason": "variant_unresolved"}]
    assert not (out / "drafts" / f"{_ITEM}.json").exists()


def test_variant_summary_is_logged(tmp_path: Path, caplog: Any) -> None:
    import logging

    data_dir = tmp_path / "savetax-products-test"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet = data_dir / "listings.parquet"
    rows = [
        _row("111111111_0000000001", 4225000),
        _row("111111111_0000000002", 5200000),
        _row("333333333_0000000001", 3000000),
        _row("444444444_0000000001", 5200000, dlv_key=7, amount=10400000, qty=2),
    ]
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
            con.execute(
                "INSERT INTO tmp VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [row.get(c) for c in _COLUMNS],
            )
        proj = ", ".join(_COLUMNS)
        con.execute(f"COPY (SELECT {proj} FROM tmp) TO '{parquet}' (FORMAT PARQUET)")  # noqa: S608
    finally:
        con.close()
    out = tmp_path / "ev"
    calls: list = []

    with caplog.at_level(logging.INFO):
        code = main(
            ["evidence", "--out-dir", str(out), "--data-dir", str(data_dir),
             "--variant-policy", "allow_low_fact"],
            environ={"SAVETAX_TOKEN": TOKEN},
            transport=_transport(calls, fact_price="5200000"),  # type: ignore[arg-type]
        )
    assert code == 0
    assert "variant_unresolved=1" in caplog.text
    assert "policy=allow_low_fact" in caplog.text


def test_series_is_the_cli_default(tmp_path: Path, caplog: Any) -> None:
    import logging

    data_dir = _write(tmp_path)
    out = tmp_path / "ev-default"
    with caplog.at_level(logging.INFO):
        code = main(
            ["evidence", "--out-dir", str(out), "--data-dir", str(data_dir)],
            environ={"SAVETAX_TOKEN": TOKEN}, transport=_transport([]),
        )
    assert code == 0
    assert "policy=series" in caplog.text


def _owned_series_transport(calls: list) -> Any:
    base = _transport(calls, fact_price="2736000")

    def run(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        if "/items/lookup" in url:
            return _resp(200, {"items": [{"input": _ITEM, "kind": "item", "item_id": _ITEM,
                                          "state": "claimed_by_me"}]})
        if "/events/current/items/" in url:
            return _resp(200, {
                "item_id": _ITEM,
                "availability": {"state": "claimed_by_me"},
                "snapshot": {"name": "T", "model": "T", "manufacturer": "M", "category": "C",
                             "source_release": "rtest", "facts_total": 1, "facts_truncated": False},
                "facts": [{
                    "fact_id": "F1", "listing_id": "331930619_1190000455", "contract_no": "331930619_1",
                    "request_no": "R1", "line_no": "1", "request_date": "2025-10-12", "unit": "대",
                    "tax_basis": "vat_included", "basis_id": "basis:대:vat_included", "period_kind": "none",
                    "unit_price_krw": "2736000", "quantity": "2", "comparable_amount_krw": "5472000",
                    "scorable": True, "exclusion_reasons": [],
                }],
            })
        if "/listings/331930619_1190000234" in url:
            return _resp(200, _listing_body("331930619_1190000234", "2508000", "331930619_1"))
        if "/listings/331930619_1190000455" in url:
            return _resp(200, _listing_body("331930619_1190000455", "2736000", "331930619_1"))
        if "/listings/122010318_1070000300" in url:
            return _resp(200, _listing_body("122010318_1070000300", "2000000", "122010318_1"))
        if "/listings/122010318_1070000700" in url:
            return _resp(200, _listing_body("122010318_1070000700", "2400000", "122010318_1"))
        if "savetax.now/listing/122010318_1070000700" in url:
            return HttpResponse(status=200, headers={}, body=_page("122010318_1070000700", "2,400,000"),
                                elapsed_seconds=0.01)
        if "savetax.now/listing/" in url:
            return HttpResponse(status=404, headers={}, body=b"", elapsed_seconds=0.01)
        return base(url, headers, timeout)

    return run


def _write_owned_series(tmp_path: Path) -> Path:
    data_dir = tmp_path / "savetax-products-owned"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet = data_dir / "listings.parquet"
    rows = [
        _row("331930619_1190000234", 2508000),
        _row("331930619_1190000455", 2736000, dlv_key=7, amount=5472000, qty=2),
        _row("122010318_1070000300", 2000000),
        _row("122010318_1070000700", 2400000),
    ]
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
            con.execute(
                "INSERT INTO tmp VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [row.get(c) for c in _COLUMNS],
            )
        proj = ", ".join(_COLUMNS)
        con.execute(f"COPY (SELECT {proj} FROM tmp) TO '{parquet}' (FORMAT PARQUET)")  # noqa: S608
    finally:
        con.close()
    return data_dir


def test_owned_rebuild_honours_the_series_policy(tmp_path: Path) -> None:
    data_dir = _write_owned_series(tmp_path)
    out = tmp_path / "ev-owned"
    calls: list = []
    code = main(
        ["evidence", "--owned", "--limit", "200", "--out-dir", str(out),
         "--data-dir", str(data_dir), "--variant-policy", "series"],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_owned_series_transport(calls),  # type: ignore[arg-type]
    )
    assert code == 0
    body = json.loads((out / "remediation" / "cross-bundles.jsonl").read_text())
    assert len(body["lines"]) == 1
    quote = body["lines"][0]["price_evidence"][0]["quote"]
    assert "122010318_1070000700" in quote
    assert "122010318_1070000300" not in quote
    assert "122010318_1070000300" not in " ".join(body["caveats"])
    draft = json.loads((out / "remediation" / "drafts" / f"{_ITEM}.json").read_text(encoding="utf-8"))
    assert draft["lines"][0]["benchmark_unit_price_krw"] == "2400000"


def test_series_unresolved_parses_remediation_bundles(tmp_path: Path) -> None:
    from src.savetax.cli import _series_unresolved
    from src.savetax.errors import DataIntegrityError

    assert _series_unresolved(tmp_path) == frozenset()
    rem = tmp_path / "remediation"
    rem.mkdir(parents=True, exist_ok=True)
    (rem / "cross-bundles.jsonl").mkdir(parents=False, exist_ok=True)
    with pytest.raises(DataIntegrityError):
        _series_unresolved(tmp_path)
    (rem / "cross-bundles.jsonl").rmdir()
    (rem / "cross-bundles.jsonl").write_text(
        "\n"
        + json.dumps({"item_id": "A", "lines": [],
                      "rejected": [{"fact_id": "F1", "reason": "variant_unresolved"}]}) + "\n"
        + json.dumps({"item_id": "B", "lines": [{"x": 1}], "rejected": []}) + "\n"
        + json.dumps({"item_id": "C", "lines": [],
                      "rejected": [{"fact_id": "F1", "reason": "no_admissible_benchmark"}]}) + "\n",
        encoding="utf-8",
    )
    assert _series_unresolved(tmp_path) == frozenset({"A"})


@pytest.mark.parametrize("payload", ["{bad json", "[1, 2]", '{"item_id": "A", "lines": []}'])
def test_invalid_remediation_bundles_block_planning(tmp_path: Path, payload: str) -> None:
    from src.savetax.cli import _series_unresolved
    from src.savetax.errors import DataIntegrityError

    rem = tmp_path / "remediation"
    rem.mkdir()
    (rem / "cross-bundles.jsonl").write_text(payload, encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        _series_unresolved(tmp_path)
