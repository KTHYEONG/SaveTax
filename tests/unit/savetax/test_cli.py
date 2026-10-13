"""Invariant guards for the screen command."""

from __future__ import annotations

import json
import runpy
import socket
import sys
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


@pytest.fixture(autouse=True)
def isolate_retry_sleep(monkeypatch: Any) -> None:
    monkeypatch.setattr("src.savetax.cli.time.sleep", lambda _: None)


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


def _write_dataset(tmp_path: Path, rows: list[dict[str, Any]], dirname: str = "savetax-products-test") -> Path:
    data_dir = tmp_path / dirname
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet = data_dir / "listings.parquet"
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


def _me_body(handle: str = "h1") -> dict:
    return {
        "handle": handle,
        "registered": True,
        "claims_used": 0,
        "claims_today": 0,
        "remaining": {"claims_total": 100, "claims_per_day": 50},
    }


def _lookup_body(ids: list[str], state: str = "available") -> dict:
    return {"items": [{"input": i, "kind": "item", "item_id": i, "state": state} for i in ids]}


def _raw_data(listing_id: str, terms: str = "현장설치도", region: str = "서울", price: str = "2508000") -> dict:
    return {
        "data": {
            "id": listing_id,
            "itemId": "11111111",
            "unit": "대",
            "model": "EB-PU2216B",
            "price": price,
            "priceKind": "contract",
            "manufacturer": "Epson",
            "category": "프린터",
            "deliveryTerms": terms,
            "supplyRegion": region,
            "vendor": {"name": "v"},
            "contract": {"no": "c1", "beginYmd": "20250101", "endYmd": "20261231"},
        },
        "meta": {"release": "r20261011T012137"},
    }


def test_offline_writes_one_file_per_batch(tmp_path: Path) -> None:
    rows = [
        _base_row(item_id=f"{10000000 + i}", listing_id=f"11111111{i}_0000000001", dlv_key=i + 1) for i in range(25)
    ]
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "out"
    calls: list = []

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        raise AssertionError("offline must not call network")

    code = main(
        [
            "screen",
            "--batch",
            "0",
            "--batches",
            "2",
            "--batch-size",
            "10",
            "--handle",
            "h",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
        ],
        environ={},
        transport=transport,  # type: ignore[arg-type]
    )
    assert code == 0
    assert (out_dir / "batch-0000.jsonl").exists()
    assert (out_dir / "batch-0001.jsonl").exists()
    assert calls == []


def test_offline_never_loads_token(tmp_path: Path) -> None:
    rows = [_base_row()]
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "out"
    code = main(
        ["screen", "--batch", "0", "--handle", "h", "--data-dir", str(data_dir), "--out-dir", str(out_dir)],
        environ={},
        transport=None,
    )
    assert code == 0


def test_offline_without_handle_usage_error(tmp_path: Path) -> None:
    rows = [_base_row()]
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "out2"
    code = main(
        ["screen", "--batch", "0", "--data-dir", str(data_dir), "--out-dir", str(out_dir)],
        environ={},
        transport=None,
    )
    assert code == 2
    assert not (out_dir / "batch-0000.jsonl").exists()


def test_stops_at_first_empty_batch(tmp_path: Path) -> None:
    rows = [_base_row(item_id=f"{10000000 + i}", listing_id=f"11111111{i}_0000000001", dlv_key=i + 1) for i in range(5)]
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "out"
    code = main(
        [
            "screen",
            "--batch",
            "0",
            "--batches",
            "4",
            "--handle",
            "h",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
        ],
        environ={},
        transport=None,
    )
    assert code == 0
    assert (out_dir / "batch-0000.jsonl").exists()
    assert not (out_dir / "batch-0001.jsonl").exists()


def _online_transport_for_pool(calls: list, lookup_state: str = "available") -> Any:
    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body("h1"))
        if "/items/lookup" in url:
            query = url.split("ids=")[1]
            ids = query.split(",")
            return _resp(200, _lookup_body(ids, lookup_state))
        if "/listings/" in url:
            lid = url.rsplit("/", 1)[-1]
            price = "2508000" if lid.startswith("222222222") else "2736000"
            return _resp(200, _raw_data(lid, price=price))
        raise AssertionError(f"unexpected url {url}")

    return transport


def test_online_resolves_handle_from_api(tmp_path: Path) -> None:
    rows = [_base_row(item_id=f"{10000000 + i}", listing_id=f"11111111{i}_0000000001", dlv_key=i + 1) for i in range(3)]
    data_dir = _write_dataset(tmp_path, rows)
    out_online = tmp_path / "online"
    out_offline = tmp_path / "offline"
    calls: list = []
    code = main(
        ["screen", "--batch", "0", "--data-dir", str(data_dir), "--out-dir", str(out_online), "--online"],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_online_transport_for_pool(calls),
    )
    assert code == 0
    code2 = main(
        ["screen", "--batch", "0", "--handle", "h1", "--data-dir", str(data_dir), "--out-dir", str(out_offline)],
        environ={},
        transport=None,
    )
    assert code2 == 0
    online_ids = [
        json.loads(line)["item_id"]
        for line in (out_online / "batch-0000.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    offline_ids = [
        json.loads(line)["item_id"]
        for line in (out_offline / "batch-0000.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert online_ids == offline_ids


def test_online_verifies_only_available(tmp_path: Path) -> None:
    rows = [
        _base_row(item_id="11111111", listing_id="111111111_0000000001", dlv_key=1, dlv_amount=27360000),
        _base_row(
            listing_id="222222222_0000000001",
            item_id="11111111",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            price=1000000,
            vendor_name="다른업체",
        ),
        _base_row(item_id="22222222", listing_id="333333333_0000000001", dlv_key=2, dlv_amount=20000000),
        _base_row(
            listing_id="444444444_0000000001",
            item_id="22222222",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            price=100,
            vendor_name="다른업체",
        ),
    ]
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "out"
    calls: list = []

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body("h"))
        if "/items/lookup" in url:
            ids = url.split("ids=")[1].split(",")
            states = {"11111111": "available", "22222222": "claimed"}
            return _resp(200, {"items": [{"input": i, "kind": "item", "item_id": i, "state": states[i]} for i in ids]})
        if "/listings/" in url:
            lid = url.rsplit("/", 1)[-1]
            if lid == "222222222_0000000001":
                return _resp(200, _raw_data(lid, price="1000000"))
            if lid == "111111111_0000000001":
                return _resp(200, _raw_data(lid, price="2736000"))
            raise AssertionError(f"should not fetch {lid}")
        raise AssertionError(url)

    code = main(
        [
            "screen",
            "--batch",
            "0",
            "--handle",
            "h",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--online",
            "--batch-size",
            "10",
        ],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=transport,  # type: ignore[arg-type]
    )
    assert code == 0
    listing_calls = [u for u in calls if "/listings/" in u]
    assert any("222222222_0000000001" in u for u in listing_calls)
    assert not any("444444444_0000000001" in u for u in listing_calls)


def test_per_candidate_api_error_contained(tmp_path: Path, caplog: Any) -> None:
    rows = [
        _base_row(item_id="11111111", listing_id="111111111_0000000001", dlv_key=1, dlv_amount=27360000),
        _base_row(
            listing_id="222222222_0000000001",
            item_id="11111111",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            price=1000000,
            vendor_name="다른업체",
        ),
    ]
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "out"

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body("h"))
        if "/items/lookup" in url:
            return _resp(200, _lookup_body(["11111111"], "available"))
        if url.endswith("222222222_0000000001"):
            return _resp(403, {"error": {"code": "forbidden", "message": "no"}})
        if "/listings/" in url:
            lid = url.rsplit("/", 1)[-1]
            return _resp(200, _raw_data(lid, price="2736000" if lid.startswith("111111111") else "1000000"))
        raise AssertionError(url)

    code = main(
        ["screen", "--batch", "0", "--handle", "h", "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--online"],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=transport,  # type: ignore[arg-type]
    )
    assert code == 0
    lines = (out_dir / "batch-0000.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    assert payload["benchmark"]["condition_note"] == "api_error:forbidden"


def test_network_failure_pauses_with_partial_output(tmp_path: Path, caplog: Any) -> None:
    import logging

    rows = [_base_row()]
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "out"

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        if "/items/lookup" in url:
            raise socket.gaierror("dns fail")
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body("h"))
        raise socket.gaierror("dns fail")

    with caplog.at_level(logging.WARNING):
        code = main(
            [
                "screen",
                "--batch",
                "0",
                "--handle",
                "h",
                "--data-dir",
                str(data_dir),
                "--out-dir",
                str(out_dir),
                "--online",
            ],
            environ={"SAVETAX_TOKEN": TOKEN},
            transport=transport,  # type: ignore[arg-type]
        )
    assert code == 3
    assert (out_dir / "batch-0000.jsonl").exists()
    payload = json.loads((out_dir / "batch-0000.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert payload["availability"] is None
    assert "자료 접근 대기" in caplog.text


def test_missing_dataset_pauses(tmp_path: Path, caplog: Any) -> None:
    import logging

    missing = tmp_path / "savetax-products-test"
    out_dir = tmp_path / "out"
    with caplog.at_level(logging.WARNING):
        code = main(
            ["screen", "--batch", "0", "--handle", "h", "--data-dir", str(missing), "--out-dir", str(out_dir)],
            environ={},
            transport=None,
        )
    assert code == 3
    assert "자료 접근 대기" in caplog.text


def test_missing_token_online_credential_error(tmp_path: Path, caplog: Any, monkeypatch: Any) -> None:
    import logging

    rows = [_base_row()]
    data_dir = _write_dataset(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.ERROR):
        code = main(
            ["screen", "--batch", "0", "--data-dir", str(data_dir), "--out-dir", str(tmp_path / "out"), "--online"],
            environ={},
            transport=None,
        )
    assert code == 2
    assert "savetax_" not in caplog.text


def test_malformed_arguments_return_2() -> None:
    assert main(["screen", "--batch", "x"], environ={}, transport=None) == 2
    assert main(["frobnicate"], environ={}, transport=None) == 2


def test_module_guard_exits_with_main_code(tmp_path: Path, monkeypatch: Any) -> None:
    rows = [_base_row()]
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        ["cli", "screen", "--batch", "0", "--handle", "h", "--data-dir", str(data_dir), "--out-dir", str(out_dir)],
    )
    monkeypatch.delitem(sys.modules, "src.savetax.cli")
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("src.savetax.cli", run_name="__main__")
    assert exc.value.code == 0


def test_no_command_and_bad_batches_return_2() -> None:
    assert main([], environ={}, transport=None) == 2
    assert main(["screen", "--batch", "0", "--batches", "0"], environ={}, transport=None) == 2


def test_invalid_batch_size_returns_2(tmp_path: Path) -> None:
    rows = [_base_row()]
    data_dir = _write_dataset(tmp_path, rows)
    code = main(
        [
            "screen",
            "--batch",
            "0",
            "--batch-size",
            "0",
            "--handle",
            "h",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        environ={},
        transport=None,
    )
    assert code == 2


def test_invalid_dataset_dir_returns_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "listings.parquet").write_bytes(b"dummy")
    code = main(
        ["screen", "--batch", "0", "--handle", "h", "--data-dir", str(bad), "--out-dir", str(tmp_path / "out")],
        environ={},
        transport=None,
    )
    assert code == 2
    code2 = main(
        [
            "screen",
            "--batch",
            "0",
            "--handle",
            "h",
            "--data-dir",
            str(bad),
            "--out-dir",
            str(tmp_path / "out2"),
            "--online",
        ],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_online_transport_for_pool([]),
    )
    assert code2 == 2


def test_negative_batch_returns_2(tmp_path: Path) -> None:
    rows = [_base_row()]
    data_dir = _write_dataset(tmp_path, rows)
    code = main(
        ["screen", "--batch", "-1", "--handle", "h", "--data-dir", str(data_dir), "--out-dir", str(tmp_path / "out")],
        environ={},
        transport=None,
    )
    assert code == 2
    code2 = main(
        [
            "screen",
            "--batch",
            "-1",
            "--handle",
            "h",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(tmp_path / "out2"),
            "--online",
        ],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_online_transport_for_pool([]),
    )
    assert code2 == 2


def test_screen_batch_pause_offline_and_online(tmp_path: Path, monkeypatch: Any) -> None:
    import src.savetax.cli as cli_mod
    from src.savetax.errors import DataAccessPending

    rows = [_base_row()]
    data_dir = _write_dataset(tmp_path, rows)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise DataAccessPending("[DATA] gone: 자료 접근 대기")

    monkeypatch.setattr(cli_mod, "screen_batch", boom)
    code = main(
        ["screen", "--batch", "0", "--handle", "h", "--data-dir", str(data_dir), "--out-dir", str(tmp_path / "o1")],
        environ={},
        transport=None,
    )
    assert code == 3
    code2 = main(
        [
            "screen",
            "--batch",
            "0",
            "--handle",
            "h",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(tmp_path / "o2"),
            "--online",
        ],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_online_transport_for_pool([]),
    )
    assert code2 == 3


def test_get_me_network_and_api_errors(tmp_path: Path) -> None:
    rows = [_base_row()]
    data_dir = _write_dataset(tmp_path, rows)

    def net_transport(url: str, headers: object, timeout: float) -> HttpResponse:
        raise socket.gaierror("dns fail")

    code = main(
        ["screen", "--batch", "0", "--data-dir", str(data_dir), "--out-dir", str(tmp_path / "o1"), "--online"],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=net_transport,  # type: ignore[arg-type]
    )
    assert code == 3

    def api_transport(url: str, headers: object, timeout: float) -> HttpResponse:
        if url.endswith("/events/current/me"):
            return _resp(403, {"code": "AUTH_REQUIRED", "message": "m", "request_id": "r", "retryable": False})
        raise AssertionError(url)

    code2 = main(
        ["screen", "--batch", "0", "--data-dir", str(data_dir), "--out-dir", str(tmp_path / "o2"), "--online"],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=api_transport,  # type: ignore[arg-type]
    )
    assert code2 == 2


def test_online_missing_dataset_pauses(tmp_path: Path) -> None:
    missing = tmp_path / "savetax-products-test"
    code = main(
        [
            "screen",
            "--batch",
            "0",
            "--handle",
            "h",
            "--data-dir",
            str(missing),
            "--out-dir",
            str(tmp_path / "out"),
            "--online",
        ],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_online_transport_for_pool([]),
    )
    assert code == 3


def test_online_empty_second_batch_breaks(tmp_path: Path) -> None:
    rows = [_base_row(item_id=f"{10000000 + i}", listing_id=f"11111111{i}_0000000001", dlv_key=i + 1) for i in range(5)]
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "out"
    code = main(
        [
            "screen",
            "--batch",
            "0",
            "--batches",
            "2",
            "--handle",
            "h",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--online",
            "--batch-size",
            "10",
        ],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_online_transport_for_pool([]),
    )
    assert code == 0
    assert (out_dir / "batch-0000.jsonl").exists()
    assert not (out_dir / "batch-0001.jsonl").exists()


def test_lookup_api_error_returns_2(tmp_path: Path) -> None:
    rows = [_base_row()]
    data_dir = _write_dataset(tmp_path, rows)

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body("h"))
        if "/items/lookup" in url:
            return _resp(403, {"code": "FORBIDDEN", "message": "m", "request_id": "r", "retryable": False})
        raise AssertionError(url)

    code = main(
        [
            "screen",
            "--batch",
            "0",
            "--handle",
            "h",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--online",
        ],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=transport,  # type: ignore[arg-type]
    )
    assert code == 2


def test_verify_network_error_pauses(tmp_path: Path) -> None:
    rows = [
        _base_row(item_id="11111111", listing_id="111111111_0000000001", dlv_key=1, dlv_amount=27360000),
        _base_row(
            listing_id="222222222_0000000001",
            item_id="11111111",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            price=1000000,
            vendor_name="다른업체",
        ),
    ]
    data_dir = _write_dataset(tmp_path, rows)

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body("h"))
        if "/items/lookup" in url:
            return _resp(200, _lookup_body(["11111111"], "available"))
        if "/listings/" in url:
            raise socket.gaierror("dns fail")
        raise AssertionError(url)

    code = main(
        [
            "screen",
            "--batch",
            "0",
            "--handle",
            "h",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--online",
        ],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=transport,  # type: ignore[arg-type]
    )
    assert code == 3


def test_verify_malformed_listing_returns_2(tmp_path: Path) -> None:
    rows = [
        _base_row(item_id="11111111", listing_id="111111111_0000000001", dlv_key=1, dlv_amount=27360000),
        _base_row(
            listing_id="222222222_0000000001",
            item_id="11111111",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
            price=1000000,
            vendor_name="다른업체",
        ),
    ]
    data_dir = _write_dataset(tmp_path, rows)

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body("h"))
        if "/items/lookup" in url:
            return _resp(200, _lookup_body(["11111111"], "available"))
        if "/listings/" in url:
            return HttpResponse(status=200, headers={}, body=b"not json", elapsed_seconds=0.01)
        raise AssertionError(url)

    code = main(
        [
            "screen",
            "--batch",
            "0",
            "--handle",
            "h",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--online",
        ],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=transport,  # type: ignore[arg-type]
    )
    assert code == 2


def test_progress_log_every_batch(tmp_path: Path, caplog: Any) -> None:
    import logging

    rows = []
    for i in range(55):
        item = f"{20000000 + i}"
        rows.append(
            _base_row(item_id=item, listing_id=f"111111{i:02d}_0000000001", dlv_key=2 * i + 1, dlv_amount=27360000)
        )
        rows.append(
            _base_row(
                listing_id=f"222222{i:02d}_0000000001",
                item_id=item,
                dlv_key=None,
                dlv_lines=None,
                dlv_qty=None,
                dlv_unit=None,
                dlv_amount=None,
                dlv_first_date=None,
                dlv_last_date=None,
                price=1000000,
                vendor_name="다른업체",
            )
        )
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "out"

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body("h"))
        if "/items/lookup" in url:
            ids = url.split("ids=")[1].split(",")
            return _resp(200, _lookup_body(ids, "available"))
        if "/listings/" in url:
            lid = url.rsplit("/", 1)[-1]
            if lid.startswith("222222"):
                return _resp(200, _raw_data(lid, price="1000000"))
            return _resp(200, _raw_data(lid, price="2736000"))
        raise AssertionError(url)

    with caplog.at_level(logging.INFO):
        code = main(
            [
                "screen",
                "--batch",
                "0",
                "--handle",
                "h",
                "--data-dir",
                str(data_dir),
                "--out-dir",
                str(out_dir),
                "--online",
                "--batch-size",
                "60",
            ],
            environ={"SAVETAX_TOKEN": TOKEN},
            transport=transport,  # type: ignore[arg-type]
        )
    assert code == 0
    assert "[AUDIT] verified=50 batch=0" in caplog.text


@pytest.mark.parametrize(
    "arguments",
    [
        ["screen", "--batch", "savetax_private-token"],
        ["screen", "--batch", "0", "--unknown", "savetax_private-token"],
        ["savetax_private-token"],
    ],
)
def test_argument_errors_redact_credentials(arguments: list[str], capsys: Any) -> None:
    assert main(arguments, environ={}) == 2
    assert "savetax_private-token" not in capsys.readouterr().err


def test_token_shaped_handle_is_refused_before_online_requests(tmp_path: Path) -> None:
    calls: list = []
    assert (
        main(
            ["screen", "--batch", "0", "--handle", TOKEN, "--online"],
            environ={"SAVETAX_TOKEN": TOKEN},
            transport=_online_transport_for_pool(calls),
        )
        == 2
    )
    assert calls == []


@pytest.mark.parametrize("failure", ["api", "network", "pending"])
def test_verification_failure_preserves_progress_and_stops_correctly(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: Any,
    failure: str,
) -> None:
    import src.savetax.cli as cli
    from src.savetax.config import AuditConfig
    from src.savetax.errors import DataAccessPending

    rows = []
    for i in range(3):
        item_id = str(11111111 + i)
        rows.extend(
            [
                _base_row(
                    item_id=item_id,
                    listing_id=f"11111111{i}_0000000001",
                    dlv_key=i + 1,
                    dlv_amount=30000000 - i * 1000000,
                ),
                _base_row(
                    item_id=item_id,
                    listing_id=f"22222222{i}_0000000001",
                    price=1000000,
                    dlv_key=None,
                    dlv_lines=None,
                    dlv_qty=None,
                    dlv_amount=None,
                ),
            ]
        )
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "output"
    calls: list[str] = []

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        if "/items/lookup" in url:
            return _resp(200, _lookup_body(url.split("ids=")[1].split(",")))
        if url.endswith("222222221_0000000001"):
            if failure == "api":
                return _resp(403, {"error": {"code": "forbidden", "message": "denied"}})
            raise socket.gaierror("DNS unavailable")
        listing_id = url.rsplit("/", 1)[-1]
        return _resp(200, _raw_data(listing_id, price="1000000"))

    original_verify = cli.verify_benchmark_conditions

    def verify(candidate: Any, raw: Any, audit: Any) -> Any:
        if failure == "pending" and candidate.item_id == "11111112":
            raise DataAccessPending("[DATA] evidence unavailable: 자료 접근 대기")
        return original_verify(candidate, raw, audit)

    monkeypatch.setattr(cli, "verify_benchmark_conditions", verify)
    monkeypatch.setattr(cli, "AuditConfig", lambda: AuditConfig(progress_every=1))
    with caplog.at_level("INFO"):
        code = main(
            [
                "screen",
                "--batch",
                "0",
                "--batches",
                "2",
                "--handle",
                "h",
                "--online",
                "--data-dir",
                str(data_dir),
                "--out-dir",
                str(out_dir),
            ],
            environ={"SAVETAX_TOKEN": TOKEN},
            transport=transport,  # type: ignore[arg-type]
        )
    payloads = [json.loads(line) for line in (out_dir / "batch-0000.jsonl").read_text().splitlines()]
    assert payloads[0]["benchmark"]["conditions_verified"] is True
    assert payloads[1]["benchmark"]["conditions_verified"] is False
    if failure == "api":
        assert code == 0
        assert payloads[1]["benchmark"]["condition_note"] == "api_error:forbidden"
        assert payloads[2]["benchmark"]["conditions_verified"] is True
        assert "[AUDIT] verified=2 batch=0" in caplog.text
        assert "[EVIDENCE] item_id=11111112" in caplog.text
    else:
        assert code == 3
        assert payloads[1]["benchmark"]["condition_note"] is None
        assert payloads[2]["benchmark"]["conditions_verified"] is False
        assert not any("222222222_0000000001" in url for url in calls)
        assert "자료 접근 대기" in caplog.text
    assert not (out_dir / "batch-0001.jsonl").exists()


def _evidence_rows() -> list[dict[str, Any]]:
    return [
        _base_row(
            listing_id="111111111_0000000001",
            item_id="11111111",
            price=2760000,
            dlv_key=1,
            dlv_lines=1,
            dlv_qty=2,
            dlv_unit="대",
            dlv_amount=5520000,
        ),
        _base_row(
            listing_id="222222222_0000000001",
            item_id="11111111",
            price=2508000,
            contract_begin="2025-01-01",
            contract_end="2026-12-31",
            vendor_name="다른업체",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
        ),
    ]


def _evidence_fact(state: str = "available") -> dict:
    return {
        "fact_id": "F1",
        "listing_id": "111111111_0000000001",
        "contract_no": "111111111_0",
        "request_no": "R1",
        "line_no": "1",
        "request_date": "2025-10-12",
        "unit": "대",
        "tax_basis": "vat_included",
        "basis_id": "basis:대:vat_included",
        "period_kind": "none",
        "unit_price_krw": "2760000",
        "quantity": "2",
        "comparable_amount_krw": "5520000",
        "scorable": True,
        "exclusion_reasons": [],
    }


def _evidence_item_body(item_id: str = "11111111", state: str = "available") -> dict:
    return {
        "item_id": item_id,
        "availability": {"state": state},
        "snapshot": {
            "name": "물탱크",
            "model": "T",
            "manufacturer": "M",
            "category": "C",
            "source_release": "rtest",
            "facts_total": 1,
            "facts_truncated": False,
        },
        "facts": [_evidence_fact()],
    }


def _evidence_listing_body(listing_id: str, contract_no: str, price: str, terms: str = "서울", region: str = "서울") -> dict:
    return {
        "data": {
            "id": listing_id,
            "itemId": "11111111",
            "unit": "대",
            "price": price,
            "priceKind": "contract",
            "deliveryTerms": terms,
            "supplyRegion": region,
            "vendor": {"name": "v"},
            "contract": {"no": contract_no, "beginYmd": "20250101", "endYmd": "20261231"},
        },
        "meta": {"release": "rtest"},
    }


_PAGE_HTML = (
    "<html><body>단위 대 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
    "계약 단가 2,508,000원 / 대 공급 업체 벤더 납품요구 없음 계약 기간 P "
    "납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음</body></html>"
)


def _evidence_transport(calls: list, *, item_state: str = "available", fail: str | None = None) -> Any:
    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body("h"))
        if url.endswith("/events/current"):
            return _resp(
                200,
                {
                    "dataset_release_id": "release-live-20261012",
                    "ruleset_version": "rules-v1",
                    "phase": "open",
                    "opens_at": "2026-10-01T00:00:00Z",
                    "closes_at": "2026-11-01T00:00:00Z",
                    "evidence_format": "v2",
                    "item_catalog": "catalog",
                },
            )
        if "/items/lookup" in url:
            ids = url.split("ids=")[1].split(",")
            if fail == "lookup_network":
                raise OSError("dns fail")
            return _resp(200, {"items": [{"input": i, "kind": "item", "item_id": i, "state": item_state} for i in ids]})
        if "/events/current/items/" in url and "/lookup" not in url:
            item_id = url.rsplit("/", 1)[-1]
            if fail == "item_api" and item_id == "11111111":
                return _resp(403, {"code": "FORBIDDEN", "message": "m", "request_id": "r", "retryable": False})
            if fail == "item_network":
                raise OSError("dns fail")
            return _resp(200, _evidence_item_body(item_id, item_state))
        if "/listings/111111111_0000000001" in url:
            return _resp(200, _evidence_listing_body("111111111_0000000001", "111111111_0", "2760000"))
        if "/listings/222222222_0000000001" in url:
            return _resp(200, _evidence_listing_body("222222222_0000000001", "222222222_0", "2508000"))
        if "savetax.now/listing/" in url:
            if fail == "page_403":
                return HttpResponse(status=403, headers={}, body=b"", elapsed_seconds=0.01)
            return HttpResponse(status=200, headers={}, body=_PAGE_HTML.encode(), elapsed_seconds=0.01)
        raise AssertionError(f"unexpected url {url}")

    return transport


def test_evidence_writes_bundles_and_drafts(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, _evidence_rows())
    out_dir = tmp_path / "ev"
    calls: list = []
    code = main(
        ["evidence", "--out-dir", str(out_dir), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_evidence_transport(calls),
    )
    assert code == 0
    lines = (out_dir / "cross-bundles.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    draft = json.loads((out_dir / "drafts" / "11111111.json").read_text(encoding="utf-8"))
    assert draft["canonical_product_id"] == "11111111"


def test_evidence_claimed_skipped_before_detail(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, _evidence_rows())
    calls: list = []
    code = main(
        ["evidence", "--out-dir", str(tmp_path / "ev"), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_evidence_transport(calls, item_state="claimed"),
    )
    assert code == 0
    assert not any("/events/current/items/11111111" in u for u in calls if "/lookup" not in u)
    assert not (tmp_path / "ev" / "drafts" / "11111111.json").exists()


def test_evidence_limit_bounds_detail_calls(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for i in range(3):
        item = f"1111111{i}"
        rows.append(_base_row(item_id=item, listing_id=f"11111111{i}_0000000001", dlv_key=10 + i, dlv_amount=5520000, dlv_qty=2, price=2760000))
        rows.append(
            _base_row(
                listing_id=f"22222222{i}_0000000001",
                item_id=item,
                price=2508000,
                vendor_name="다른업체",
                dlv_key=None,
                dlv_lines=None,
                dlv_qty=None,
                dlv_unit=None,
                dlv_amount=None,
                dlv_first_date=None,
                dlv_last_date=None,
            )
        )
    data_dir = _write_dataset(tmp_path, rows)
    calls: list = []

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        if url.endswith("/events/current"):
            return _resp(
                200,
                {
                    "dataset_release_id": "r",
                    "ruleset_version": "v",
                    "phase": "open",
                    "opens_at": "2026-10-01T00:00:00Z",
                    "closes_at": "2026-11-01T00:00:00Z",
                    "evidence_format": "v2",
                    "item_catalog": "c",
                },
            )
        if "/items/lookup" in url:
            ids = url.split("ids=")[1].split(",")
            return _resp(200, {"items": [{"input": i, "kind": "item", "item_id": i, "state": "available"} for i in ids]})
        if "/events/current/items/" in url:
            item_id = url.rsplit("/", 1)[-1]
            body = _evidence_item_body(item_id)
            body["facts"][0]["contract_no"] = "111111111"
            return _resp(200, body)
        if "/listings/" in url:
            lid = url.rsplit("/", 1)[-1]
            if lid.startswith("11111111"):
                return _resp(200, {
                    "data": {"id": lid, "itemId": lid[:8] if False else "x", "unit": "대", "price": "2760000",
                             "priceKind": "contract", "deliveryTerms": "서울", "supplyRegion": "서울",
                             "vendor": {"name": "v"}, "contract": {"no": "111111111", "beginYmd": "20250101", "endYmd": "20261231"}},
                    "meta": {"release": "rtest"},
                })
            return _resp(200, {
                "data": {"id": lid, "itemId": "x", "unit": "대", "price": "2508000",
                         "priceKind": "contract", "deliveryTerms": "서울", "supplyRegion": "서울",
                         "vendor": {"name": "v"}, "contract": {"no": "222222222", "beginYmd": "20250101", "endYmd": "20261231"}},
                "meta": {"release": "rtest"},
            })
        if "savetax.now/listing/" in url:
            return HttpResponse(status=200, headers={}, body=_PAGE_HTML.encode(), elapsed_seconds=0.01)
        raise AssertionError(url)

    code = main(
        ["evidence", "--limit", "1", "--out-dir", str(tmp_path / "ev"), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=transport,  # type: ignore[arg-type]
    )
    assert code == 0
    item_calls = [u for u in calls if "/events/current/items/" in u and "/lookup" not in u]
    assert len(item_calls) == 1


def test_evidence_items_filter_restricts_run(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, _evidence_rows())
    calls: list = []
    code = main(
        ["evidence", "--items", "11111111", "--out-dir", str(tmp_path / "ev"), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_evidence_transport(calls),
    )
    assert code == 0
    assert any("ids=11111111" in u for u in calls if "/lookup" in u)


def test_evidence_api_error_on_one_item_continues(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, _evidence_rows())
    calls: list = []
    code = main(
        ["evidence", "--out-dir", str(tmp_path / "ev"), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_evidence_transport(calls, fail="item_api"),
    )
    assert code == 0


def test_evidence_network_failure_pauses_with_partial_output(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, _evidence_rows())
    calls: list = []
    code = main(
        ["evidence", "--out-dir", str(tmp_path / "ev"), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_evidence_transport(calls, fail="item_network"),
    )
    assert code == 3
    assert (tmp_path / "ev" / "cross-bundles.jsonl").exists()


def test_evidence_invalid_limit_rejected(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, _evidence_rows())
    assert (
        main(
            ["evidence", "--limit", "0", "--out-dir", str(tmp_path / "ev"), "--data-dir", str(data_dir)],
            environ={"SAVETAX_TOKEN": TOKEN},
            transport=_evidence_transport([]),
        )
        == 2
    )


def _retail_danawa_transport(calls: list, headers_seen: list) -> Any:
    fallback_calls: list = []

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        if isinstance(headers, dict):
            headers_seen.append((url, dict(headers)))
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body("h"))
        if url.endswith("/events/current"):
            return _resp(
                200,
                {
                    "dataset_release_id": "release-live-20261012",
                    "ruleset_version": "rules-v1",
                    "phase": "open",
                    "opens_at": "2026-10-01T00:00:00Z",
                    "closes_at": "2026-11-01T00:00:00Z",
                    "evidence_format": "v2",
                    "item_catalog": "catalog",
                },
            )
        if "/items/lookup" in url:
            ids = url.split("ids=")[1].split(",")
            return _resp(200, {"items": [{"input": i, "kind": "item", "item_id": i, "state": "available"} for i in ids]})
        if "/events/current/items/" in url and "/lookup" not in url:
            body = _evidence_item_body("11111111")
            body["snapshot"]["model"] = "EB-PU2216B"
            body["snapshot"]["manufacturer"] = "Epson"
            return _resp(200, body)
        if "/listings/111111111_0000000001" in url:
            return _resp(200, _evidence_listing_body("111111111_0000000001", "111111111_0", "2760000"))
        if "search.danawa.com" in url:
            html = (
                '<li data-product-seq="77"><p class="mb-2 x">EB-PU2216B 프로젝터</p>'
                "<span>1,500,000 원</span></li>"
            )
            return HttpResponse(status=200, headers={}, body=html.encode(), elapsed_seconds=0.01)
        if "getProductPriceList" in url:
            payload = json.dumps({"24": {"result": [{"date": "25-10", "minPrice": 1500000}]}, "12": {"result": []}})
            return HttpResponse(status=200, headers={}, body=payload.encode(), elapsed_seconds=0.01)
        if "prod.danawa.com/info" in url:
            html = (
                '<script type="application/ld+json">{"@type": "Product", "name": "EB-PU2216B", '
                '"brand": {"name": "Sony"}, "description": "레이저 프로젝터", '
                '"offers": {"lowPrice": 1500000, "offerCount": 2}}</script>'
                '<ul class="list__mall-price">'
                '<li><span aria-label="몰A"></span><span class="text__num">1,500,000</span><span>무료배송</span></li>'
                '<li><span aria-label="몰B"></span><span class="text__num">1,600,000</span><span>무료배송</span></li>'
                "</ul><span>등록월: 2025-03</span>"
            )
            return HttpResponse(status=200, headers={}, body=html.encode(), elapsed_seconds=0.01)
        if "savetax.now/listing/" in url:
            return HttpResponse(status=200, headers={}, body=_PAGE_HTML.encode(), elapsed_seconds=0.01)
        fallback_calls.append(url)
        raise AssertionError(f"unexpected url {url}")

    return transport


def test_retail_lane_writes_review_bundles_without_drafts(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, [_base_row()])
    out_dir = tmp_path / "ev"
    calls: list = []
    seen: list = []
    code = main(
        ["evidence", "--lane", "retail", "--out-dir", str(out_dir), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_retail_danawa_transport(calls, seen),
    )
    assert code == 0
    lines = (out_dir / "retail-bundles.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["review_required"] is True
    assert not (out_dir / "drafts" / "11111111.json").exists()


def test_retail_lane_include_review_drafts_them(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, [_base_row()])
    out_dir = tmp_path / "ev"
    calls: list = []
    seen: list = []
    code = main(
        ["evidence", "--lane", "retail", "--out-dir", str(out_dir), "--data-dir", str(data_dir), "--include-review"],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_retail_danawa_transport(calls, seen),
    )
    assert code == 0
    assert (out_dir / "drafts" / "11111111.json").exists()


def test_retail_public_requests_carry_no_credential(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, [_base_row()])
    calls: list = []
    seen: list = []
    code = main(
        ["evidence", "--lane", "retail", "--out-dir", str(tmp_path / "ev"), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_retail_danawa_transport(calls, seen),
    )
    assert code == 0
    public = [(u, h) for u, h in seen if "danawa" in u or "shopexample" in u]
    assert public
    assert all("Authorization" not in h for _, h in public)


def test_retail_items_filter_and_empty_bundle(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, [_base_row()])
    calls: list = []
    seen: list = []
    code = main(
        ["evidence", "--lane", "retail", "--items", "11111111", "--out-dir", str(tmp_path / "ev"), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_retail_danawa_transport(calls, seen),
    )
    assert code == 0
    assert any("ids=11111111" in u for u in calls if "/lookup" in u)

    def empty_transport(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body("h"))
        if url.endswith("/events/current"):
            return _resp(
                200,
                {
                    "dataset_release_id": "r",
                    "ruleset_version": "v",
                    "phase": "open",
                    "opens_at": "2026-10-01T00:00:00Z",
                    "closes_at": "2026-11-01T00:00:00Z",
                    "evidence_format": "v2",
                    "item_catalog": "c",
                },
            )
        if "/items/lookup" in url:
            return _resp(200, _lookup_body(["11111111"]))
        if "/events/current/items/" in url:
            return _resp(200, _evidence_item_body("11111111"))
        if "/listings/" in url:
            return _resp(200, _evidence_listing_body(url.rsplit("/", 1)[-1], "111111111_0", "2760000"))
        return HttpResponse(status=404, headers={}, body=b"", elapsed_seconds=0.01)

    out_dir = tmp_path / "ev-empty"
    code2 = main(
        ["evidence", "--lane", "retail", "--out-dir", str(out_dir), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=empty_transport,  # type: ignore[arg-type]
    )
    assert code2 == 0
    assert (out_dir / "retail-bundles.jsonl").exists()


def test_retail_item_api_error_continues(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, [_base_row()])
    calls: list = []
    seen: list = []
    base = _retail_danawa_transport(calls, seen)

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        if "/events/current/items/11111111" in url and "/lookup" not in url:
            return _resp(403, {"code": "FORBIDDEN", "message": "m", "request_id": "r", "retryable": False})
        return base(url, headers, timeout)

    code = main(
        ["evidence", "--lane", "retail", "--out-dir", str(tmp_path / "ev"), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=transport,  # type: ignore[arg-type]
    )
    assert code == 0


def test_evidence_bad_batch_returns_2(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, [_base_row()])
    assert (
        main(
            ["evidence", "--lane", "retail", "--batches", "0", "--out-dir", str(tmp_path / "ev"), "--data-dir", str(data_dir)],
            environ={"SAVETAX_TOKEN": TOKEN},
            transport=_evidence_transport([]),
        )
        == 2
    )


def test_evidence_only_get_requests(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, _evidence_rows())
    calls: list = []
    code = main(
        ["evidence", "--out-dir", str(tmp_path / "ev"), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_evidence_transport(calls),
    )
    assert code == 0
    assert not any("submissions" in u or "reports" in u for u in calls)


def test_evidence_no_candidates_writes_empty(tmp_path: Path) -> None:
    rows = [
        _base_row(
            listing_id="111111111_0000000001",
            dlv_key=None,
            dlv_lines=None,
            dlv_qty=None,
            dlv_unit=None,
            dlv_amount=None,
            dlv_first_date=None,
            dlv_last_date=None,
        )
    ]
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "ev"
    code = main(
        ["evidence", "--out-dir", str(out_dir), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_evidence_transport([]),
    )
    assert code == 0
    assert (out_dir / "cross-bundles.jsonl").exists()


def test_evidence_review_bundle_needs_flag(tmp_path: Path, monkeypatch: Any) -> None:
    import src.savetax.cli as cli_mod
    from src.savetax.evidence import EvidenceBundle, Lane

    data_dir = _write_dataset(tmp_path, _evidence_rows())

    def fake_bundle(*_a: Any, **_k: Any) -> EvidenceBundle:
        import datetime as _dt
        from datetime import date as _date
        from decimal import Decimal as _Decimal

        from src.savetax.evidence import LineProposal, PriceEvidence

        proof = PriceEvidence(
            source_url="https://savetax.now/listing/222222222_0000000001",
            quote="물품식별번호 11111111 / savetax.now 등재 화면, 2026-10-13 확인",
            observed_price_krw=2508000,
            observed_at=_date(2026, 10, 13),
            vat_included=True,
            shipping_included=True,
        )
        line = LineProposal(
            fact_id="F1",
            benchmark_unit_price_krw=_Decimal("2508000"),
            comparison_basis_id="basis:대:vat_included",
            unit="대",
            tax_basis="vat_included",
            fact_unit_price_krw=_Decimal("2760000"),
            quantity=_Decimal("2"),
            comparable_amount_krw=_Decimal("5520000"),
            request_date=_date(2025, 10, 12),
            expected_krw=504000,
            price_evidence=(proof,),
        )
        return EvidenceBundle(
            item_id="11111111",
            lane=Lane.CROSS_CONTRACT,
            title="같은 물품의 다른 계약 단가 대비 1.10배",
            claim="청구",
            comparison_scope="범위",
            reproduction_notes="재현",
            lines=(line,),
            rejected=(),
            caveats=(),
            review_required=True,
            generated_at=_dt.datetime(2026, 10, 13, tzinfo=_dt.UTC),
        )

    monkeypatch.setattr(cli_mod, "build_cross_bundle", fake_bundle)
    out_plain = tmp_path / "ev-plain"
    code = main(
        ["evidence", "--out-dir", str(out_plain), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_evidence_transport([]),
    )
    assert code == 0
    assert not (out_plain / "drafts" / "11111111.json").exists()
    out_flag = tmp_path / "ev-flag"
    code2 = main(
        ["evidence", "--out-dir", str(out_flag), "--data-dir", str(data_dir), "--include-review"],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_evidence_transport([]),
    )
    assert code2 == 0
    assert (out_flag / "drafts" / "11111111.json").exists()


@pytest.mark.parametrize("failure", ["api", "network", "pending"])
def test_evidence_preserves_first_bundle_and_handles_second_item(tmp_path: Path, failure: str, caplog: Any) -> None:
    from src.savetax.errors import DataAccessPending

    rows = []
    listing_info = {}
    for i in range(3):
        item = str(11111111 + i)
        order_id = f"11111111{i}_0000000001"
        benchmark_id = f"22222222{i}_0000000001"
        for row, lid in zip(_evidence_rows(), (order_id, benchmark_id), strict=True):
            row.update(item_id=item, listing_id=lid)
            if row["dlv_key"] is not None:
                row["dlv_key"] = i + 1
            rows.append(row)
            listing_info[lid] = (item, str(row["price"]))
    data_dir = _write_dataset(tmp_path, rows)
    calls: list[str] = []
    fallback = _evidence_transport([])

    def transport(url: str, headers: Any, timeout: float) -> HttpResponse:
        calls.append(url)
        if "/items/lookup" in url:
            return _resp(200, _lookup_body(url.split("ids=")[1].split(",")))
        if "/events/current/items/" in url:
            item = url.rsplit("/", 1)[-1]
            if item == "11111112":
                if failure == "api":
                    return _resp(403, {"code": "FORBIDDEN", "message": "denied", "retryable": False})
                if failure == "pending":
                    raise DataAccessPending("[DATA] facts pending: 자료 접근 대기")
                raise socket.gaierror("unreachable")
            body = _evidence_item_body(item)
            lid = next(l for l, (pid, _) in listing_info.items() if pid == item and l.startswith("111"))
            body["facts"][0].update(listing_id=lid, contract_no=lid[:11])
            return _resp(200, body)
        if "/listings/" in url:
            lid = url.rsplit("/", 1)[-1]
            item, price = listing_info[lid]
            body = _evidence_listing_body(lid, lid[:11], price)
            body["data"]["itemId"] = item
            return _resp(200, body)
        if "https://savetax.now/listing/" in url:
            assert "Authorization" not in headers
            lid = url.rsplit("/", 1)[-1]
            item, _ = listing_info[lid]
            html = _PAGE_HTML.replace("11111111", item).replace("222222222_0000000001", lid)
            return _resp(200, html.encode())
        return fallback(url, headers, timeout)

    out = tmp_path / "evidence"
    with caplog.at_level("INFO"):
        code = main(["evidence", "--data-dir", str(data_dir), "--out-dir", str(out)],
                    environ={"SAVETAX_TOKEN": TOKEN}, transport=transport)
    bundles = [json.loads(line) for line in (out / "cross-bundles.jsonl").read_text().splitlines()]
    assert bundles[0]["item_id"] == "11111111"
    assert (out / "drafts" / "11111111.json").is_file()
    assert calls.count("https://hackathon.savetax.now/v1/events/current") == 1
    assert "[EVIDENCE] lane=cross" in caplog.text
    if failure == "api":
        assert code == 0
        assert [b["item_id"] for b in bundles] == ["11111111", "11111113"]
        assert (out / "drafts" / "11111113.json").is_file()
        assert "skipped=api_error:FORBIDDEN" in caplog.text
    else:
        assert code == 3
        assert len(bundles) == 1
        assert not any(u.endswith("/items/11111113") for u in calls)
        assert "자료 접근 대기" in caplog.text


def test_evidence_all_lane_writes_both_bundles(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, _evidence_rows())
    out_dir = tmp_path / "ev"
    code = main(
        ["evidence", "--lane", "all", "--out-dir", str(out_dir), "--data-dir", str(data_dir)],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_evidence_transport([]),
    )
    assert code == 0
    assert (out_dir / "cross-bundles.jsonl").exists()
    assert (out_dir / "retail-bundles.jsonl").exists()


def test_all_lane_drafts_do_not_overwrite_cross_evidence(tmp_path: Path) -> None:
    rows = _evidence_rows()
    rows[0]["dlv_amount"] = 27600000
    rows[0]["dlv_qty"] = 10
    data_dir = _write_dataset(tmp_path, rows)
    out_dir = tmp_path / "out"
    retail = _retail_danawa_transport([], [])
    cross = _evidence_transport([])

    def transport(url: str, headers: Any, timeout: float) -> HttpResponse:
        if url.endswith("/listings/222222222_0000000001"):
            return cross(url, headers, timeout)
        return retail(url, headers, timeout)

    assert main(["evidence", "--lane", "all", "--include-review", "--data-dir", str(data_dir),
                 "--out-dir", str(out_dir)], environ={"SAVETAX_TOKEN": TOKEN}, transport=transport) == 0
    cross_draft = json.loads((out_dir / "drafts" / "11111111.json").read_text())
    retail_draft = json.loads((out_dir / "drafts" / "retail" / "11111111.json").read_text())
    assert "savetax.now/listing/" in cross_draft["lines"][0]["price_evidence"][0]["source_url"]
    assert "prod.danawa.com" in retail_draft["lines"][0]["price_evidence"][0]["source_url"]


@pytest.mark.parametrize("options", [["--limit", "1"], ["--items", "11111113"]])
def test_evidence_selection_from_multiple_real_candidates(tmp_path: Path, options: list[str]) -> None:
    rows = []
    for i in range(3):
        for row, prefix in zip(_evidence_rows(), ("11111111", "22222222"), strict=True):
            row.update(item_id=str(11111111 + i), listing_id=f"{prefix}{i}_0000000001")
            if row["dlv_key"] is not None:
                row["dlv_key"] = i + 1
            rows.append(row)
    data_dir = _write_dataset(tmp_path, rows)
    looked_up: list[str] = []
    detailed: list[str] = []
    fallback = _evidence_transport([])

    def transport(url: str, headers: Any, timeout: float) -> HttpResponse:
        if "/items/lookup" in url:
            ids = url.split("ids=")[1].split(",")
            looked_up.extend(ids)
            return _resp(200, _lookup_body(ids))
        if "/events/current/items/" in url:
            item = url.rsplit("/", 1)[-1]
            detailed.append(item)
            return _resp(200, {"item_id": item, "availability": {"state": "claimed"}, "snapshot": None, "facts": []})
        return fallback(url, headers, timeout)

    assert main(["evidence", "--data-dir", str(data_dir), "--out-dir", str(tmp_path / "out"), *options],
                environ={"SAVETAX_TOKEN": TOKEN}, transport=transport) == 0
    if options[0] == "--items":
        assert looked_up == ["11111113"]
        assert detailed == ["11111113"]
    else:
        assert looked_up == ["11111111", "11111112", "11111113"]
        assert detailed == ["11111111"]


def _conflict_transport(calls: list, *, fact_unit: str = "조", raw_unit: str = "조", page_unit: str = "개") -> Any:
    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body("h"))
        if url.endswith("/events/current"):
            return _resp(
                200,
                {
                    "dataset_release_id": "release-live-20261012",
                    "ruleset_version": "rules-v1",
                    "phase": "open",
                    "opens_at": "2026-10-01T00:00:00Z",
                    "closes_at": "2026-11-01T00:00:00Z",
                    "evidence_format": "v2",
                    "item_catalog": "catalog",
                },
            )
        if "/items/lookup" in url:
            ids = url.split("ids=")[1].split(",")
            return _resp(200, {"items": [{"input": i, "kind": "item", "item_id": i, "state": "available"} for i in ids]})
        if "/events/current/items/" in url and "/lookup" not in url:
            body = _evidence_item_body("11111111")
            body["facts"][0]["unit"] = fact_unit
            return _resp(200, body)
        if "/listings/111111111_0000000001" in url:
            body = _evidence_listing_body("111111111_0000000001", "111111111_0", "2760000")
            body["data"]["unit"] = fact_unit
            return _resp(200, body)
        if "/listings/222222222_0000000001" in url:
            body = _evidence_listing_body("222222222_0000000001", "222222222_0", "2508000")
            body["data"]["unit"] = raw_unit
            return _resp(200, body)
        if "savetax.now/listing/" in url:
            html = (
                f"<html><body>단위 {page_unit} 물품식별번호 11111111 계약품목관리번호 222222222_0000000001 "
                f"계약 단가 2,508,000원 / {page_unit} 공급 업체 벤더 납품요구 없음 계약 기간 P "
                "납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음</body></html>"
            )
            return HttpResponse(status=200, headers={}, body=html.encode(), elapsed_seconds=0.01)
        raise AssertionError(f"unexpected url {url}")

    return transport


def test_unit_conflicts_are_logged(tmp_path: Path, caplog: Any) -> None:
    import logging

    data_dir = _write_dataset(tmp_path, _evidence_rows())
    out_dir = tmp_path / "ev"
    calls: list = []
    with caplog.at_level(logging.INFO):
        code = main(
            ["evidence", "--out-dir", str(out_dir), "--data-dir", str(data_dir)],
            environ={"SAVETAX_TOKEN": TOKEN},
            transport=_conflict_transport(calls),
        )
    assert code == 0
    lines = (out_dir / "unit-conflicts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["fact_unit"] == "조"
    assert payload["display_unit"] == "개"
    assert "unit_conflicts=1" in caplog.text
    assert (out_dir / "unit-conflicts.jsonl").stat().st_mode & 0o777 == 0o600


def test_unit_policy_option_is_wired(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path, _evidence_rows())
    out_dir = tmp_path / "ev"
    code = main(
        ["evidence", "--out-dir", str(out_dir), "--data-dir", str(data_dir), "--unit-policy", "reject"],
        environ={"SAVETAX_TOKEN": TOKEN},
        transport=_conflict_transport([]),
    )
    assert code == 0
    bundles = [json.loads(line) for line in (out_dir / "cross-bundles.jsonl").read_text().splitlines()]
    assert len(bundles) == 1
    assert bundles[0]["lines"] == []
    assert bundles[0]["rejected"] == [{"fact_id": "F1", "reason": "unit_display_conflict"}]
    assert not (out_dir / "drafts" / "11111111.json").exists()
    assert len((out_dir / "unit-conflicts.jsonl").read_text().splitlines()) == 1


def test_malformed_unit_note_cannot_hide_structured_conflicts(tmp_path: Path, monkeypatch: Any) -> None:
    import datetime as _dt
    from datetime import date as _date
    from decimal import Decimal as _Decimal

    import src.savetax.cli as cli
    from src.savetax.evidence import EvidenceBundle, Lane, LineProposal, PriceEvidence

    proof = PriceEvidence(
        source_url="https://savetax.now/listing/1",
        quote="물품식별번호 1 / savetax.now 등재 화면, 2026-10-13 확인",
        observed_price_krw=100,
        observed_at=_date(2026, 10, 13),
        vat_included=True,
        shipping_included=True,
    )
    line = LineProposal(
        fact_id="F1",
        benchmark_unit_price_krw=_Decimal("100"),
        comparison_basis_id="b",
        unit="조",
        tax_basis="vat_included",
        fact_unit_price_krw=_Decimal("200"),
        quantity=_Decimal("1"),
        comparable_amount_krw=_Decimal("200"),
        request_date=_date(2025, 10, 12),
        expected_krw=100,
        price_evidence=(proof,),
    )
    bundle = EvidenceBundle(
        item_id="1",
        lane=Lane.CROSS_CONTRACT,
        title="t",
        claim="c",
        comparison_scope="s",
        reproduction_notes="r",
        lines=(line,),
        rejected=(),
        caveats=(),
        review_required=False,
        generated_at=_dt.datetime(2026, 10, 13, tzinfo=_dt.UTC),
        unit_notes=("malformed note without structure",),
    )
    def build(*args: Any, **kwargs: Any) -> EvidenceBundle:
        kwargs["on_unit_conflict"]({
            "item_id": "1", "listing_id": "1", "fact_unit": " EA ",
            "source_unit": "ea", "display_unit": "box unit",
            "observed_at": "2026-10-13",
        })
        return bundle

    monkeypatch.setattr(cli, "build_cross_bundle", build)
    data_dir = _write_dataset(tmp_path, _evidence_rows())
    out_dir = tmp_path / "out"
    assert main(["evidence", "--data-dir", str(data_dir), "--out-dir", str(out_dir)],
                environ={"SAVETAX_TOKEN": TOKEN}, transport=_evidence_transport([])) == 0
    record = json.loads((out_dir / "unit-conflicts.jsonl").read_text())
    assert record["fact_unit"] == " EA "
    assert record["display_unit"] == "box unit"


@pytest.mark.parametrize("policy", ["annotate", "reject"])
def test_conflict_audit_preserves_whitespace_and_appends(tmp_path: Path, policy: str, caplog: Any) -> None:
    data_dir = _write_dataset(tmp_path, _evidence_rows())
    out = tmp_path / "out"
    args = ["evidence", "--data-dir", str(data_dir), "--out-dir", str(out), "--unit-policy", policy]
    for _ in range(2):
        with caplog.at_level("INFO"):
            assert main(args, environ={"SAVETAX_TOKEN": TOKEN},
                        transport=_conflict_transport([], fact_unit="EA", raw_unit=" EA ", page_unit="box unit")) == 0
        assert (out / "unit-conflicts.jsonl").stat().st_mode & 0o777 == 0o600
        (out / "unit-conflicts.jsonl").chmod(0o644)
    records = [json.loads(line) for line in (out / "unit-conflicts.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["source_unit"] == " EA "
    assert records[0]["display_unit"] == "box unit"
    assert "unit_conflicts=1 items=1" in caplog.text


def test_conflict_audit_survives_later_network_failure(tmp_path: Path, caplog: Any) -> None:
    rows = _evidence_rows()
    rows[0]["dlv_amount"] = 27600000
    rows[0]["dlv_qty"] = 10
    data_dir = _write_dataset(tmp_path, rows)
    out = tmp_path / "out"
    fallback = _conflict_transport([])

    def transport(url: str, headers: Any, timeout: float) -> HttpResponse:
        if "search.danawa.com" in url:
            raise socket.gaierror("unreachable")
        return fallback(url, headers, timeout)

    with caplog.at_level("INFO"):
        assert main(["evidence", "--lane", "all", "--data-dir", str(data_dir), "--out-dir", str(out)],
                    environ={"SAVETAX_TOKEN": TOKEN}, transport=transport) == 3
    assert len((out / "unit-conflicts.jsonl").read_text().splitlines()) == 1
    assert "unit_conflicts=1" in caplog.text


def test_conflict_audit_survives_bundle_threshold(tmp_path: Path, monkeypatch: Any) -> None:
    import src.savetax.cli as cli
    from src.savetax.config import EvidenceConfig

    monkeypatch.setattr(cli, "EvidenceConfig", lambda: EvidenceConfig(min_item_expected_krw=1_000_000))
    rows = _evidence_rows()
    rows[0]["dlv_amount"] = 27600000
    rows[0]["dlv_qty"] = 10
    data_dir = _write_dataset(tmp_path, rows)
    out = tmp_path / "out"
    assert main(["evidence", "--data-dir", str(data_dir), "--out-dir", str(out)],
                environ={"SAVETAX_TOKEN": TOKEN}, transport=_conflict_transport([])) == 0
    assert (out / "cross-bundles.jsonl").read_text() == ""
    assert len((out / "unit-conflicts.jsonl").read_text().splitlines()) == 1
