"""CLI guards for owned evidence and remediation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.savetax.cli import main
from src.savetax.http import HttpResponse

TOKEN = "savetax_test-token_1"


def _resp(status: int, payload: object) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(status=status, headers={}, body=body, elapsed_seconds=0.01)


def _event_body(phase: str = "open") -> dict:
    return {"dataset_release_id": "release-1", "ruleset_version": "rules-v1", "phase": phase,
            "opens_at": "2026-10-01T00:00:00Z", "closes_at": "2026-11-01T00:00:00Z",
            "evidence_format": "v2", "item_catalog": "catalog"}


def _me_body() -> dict:
    return {"handle": "h", "registered": True, "claims_used": 0, "claims_today": 0,
            "remaining": {"claims_total": 100, "claims_per_day": 50}}


def _draft_body(item: str = "11111111") -> dict:
    entry = _bundle_line(item)["lines"][0]
    evidence = dict(entry["price_evidence"][0], kind="price_page", observed_price_krw="1000")
    evidence.pop("note", None)
    line = {key: entry[key] for key in ("fact_id", "benchmark_unit_price_krw", "comparison_basis_id", "unit", "tax_basis")}
    line.update(period=None, price_evidence=[evidence])
    return {"dataset_release_id": "release-1", "ruleset_version": "rules-v1",
            "canonical_product_id": item, "title": f"title-{item}", "lines": [line]}


def _bundle_line(item: str = "11111111", total: int = 2000) -> dict:
    return {"item_id": item, "lane": "cross_contract", "title": f"title-{item}", "claim": "c",
            "comparison_scope": "s", "reproduction_notes": "r",
            "lines": [{"fact_id": "F1", "benchmark_unit_price_krw": "1000", "comparison_basis_id": "b",
                       "unit": "대", "tax_basis": "vat_included", "fact_unit_price_krw": "2000",
                       "quantity": "2", "comparable_amount_krw": "4000", "request_date": "2026-10-01",
                       "expected_krw": total,
                       "price_evidence": [{"source_url": "https://savetax.now/listing/x", "quote": "q tag",
                                           "observed_price_krw": 1000, "observed_at": "2026-10-01",
                                           "vat_included": True, "shipping_included": True, "note": None}]}],
            "rejected": [], "caveats": [], "review_required": False,
            "generated_at": "2026-10-13T12:00:00+00:00", "expected_total_krw": total, "unit_notes": []}


def _stage_remediation(tmp_path: Path, items: list[str]) -> None:
    rem = tmp_path / "scratch" / "evidence" / "remediation"
    drafts = rem / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    bundles = []
    for item in items:
        (drafts / f"{item}.json").write_text(json.dumps(_draft_body(item)), encoding="utf-8")
        bundles.append(_bundle_line(item))
    (rem / "cross-bundles.jsonl").write_text("\n".join(json.dumps(b) for b in bundles) + "\n", encoding="utf-8")
    (tmp_path / "scratch" / "evidence" / "retail-bundles.jsonl").write_text("", encoding="utf-8")


def _submission_payload(sid: str, item: str, revision: int = 2, amount: int = 2000) -> dict:
    return {"submission_id": sid, "item_id": item, "submission_state": "accepted",
            "current_revision": revision, "amount_krw": amount, "lint_status": "clean"}


def _item_body(item: str) -> dict:
    return {"item_id": item, "availability": {"state": "claimed_by_me"},
            "snapshot": {"name": "n", "model": "m", "manufacturer": "f", "category": "c",
                         "source_release": "r", "facts_total": 1, "facts_truncated": False},
            "facts": [{"fact_id": "F1", "listing_id": "L1", "contract_no": "C1", "request_no": "R1",
                       "line_no": "1", "request_date": "2026-10-01", "unit": "대",
                       "tax_basis": "vat_included", "basis_id": "b", "period_kind": "none",
                       "unit_price_krw": "2000", "quantity": "2", "comparable_amount_krw": "4000",
                       "scorable": True, "exclusion_reasons": []}]}


_LISTING_PAGE = ("단위 대 물품식별번호 11111111 계약품목관리번호 x "
                  "계약 단가 1,000원 / 대 공급 업체 v 납품요구 없음 계약 기간 P "
                  "납품 조건 서울 납품요구 기록 없음 공급 지역 서울 인증 없음")


def _remediate_transports(calls: list, posts: list, own: list[tuple[str, str]], preview_map: dict | None = None):
    preview_map = preview_map or {}

    def read(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(("GET", url))
        if url.endswith("/events/current"):
            return _resp(200, _event_body("RUNNING"))
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body())
        if url.endswith("/events/current/me/submissions"):
            return _resp(200, {"submissions": [_submission_payload(sid, item) for sid, item in own]})
        if "/events/current/items/" in url and "/lookup" not in url:
            item = url.rsplit("/", 1)[-1]
            return _resp(200, _item_body(item))
        if "/items/lookup" in url:
            ids = url.split("ids=")[1].split(",")
            return _resp(200, {"items": [{"input": i, "kind": "item", "item_id": i,
                                          "state": "claimed_by_me"} for i in ids]})
        if "/submission-requests/" in url:
            return _resp(404, {"code": "IDEMPOTENCY_KEY_NOT_FOUND", "message": "m", "retryable": False})
        if "/events/current/submissions/" in url:
            sid = url.rsplit("/", 1)[-1]
            item = next((it for s, it in own if s == sid), own[0][1] if own else "11111111")
            return _resp(200, _submission_payload(sid, item))
        if "savetax.now/listing" in url:
            return HttpResponse(status=200, headers={}, body=_LISTING_PAGE.encode(), elapsed_seconds=0.01)
        raise AssertionError(f"unexpected GET {url}")

    def write(method: str, url: str, headers: object, body: bytes | None, timeout: float) -> HttpResponse:
        posts.append(url)
        assert url.endswith("/events/current/submissions/preview"), f"read-only plan must only preview, got {url}"
        payload = json.loads(body.decode()) if body else {}
        item = payload.get("canonical_product_id", "11111111")
        total = preview_map.get(item, 2000)
        return _resp(200, {"ok": True, "total_amount_krw": total, "line_amounts_krw": {},
                           "next_action": "revise_existing", "product_already_claimed": False, "claimed_by_me": True,
                           "existing_submission_id": "SID1", "current_revision": 2})

    return read, write


def test_plan_is_read_only(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage_remediation(tmp_path, ["11111111"])
    calls: list = []
    posts: list = []
    read, write = _remediate_transports(calls, posts, [("SID1", "11111111")])
    code = main(["remediate", "plan"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write)  # type: ignore[arg-type]
    assert code == 0
    assert posts
    assert all(p.endswith("/events/current/submissions/preview") for p in posts)
    out = capsys.readouterr().out
    assert "item | sid | action" in out


def test_plan_prints_two_isolated_challenges(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage_remediation(tmp_path, ["11111111"])
    calls: list = []
    posts: list = []
    read, write = _remediate_transports(calls, posts, [("SID1", "11111111"), ("SID2", "22222222")])
    code = main(["remediate", "plan"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write)  # type: ignore[arg-type]
    assert code == 0
    out = capsys.readouterr().out
    assert "revise challenge=" in out
    assert "withdraw challenge=" in out
    revise_line = next(line for line in out.splitlines() if line.startswith("revise challenge="))
    withdraw_line = next(line for line in out.splitlines() if line.startswith("withdraw challenge="))
    assert revise_line.split("challenge=")[1] != withdraw_line.split("challenge=")[1]


def test_run_without_execute_sends_nothing(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage_remediation(tmp_path, ["11111111"])
    calls: list = []
    posts: list = []
    read, write = _remediate_transports(calls, posts, [("SID1", "11111111"), ("SID2", "22222222")])
    assert main(["remediate", "plan"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0  # type: ignore[arg-type]
    capsys.readouterr()
    posts.clear()
    revises: list = []

    def write2(method: str, url: str, headers: object, body: bytes | None, timeout: float) -> HttpResponse:
        posts.append(url)
        if url.endswith("/preview"):
            return _resp(200, {"ok": True, "total_amount_krw": 2000, "line_amounts_krw": {},
                               "next_action": "revise_existing", "product_already_claimed": False,
                               "claimed_by_me": True})
        revises.append(url)
        return _resp(200, {"submission_id": "SID1", "revision": 3, "submission_state": "accepted",
                           "amount_krw": 2000, "lint_status": "clean"})

    assert main(["remediate", "run", "--kind", "revise"], environ={"SAVETAX_TOKEN": TOKEN},  # type: ignore[arg-type]
                transport=read, write_transport=write2) == 0
    assert revises == []
    assert "would-revise" in capsys.readouterr().out


def test_run_executes_only_approved_rows(tmp_path: Path, monkeypatch: Any) -> None:
    from src.savetax.ledger import Ledger, LedgerEntry
    from src.savetax.submissions import SubmissionDetail
    from src.savetax.workflow import action_hash

    monkeypatch.chdir(tmp_path)
    _stage_remediation(tmp_path, ["11111111", "22222222"])
    calls: list = []
    posts: list = []
    read, write = _remediate_transports(calls, posts, [("SID1", "11111111"), ("SID2", "22222222")])
    assert main(["remediate", "plan"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0  # type: ignore[arg-type]
    with (tmp_path / "scratch" / "evidence" / "remediation" / "plan.jsonl").open(encoding="utf-8") as handle:
        out = handle.read()
    assert "revise" in out
    ledger = Ledger(tmp_path / "scratch" / "evidence" / "ledger.jsonl", now=lambda: datetime.now(UTC))
    body1 = json.loads((tmp_path / "scratch" / "evidence" / "remediation" / "drafts" / "11111111.json").read_text(encoding="utf-8"))
    detail1 = SubmissionDetail("SID1", "11111111", "accepted", 2, 2000, False, "clean", (), ())
    ledger.append(LedgerEntry(at=datetime.now(UTC), kind="approved", item_id="11111111", submission_id="SID1",
                              body_sha256=action_hash("revise", detail1, body1), idempotency_key=None,
                              data={"kind": "revise", "note": "yes", "total": 2000}))
    sent: list = []

    def read2(url: str, headers: object, timeout: float) -> HttpResponse:
        if url.endswith("/events/current"):
            return _resp(200, _event_body("RUNNING"))
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body())
        if "/events/current/submissions/" in url and "preview" not in url and "me/submissions" not in url:
            sid = url.rsplit("/", 1)[-1]
            item = "11111111" if sid == "SID1" else "22222222"
            return _resp(200, _submission_payload(sid, item))
        if url.endswith("/events/current/me/submissions"):
            return _resp(200, {"submissions": [_submission_payload("SID1", "11111111"),
                                                _submission_payload("SID2", "22222222")]})
        if "/events/current/items/" in url:
            return _resp(200, _item_body(url.rsplit("/", 1)[-1]))
        if "savetax.now/listing" in url:
            return HttpResponse(status=200, headers={}, body=_LISTING_PAGE.encode(), elapsed_seconds=0.01)
        raise AssertionError(url)

    def write2(method: str, url: str, headers: object, body: bytes | None, timeout: float) -> HttpResponse:
        if url.endswith("/preview"):
            return _resp(200, {"ok": True, "total_amount_krw": 2000, "line_amounts_krw": {},
                               "next_action": "revise_existing", "product_already_claimed": False,
                               "claimed_by_me": True})
        sent.append(url)
        return _resp(200, {"submission_id": "SID1", "revision": 3, "submission_state": "accepted",
                           "amount_krw": 2000, "lint_status": "clean"})

    code = main(["remediate", "run", "--kind", "revise", "--execute"], environ={"SAVETAX_TOKEN": TOKEN},  # type: ignore[arg-type]
                transport=read2, write_transport=write2)
    assert code == 0
    assert len(sent) == 1
    entries = Ledger(tmp_path / "scratch" / "evidence" / "ledger.jsonl", now=lambda: datetime.now(UTC)).entries()
    assert any(e.kind == "refused" for e in entries)


def test_stale_plan_is_refused(tmp_path: Path, monkeypatch: Any) -> None:
    from src.savetax.ledger import Ledger, LedgerEntry
    from src.savetax.submissions import SubmissionDetail
    from src.savetax.workflow import action_hash

    monkeypatch.chdir(tmp_path)
    _stage_remediation(tmp_path, ["11111111"])
    calls: list = []
    posts: list = []
    read, write = _remediate_transports(calls, posts, [("SID1", "11111111")])
    assert main(["remediate", "plan"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0  # type: ignore[arg-type]
    ledger = Ledger(tmp_path / "scratch" / "evidence" / "ledger.jsonl", now=lambda: datetime.now(UTC))
    body1 = json.loads((tmp_path / "scratch" / "evidence" / "remediation" / "drafts" / "11111111.json").read_text(encoding="utf-8"))
    stale = SubmissionDetail("SID1", "11111111", "accepted", 2, 2000, False, "clean", (), ())
    ledger.append(LedgerEntry(at=datetime.now(UTC), kind="approved", item_id="11111111", submission_id="SID1",
                              body_sha256=action_hash("revise", stale, body1), idempotency_key=None,
                              data={"kind": "revise", "note": "yes", "total": 2000}))

    def read2(url: str, headers: object, timeout: float) -> HttpResponse:
        if url.endswith("/events/current"):
            return _resp(200, _event_body("RUNNING"))
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body())
        if "/events/current/submissions/SID1" in url:
            payload = _submission_payload("SID1", "11111111", revision=9)
            return _resp(200, payload)
        if url.endswith("/events/current/me/submissions"):
            return _resp(200, {"submissions": [_submission_payload("SID1", "11111111", revision=9)]})
        if "/events/current/items/" in url:
            return _resp(200, _item_body("11111111"))
        if "savetax.now/listing" in url:
            return HttpResponse(status=200, headers={}, body=_LISTING_PAGE.encode(), elapsed_seconds=0.01)
        raise AssertionError(url)

    def write2(method: str, url: str, headers: object, body: bytes | None, timeout: float) -> HttpResponse:
        if url.endswith("/preview"):
            return _resp(200, {"ok": True, "total_amount_krw": 2000, "line_amounts_krw": {},
                               "next_action": "revise_existing", "product_already_claimed": False,
                               "claimed_by_me": True})
        raise AssertionError(f"stale approval must not transmit to {url}")

    code = main(["remediate", "run", "--kind", "revise", "--execute"], environ={"SAVETAX_TOKEN": TOKEN},  # type: ignore[arg-type]
                transport=read2, write_transport=write2)
    assert code == 0


def test_load_rebuilt_skips_draft_without_bundle(tmp_path, monkeypatch):
    import pytest
    from src.savetax.errors import DataIntegrityError
    from datetime import datetime
    from src.savetax.cli import _load_rebuilt
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "11111111.json").write_text(__import__("json").dumps(_draft_body("11111111")), encoding="utf-8")
    bundles = tmp_path / "cross-bundles.jsonl"
    bundles.write_text("", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        _load_rebuilt(drafts, bundles, datetime.now(UTC))


def test_submitted_hashes_collects_submitted_and_revised(tmp_path):
    from src.savetax.cli import _submitted_hashes
    from src.savetax.ledger import Ledger, LedgerEntry
    ledger = Ledger(tmp_path / "ledger.jsonl", now=lambda: datetime.now(UTC))
    ledger.append(LedgerEntry(at=datetime.now(UTC), kind="submitted", item_id="11111111",
                              submission_id="SID1", body_sha256="h1", idempotency_key="k",
                              data={"submission_id": "SID1"}))
    ledger.append(LedgerEntry(at=datetime.now(UTC), kind="revised", item_id="22222222",
                              submission_id="SID2", body_sha256="h2", idempotency_key="k2",
                              data={"submission_id": "SID2", "body_hash": "h2"}))
    assert _submitted_hashes(ledger) == {"11111111": "h1", "22222222": "h2"}


def test_read_plan_rows_guards(tmp_path):
    import pytest
    from src.savetax.cli import _read_plan_rows
    from src.savetax.errors import DataIntegrityError
    with pytest.raises(DataIntegrityError):
        _read_plan_rows(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        _read_plan_rows(bad)
    non_dict = tmp_path / "nd.jsonl"
    non_dict.write_text("[1,2]\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        _read_plan_rows(non_dict)


def test_read_plan_rows_skips_blank_lines(tmp_path):
    from src.savetax.cli import _read_plan_rows
    plan = tmp_path / "plan.jsonl"
    plan.write_text("\n   \n{\"a\": 1}\n\n", encoding="utf-8")
    assert _read_plan_rows(plan) == [{"a": 1}]


def test_compute_remediation_skips_none_item(tmp_path, monkeypatch):
    from src.savetax import cli as cli_mod
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scratch" / "evidence" / "remediation" / "drafts").mkdir(parents=True)
    (tmp_path / "scratch" / "evidence" / "remediation" / "cross-bundles.jsonl").write_text("", encoding="utf-8")
    class FakeSub:
        def list_my_submissions(self):
            from src.savetax.submissions import SubmissionDetail
            return [SubmissionDetail("SIDX", None, "accepted", 1, 100, False, "clean", (), ())]
        def preview(self, body):
            raise AssertionError("no previews expected")
    class FakeComp:
        def get_item(self, item_id):
            raise AssertionError("no item fetch expected")
    monkeypatch.setattr(cli_mod, "_submission_stack", lambda *a, **k: (None, FakeComp(), FakeSub(), None))
    out = cli_mod._compute_remediation({}, None, None)
    assert out[-2] == []


def test_compute_remediation_pending_on_empty_facts(tmp_path, monkeypatch):
    import pytest
    from src.savetax import cli as cli_mod
    from src.savetax.errors import DataAccessPending
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scratch" / "evidence" / "remediation" / "drafts").mkdir(parents=True)
    (tmp_path / "scratch" / "evidence" / "remediation" / "cross-bundles.jsonl").write_text("", encoding="utf-8")
    class FakeSub:
        def list_my_submissions(self):
            from src.savetax.submissions import SubmissionDetail
            return [SubmissionDetail("SID1", "11111111", "accepted", 1, 100, False, "clean", (), ())]
        def preview(self, body):
            raise AssertionError("unreachable")
    class FakeComp:
        def get_item(self, item_id):
            from src.savetax.competition import ItemDetail
            return ItemDetail(item_id, "claimed_by_me", None, None, None, None, None, None, False, ())
    monkeypatch.setattr(cli_mod, "_submission_stack", lambda *a, **k: (None, FakeComp(), FakeSub(), None))
    with pytest.raises(DataAccessPending):
        cli_mod._compute_remediation({}, None, None)


def test_compute_remediation_preview_error_skipped(tmp_path, monkeypatch, capsys):
    from src.savetax import cli as cli_mod
    from src.savetax.errors import ApiError
    monkeypatch.chdir(tmp_path)
    _stage_remediation(tmp_path, ["11111111"])
    class FakeSub:
        def list_my_submissions(self):
            return []
        def preview(self, body):
            raise ApiError("competition", 500, "TRY_AGAIN", "x", True)
    class FakeComp:
        def get_item(self, item_id):
            return _item_body(item_id) and __import__("src.savetax.competition", fromlist=["ItemDetail"]).ItemDetail(
                item_id, __import__("src.savetax.competition", fromlist=["ItemState"]).ItemState.CLAIMED_BY_ME,
                None, None, None, None, None, 0, False, ())
        def get_event(self):
            return None
    # reuse real item facts via _item_body check: need facts non-empty so build minimal
    from src.savetax.competition import ItemDetail, ItemState, Fact
    import datetime as _dt
    from decimal import Decimal as _D
    class FakeComp2:
        def get_item(self, item_id):
            fact = Fact("F1", "L1", "C1", "R1", "1", _dt.date(2026, 10, 1), "EA", "vat_included",
                        "b", "none", _D("2000"), _D("2"), _D("4000"), True, ())
            return ItemDetail(item_id, ItemState.CLAIMED_BY_ME, None, None, None, None, None, 1, False, (fact,))
        def get_event(self):
            return None
    monkeypatch.setattr(cli_mod, "_submission_stack", lambda *a, **k: (None, FakeComp2(), FakeSub(), None))
    out = cli_mod._compute_remediation({}, None, None)
    assert out[-2] == []


def test_remediate_plan_tolerates_review_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _stage_remediation(tmp_path, ["11111111"])
    calls: list = []
    posts: list = []
    read, write = _remediate_transports(calls, posts, [("SID1", "11111111")])
    import src.savetax.workflow as _wf
    orig = _wf.action_review
    def boom(drafts, client, kind, *, out=None):
        from src.savetax.errors import ConfigError
        raise ConfigError("[SYS] action requires exactly one current own submission")
    monkeypatch.setattr(_wf, "action_review", boom)
    try:
        code = main(["remediate", "plan"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write)  # type: ignore[arg-type]
    finally:
        monkeypatch.setattr(_wf, "action_review", orig)
    assert code == 0
    assert "totals" in capsys.readouterr().out


def test_owned_evidence_writes_only_remediation(tmp_path, monkeypatch):
    import duckdb
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "savetax-products-test"
    data_dir.mkdir(parents=True, exist_ok=True)
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
        con.execute(
            "INSERT INTO tmp VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["111111111_0000000001", "11111111", "n", "d", "c", "m", "mod", "EA",
             1200000, "2025-01-01", "2026-12-31", "v", 1, 1, 10, "EA", 12000000,
             "2025-10-11", "2025-10-12"],
        )
        cols = ("listing_id,item_id,name,detail_category,category_l_name,manufacturer,model,unit,price,"
                "contract_begin,contract_end,vendor_name,dlv_key,dlv_lines,dlv_qty,dlv_unit,dlv_amount,"
                "dlv_first_date,dlv_last_date")
        con.execute(f"COPY (SELECT {cols} FROM tmp) TO '{data_dir / 'listings.parquet'}' (FORMAT PARQUET)")  # noqa: S608
    finally:
        con.close()
    def transport(url, headers, timeout):
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body())
        if url.endswith("/events/current"):
            return _resp(200, _event_body("open"))
        if "/items/lookup" in url:
            ids = url.split("ids=")[1].split(",")
            return _resp(200, {"items": [{"input": i, "kind": "item", "item_id": i,
                                          "state": "claimed_by_me"} for i in ids]})
        if "/events/current/items/" in url:
            return _resp(200, _item_body(url.rsplit("/", 1)[-1]))
        raise AssertionError(url)
    out_dir = tmp_path / "ev"
    code = main(["evidence", "--owned", "--out-dir", str(out_dir), "--data-dir", str(data_dir)],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=transport)
    assert code == 0
    assert (out_dir / "remediation" / "cross-bundles.jsonl").exists()
    assert not (out_dir / "cross-bundles.jsonl").exists()


def test_remediate_approve_revise_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _stage_remediation(tmp_path, ["11111111"])
    calls: list = []
    posts: list = []
    read, write = _remediate_transports(calls, posts, [("SID1", "11111111")])
    assert main(["remediate", "plan"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    challenge = next(line for line in out.splitlines() if line.startswith("revise challenge=")).split("challenge=")[1].split(" total=")[0]
    assert main(["remediate", "approve", "--kind", "revise", "--challenge", challenge, "--note", "ok"],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0
    assert main(["remediate", "approve", "--kind", "revise", "--challenge", "WRONG", "--note", "ok"],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 2
    assert main(["remediate", "approve", "--kind", "revise", "--challenge", challenge, "--note", "  "],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 2


def test_remediate_approve_withdraw_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rem = tmp_path / "scratch" / "evidence" / "remediation"
    (rem / "drafts").mkdir(parents=True, exist_ok=True)
    (rem / "cross-bundles.jsonl").write_text("", encoding="utf-8")
    calls: list = []
    posts: list = []
    read, write = _remediate_transports(calls, posts, [("SID9", "99999999")])
    assert main(["remediate", "plan"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "withdraw challenge=" in out
    challenge = next(line for line in out.splitlines() if line.startswith("withdraw challenge=")).split("challenge=")[1].split(" total=")[0]
    assert main(["remediate", "approve", "--kind", "withdraw", "--challenge", challenge, "--note", "bye"],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0


def test_remediate_plan_withdraw_failure_tolerated(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rem = tmp_path / "scratch" / "evidence" / "remediation"
    (rem / "drafts").mkdir(parents=True, exist_ok=True)
    (rem / "cross-bundles.jsonl").write_text("", encoding="utf-8")
    calls: list = []
    posts: list = []
    read, write = _remediate_transports(calls, posts, [("SID9", "99999999")])
    import src.savetax.workflow as _wf
    def boom(drafts, client, kind, *, out=None):
        from src.savetax.errors import ConfigError
        raise ConfigError("[SYS] boom")
    monkeypatch.setattr(_wf, "action_review", boom)
    code = main(["remediate", "plan"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write)  # type: ignore[arg-type]
    assert code == 0


def test_remediate_run_guards_and_withdraw(tmp_path, monkeypatch, capsys):
    from src.savetax.ledger import Ledger, LedgerEntry
    from src.savetax.submissions import SubmissionDetail
    from src.savetax.workflow import action_hash
    monkeypatch.chdir(tmp_path)
    rem = tmp_path / "scratch" / "evidence" / "remediation"
    (rem / "drafts").mkdir(parents=True, exist_ok=True)
    (rem / "cross-bundles.jsonl").write_text("", encoding="utf-8")
    plan = rem / "plan.jsonl"
    import json as _j
    plan.write_text(
        _j.dumps({"item_id": "99999999", "submission_id": "SID9", "action": "withdraw", "revision": 2}) + "\n"
        + _j.dumps({"item_id": "bad", "action": "withdraw"}) + "\n"
        + _j.dumps({"item_id": "11111111", "submission_id": "SID1", "action": "revise", "revision": 2}) + "\n",
        encoding="utf-8",
    )
    assert main(["remediate", "run", "--kind", "revise", "--limit", "0"],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=None) == 2
    ledger = Ledger(tmp_path / "scratch" / "evidence" / "ledger.jsonl", now=lambda: datetime.now(UTC))
    detail = SubmissionDetail("SID9", "99999999", "accepted", 2, 100, False, "clean", (), ())
    ledger.append(LedgerEntry(at=datetime.now(UTC), kind="approved", item_id="99999999", submission_id="SID9",
                              body_sha256=action_hash("withdraw", detail, {"canonical_product_id": "99999999"}),
                              idempotency_key=None, data={"kind": "withdraw", "note": "bye", "total": 0}))
    def read(url, headers, timeout):
        if url.endswith("/events/current"):
            return _resp(200, _event_body("RUNNING"))
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body())
        if "/events/current/submissions/SID9" in url:
            return _resp(200, {"submission_id": "SID9", "item_id": "99999999", "submission_state": "accepted",
                               "current_revision": 2, "amount_krw": 100, "lint_status": "clean"})
        raise AssertionError(url)
    sent: list = []
    def write(url_or_method, url=None, headers=None, body=None, timeout=None):
        # supports both read-style and write-style calls
        target = url if isinstance(url, str) else url_or_method
        if isinstance(target, str) and target.endswith("/preview"):
            return _resp(200, {"ok": True, "total_amount_krw": 0, "line_amounts_krw": {},
                               "next_action": "submit", "product_already_claimed": False, "claimed_by_me": True})
        sent.append(target)
        return _resp(200, {"submission_id": "SID9", "revision": 2, "submission_state": "withdrawn",
                           "amount_krw": 0, "lint_status": "clean"})
    code = main(["remediate", "run", "--kind", "withdraw", "--execute"],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write)
    assert code == 0
    assert sent
    # revise with missing rebuilt is skipped
    code2 = main(["remediate", "run", "--kind", "revise", "--limit", "1"],  # type: ignore[arg-type]
                 environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write)
    assert code2 == 0
