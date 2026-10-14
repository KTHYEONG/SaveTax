"""Invariant guards for the review/approve/submit/status/revise/withdraw commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.savetax.cli import main
from src.savetax.config import ApprovalConfig
from src.savetax.errors import ConfigError
from src.savetax.http import HttpResponse

import pytest


TOKEN = "savetax_test-token_1"


def _resp(status: int, payload: object) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(status=status, headers={}, body=body, elapsed_seconds=0.01)


def _event_body() -> dict:
    return {
        "dataset_release_id": "release-1", "ruleset_version": "rules-v1", "phase": "open",
        "opens_at": "2026-10-01T00:00:00Z", "closes_at": "2026-11-01T00:00:00Z",
        "evidence_format": "v2", "item_catalog": "catalog",
    }


def _me_body() -> dict:
    return {"handle": "h", "registered": True, "claims_used": 0, "claims_today": 0,
            "remaining": {"claims_total": 100, "claims_per_day": 50}}


def _preview_body(total: int = 2000) -> dict:
    return {"ok": True, "total_amount_krw": total, "line_amounts_krw": {}, "next_action": "submit",
            "product_already_claimed": False, "claimed_by_me": False}


def _draft_body(item: str = "11111111") -> dict:
    entry = _bundle_line(item)["lines"][0]
    evidence = dict(entry["price_evidence"][0], kind="price_page", observed_price_krw="1000")
    evidence.pop("note")
    line = {key: entry[key] for key in ("fact_id", "benchmark_unit_price_krw", "comparison_basis_id", "unit", "tax_basis")}
    line.update(period=None, price_evidence=[evidence])
    return {"dataset_release_id": "release-1", "ruleset_version": "rules-v1",
            "canonical_product_id": item, "title": f"title-{item}", "lines": [line]}


def _bundle_line(item: str = "11111111", total: int = 2000) -> dict:
    return {
        "item_id": item, "lane": "cross_contract", "title": f"title-{item}", "claim": "c",
        "comparison_scope": "s", "reproduction_notes": "r",
        "lines": [{"fact_id": "F1", "benchmark_unit_price_krw": "1000", "comparison_basis_id": "b",
                   "unit": "대", "tax_basis": "vat_included", "fact_unit_price_krw": "2000",
                   "quantity": "2", "comparable_amount_krw": "4000", "request_date": "2026-10-01",
                   "expected_krw": total,
                   "price_evidence": [{"source_url": "https://savetax.now/listing/x", "quote": "q tag",
                                       "observed_price_krw": 1000, "observed_at": "2026-10-01",
                                       "vat_included": True, "shipping_included": True, "note": None}]}],
        "rejected": [], "caveats": [], "review_required": False,
        "generated_at": "2026-10-13T12:00:00+00:00", "expected_total_krw": total, "unit_notes": [],
    }


def _stage(tmp_path: Path, item: str = "11111111", total: int = 2000) -> None:
    evidence_dir = tmp_path / "scratch" / "evidence"
    drafts = evidence_dir / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    (drafts / f"{item}.json").write_text(json.dumps(_draft_body(item)), encoding="utf-8")
    (evidence_dir / "cross-bundles.jsonl").write_text(json.dumps(_bundle_line(item, total)) + "\n", encoding="utf-8")
    (evidence_dir / "retail-bundles.jsonl").write_text("", encoding="utf-8")


def _transport(calls: list, posts: list, *, preview_total: int = 2000) -> Any:
    def read(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(("GET", url))
        if url.endswith("/events/current"):
            return _resp(200, _event_body())
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body())
        if "/items/lookup" in url:
            ids = url.split("ids=")[1].split(",")
            return _resp(200, {"items": [{"input": i, "kind": "item", "item_id": i, "state": "available"} for i in ids]})
        if "/events/current/submissions/" in url:
            return _resp(200, {"submission_id": "SID1", "item_id": "11111111", "submission_state": "accepted",
                               "current_revision": 1, "amount_krw": 2000, "lint_status": "clean"})
        if "/events/current/me/submissions" in url:
            return _resp(200, {"submissions": []})
        if "/submission-requests/" in url:
            return _resp(404, {"code": "IDEMPOTENCY_KEY_NOT_FOUND", "message": "m", "retryable": False})
        if "savetax.now/listing" in url:
            return HttpResponse(status=200, headers={}, body="단위 대 물품식별번호 11111111 계약품목관리번호 x 계약 단가 1,000원 / 대 공급 업체 v".encode(), elapsed_seconds=0.01)
        raise AssertionError(f"unexpected GET {url}")

    def write(method: str, url: str, headers: object, body: bytes | None, timeout: float) -> HttpResponse:
        posts.append(url)
        payload = json.loads(body.decode()) if body else {}
        if url.endswith("/events/current/submissions/preview"):
            return _resp(200, _preview_body(preview_total))
        if url.endswith("/events/current/submissions"):
            return _resp(201, {"submission_id": "SID1", "revision": 1, "submission_state": "accepted",
                               "amount_krw": 2000, "lint_status": "clean"})
        raise AssertionError(f"unexpected POST {url}")

    return read, write


def test_read_only_commands_never_post_write_path(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    calls: list = []
    posts: list = []
    read, write = _transport(calls, posts)
    assert main(["review"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0  # type: ignore[arg-type]
    assert posts
    assert all(p.endswith("/events/current/submissions/preview") for p in posts)
    posts.clear()
    assert main(["status"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0  # type: ignore[arg-type]
    assert posts == []
    out = capsys.readouterr().out
    assert "quote" not in out


def test_submit_without_execute_is_dry_run(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    calls: list = []
    posts: list = []
    read, write = _transport(calls, posts)
    assert main(["submit"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0  # type: ignore[arg-type]
    assert not any(p.endswith("/events/current/submissions") for p in posts)
    assert "would-submit" in capsys.readouterr().out


def test_approve_requires_note(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    calls: list = []
    posts: list = []
    read, write = _transport(calls, posts)
    assert main(["approve", "--challenge", "APPROVE 0 0 deadbeef", "--note", "  "],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 2


def test_approve_then_submit_flow(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    calls: list = []
    posts: list = []
    read, write = _transport(calls, posts)
    kwargs: dict[str, Any] = {"environ": {"SAVETAX_TOKEN": TOKEN}, "transport": read, "write_transport": write}
    assert main(["review"], **kwargs) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    challenge = next(line for line in out.splitlines() if line.startswith("total=")).split("challenge=")[1]
    assert main(["approve", "--challenge", challenge, "--note", "looks good"], **kwargs) == 0  # type: ignore[arg-type]
    posts.clear()
    assert main(["submit", "--execute"], **kwargs) == 0  # type: ignore[arg-type]
    assert any(p.endswith("/events/current/submissions") for p in posts)


def test_revise_and_withdraw_dry_runs(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    draft = tmp_path / "scratch" / "evidence" / "drafts" / "11111111.json"
    assert main(["revise", "--item", "11111111", "--draft", str(draft)], environ={}) == 0
    assert "would-revise" in capsys.readouterr().out
    assert main(["withdraw", "--item", "SID1"], environ={}) == 0
    assert "would-withdraw" in capsys.readouterr().out


def test_token_never_printed(tmp_path: Path, monkeypatch: Any, capsys: Any, caplog: Any) -> None:
    import logging

    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    calls: list = []
    posts: list = []
    read, write = _transport(calls, posts)
    kwargs: dict[str, Any] = {"environ": {"SAVETAX_TOKEN": TOKEN}, "transport": read, "write_transport": write}
    with caplog.at_level(logging.INFO):
        assert main(["review"], **kwargs) == 0  # type: ignore[arg-type]
        assert main(["status"], **kwargs) == 0  # type: ignore[arg-type]
        assert main(["submit"], **kwargs) == 0  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert TOKEN not in captured.out
    assert TOKEN not in captured.err
    assert TOKEN not in caplog.text


def test_evidence_path_has_no_write_capability() -> None:
    import pathlib

    for name in ("evidence.py", "crosscontract.py", "retail.py"):
        text = (pathlib.Path("src/savetax") / name).read_text(encoding="utf-8")
        assert "writehttp" not in text
        assert "submissions" not in text


def test_approval_config_defaults_and_bounds() -> None:
    cfg = ApprovalConfig()
    assert (cfg.ttl_hours, cfg.reserve_claims, cfg.ledger_path) == (6, 5, Path("scratch/evidence/ledger.jsonl"))
    for kw in ({"ttl_hours": 0}, {"reserve_claims": -1}, {"lint_poll_seconds": 0.0},
               {"lint_timeout_seconds": 1.0, "lint_poll_seconds": 7.0},
               {"pause_between_submits_seconds": -1.0}, {"stop_after_consecutive_suspect": 0}):
        with pytest.raises(ConfigError):
            ApprovalConfig(**kw)  # type: ignore[arg-type]


@pytest.mark.parametrize(("value", "expected"), [(None, 5), (0, 0), (3, 3)])
def test_reserve_override_defaults_to_config(value: int | None, expected: int) -> None:
    import argparse

    from src.savetax.cli import _reserve

    assert _reserve(argparse.Namespace(reserve=value), 5) == expected


def test_negative_reserve_rejected() -> None:
    import argparse

    from src.savetax.cli import _reserve

    with pytest.raises(ConfigError, match="reserve"):
        _reserve(argparse.Namespace(reserve=-1), 5)


@pytest.mark.parametrize(("value", "expected"), [(None, 2), (1, 1), (11, 11)])
def test_stop_after_suspect_override(value: int | None, expected: int) -> None:
    import argparse

    from src.savetax.cli import _stop_after_suspect

    assert _stop_after_suspect(argparse.Namespace(stop_after_suspect=value), 2) == expected


def test_zero_stop_after_suspect_rejected() -> None:
    import argparse

    from src.savetax.cli import _stop_after_suspect

    with pytest.raises(ConfigError, match="stop-after-suspect"):
        _stop_after_suspect(argparse.Namespace(stop_after_suspect=0), 2)


def test_review_limit_validated(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    calls: list = []
    posts: list = []
    read, write = _transport(calls, posts)
    assert main(["review", "--limit", "0"], environ={"SAVETAX_TOKEN": TOKEN},  # type: ignore[arg-type]
                transport=read, write_transport=write) == 2
    assert main(["submit", "--execute", "--limit", "0"], environ={"SAVETAX_TOKEN": TOKEN},  # type: ignore[arg-type]
                transport=read, write_transport=write) == 2


def test_malformed_draft_revise_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert main(["revise", "--item", "11111111", "--draft", str(bad), "--execute"],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=None) == 2
    lst = tmp_path / "list.json"
    lst.write_text("[1]", encoding="utf-8")
    assert main(["revise", "--item", "11111111", "--draft", str(lst), "--execute"],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=None) == 2


def test_live_price_none_on_unreadable_page() -> None:
    from src.savetax.cli import _live_price
    from src.savetax.errors import ConfigError

    class _Page:
        def __init__(self, status: int, text: str) -> None:
            self.status = status
            self.text = text

    class _Fetcher:
        def __init__(self, page: Any = None, error: Any = None) -> None:
            self._page = page
            self._error = error

        def get(self, url: str) -> Any:
            if self._error is not None:
                raise self._error("[SYS] denied")
            return self._page

    assert _live_price(_Fetcher(_Page(403, "")), "https://savetax.now/listing/x") is None  # type: ignore[arg-type]
    assert _live_price(_Fetcher(_Page(200, "<html>no price</html>")), "https://savetax.now/listing/x") is None  # type: ignore[arg-type]
    assert _live_price(_Fetcher(error=ConfigError), "https://savetax.now/listing/x") is None  # type: ignore[arg-type]
    text = "단위 대 물품식별번호 11111111 계약품목관리번호 x 계약 단가 2,500원 / 대 공급 업체 v"
    assert _live_price(_Fetcher(_Page(200, text)), "https://savetax.now/listing/x") == 2500  # type: ignore[arg-type]


def test_read_only_transport_refuses_writes() -> None:
    from src.savetax.cli import _refusing_write_transport

    with pytest.raises(ConfigError):
        _refusing_write_transport("POST", "https://example.com", {}, None, 1.0)


def test_review_limit_selects_top_row(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    calls: list = []
    posts: list = []
    read, write = _transport(calls, posts)
    assert main(["review", "--limit", "1"], environ={"SAVETAX_TOKEN": TOKEN},  # type: ignore[arg-type]
                transport=read, write_transport=write) == 0
    assert "APPROVE 1 2000" in capsys.readouterr().out


def test_withdraw_unknown_item_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    assert main(["withdraw", "--item", "99999999", "--execute"], environ={}) == 2  # type: ignore[arg-type]


def test_status_lists_submissions(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    calls: list = []
    posts: list = []

    def read(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        if url.endswith("/events/current/me/submissions"):
            return _resp(200, {"submissions": [{"submission_id": "SID1", "item_id": "11111111",
                                                "submission_state": "accepted", "current_revision": 1,
                                                "amount_krw": 2000, "lint_status": "clean"}]})
        if url.endswith("/events/current/me"):
            return _resp(200, _me_body())
        raise AssertionError(url)

    assert main(["status"], environ={"SAVETAX_TOKEN": TOKEN}, transport=read) == 0  # type: ignore[arg-type]
    assert "SID1" in capsys.readouterr().out


def test_revise_execute_with_approval(tmp_path: Path, monkeypatch: Any) -> None:
    from src.savetax.ledger import Ledger, LedgerEntry

    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    from src.savetax.submissions import SubmissionDetail
    from src.savetax.workflow import action_hash

    detail = SubmissionDetail("SID1", "11111111", "accepted", 2, 2000, False, "clean", (), ())
    digest = action_hash("revise", detail, _draft_body())
    ledger = Ledger(tmp_path / "scratch" / "evidence" / "ledger.jsonl", now=lambda: datetime.now(UTC))
    ledger.append(LedgerEntry(at=datetime.now(UTC), kind="approved", item_id="11111111", submission_id=None,
                              body_sha256=digest, idempotency_key=None,
                              data={"kind": "revise", "note": "yes", "total": 1}))
    posts: list = []

    def read(url: str, headers: object, timeout: float) -> HttpResponse:
        if url.endswith("/events/current"):
            return _resp(200, _event_body())
        if url.endswith("/me/submissions"):
            return _resp(200, {"submissions": [{"submission_id": "SID1", "item_id": "11111111", "state": "accepted", "revision": 2, "lint_status": "clean"}]})
        if "savetax.now/listing" in url:
            return _resp(200, "단위 대 물품식별번호 11111111 계약품목관리번호 x 계약 단가 1,000원 / 대 공급 업체 v".encode())
        if url.endswith("/events/current/submissions/SID1"):
            return _resp(200, {"submission_id": "SID1", "item_id": "11111111", "submission_state": "accepted",
                               "current_revision": 2, "amount_krw": 2000, "lint_status": "clean"})
        raise AssertionError(url)

    def write(method: str, url: str, headers: object, body: bytes | None, timeout: float) -> HttpResponse:
        posts.append(url)
        if url.endswith("/preview"):
            return _resp(200, dict(_preview_body(), next_action="revise_existing"))
        return _resp(200, {"submission_id": "SID1", "revision": 3, "submission_state": "accepted",
                           "amount_krw": 2000, "lint_status": "clean"})

    draft = tmp_path / "scratch" / "evidence" / "drafts" / "11111111.json"
    assert main(["revise", "--item", "11111111", "--draft", str(draft), "--execute"],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0
    assert any(p.endswith("/revisions") for p in posts)


def test_withdraw_execute_flow(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    from src.savetax.ledger import Ledger, LedgerEntry
    from src.savetax.submissions import SubmissionDetail
    from src.savetax.workflow import action_hash

    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    assert main(["withdraw", "--item", "11111111", "--execute"],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=None) == 2
    ledger = Ledger(tmp_path / "scratch" / "evidence" / "ledger.jsonl", now=lambda: datetime.now(UTC))
    ledger.append(LedgerEntry(at=datetime.now(UTC), kind="submitted", item_id="11111111", submission_id="SID1",
                              body_sha256="h", idempotency_key="k", data={"submission_id": "SID1"}))
    ledger.append(LedgerEntry(at=datetime.now(UTC), kind="approved", item_id="11111111",
                              submission_id=None, body_sha256=action_hash("withdraw", SubmissionDetail(
                                  "SID1", "11111111", "accepted", 1, 2000, False, "clean", (), ()), _draft_body()),
                              idempotency_key=None, data={"kind": "withdraw", "note": "yes", "total": 1}))
    posts: list = []
    read, _ = _transport([], [])

    def write(method: str, url: str, headers: object, body: bytes | None, timeout: float) -> HttpResponse:
        posts.append(url)
        return _resp(200, {"submission_id": "SID1", "revision": 1, "submission_state": "withdrawn",
                           "amount_krw": 2000, "lint_status": "clean"})

    assert main(["withdraw", "--item", "11111111", "--execute"],  # type: ignore[arg-type]
                environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0
    assert any(p.endswith("/withdraw") for p in posts)
    assert "does not refund" in capsys.readouterr().out
