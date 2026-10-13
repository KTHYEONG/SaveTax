"""End-to-end guards for explicit action approval and public price parsing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.savetax.cli as cli
from src.savetax.errors import ConfigError, DataIntegrityError, SubmissionPending
from src.savetax.http import HttpResponse
from src.savetax.ledger import Ledger, LedgerEntry
from src.savetax.workflow import body_hash
from tests.unit.savetax.test_cli_workflow import (
    TOKEN, _draft_body, _event_body, _me_body, _preview_body, _resp, _stage,
)
from tests.unit.savetax.test_workflow import _now


def _network(kind: str, posts: list[tuple[str, Any, Any]], state: dict[str, Any]) -> tuple[Any, Any]:
    def read(url: str, headers: Any, timeout: float) -> HttpResponse:
        if url.endswith("/events/current"):
            return _resp(200, _event_body())
        if url.endswith("/me"):
            return _resp(200, _me_body())
        detail = {"submission_id": "SID1", "item_id": "11111111", "state": "accepted", "revision": state["revision"],
                  "amount_krw": 2000, "lint_status": "clean"}
        if url.endswith("/me/submissions"):
            return _resp(200, {"submissions": [detail]})
        if url.endswith("/submissions/SID1"):
            return _resp(200, detail)
        if "savetax.now/listing" in url:
            return _resp(200, "단위 대 물품식별번호 11111111 계약품목관리번호 x 계약 단가 1,000원 / 대 공급 업체 v".encode())
        raise AssertionError(url)

    def write(method: str, url: str, headers: Any, payload: bytes | None, timeout: float) -> HttpResponse:
        posts.append((url, dict(headers), json.loads(payload) if payload else None))
        if url.endswith("/preview"):
            return _resp(200, dict(_preview_body(), next_action="revise_existing"))
        if url.endswith("/revisions"):
            assert kind == "revise"
            assert headers["If-Match"] == str(state["revision"])
            state["revision"] += 1
            return _resp(200, {"submission_id": "SID1", "revision": state["revision"], "state": "accepted",
                               "amount_krw": 2000, "lint_status": "clean"})
        if url.endswith("/withdraw"):
            assert kind == "withdraw"
            assert payload is None
            return _resp(200, {"submission_id": "SID1", "revision": state["revision"], "state": "withdrawn",
                               "amount_krw": 2000, "lint_status": "clean"})
        raise AssertionError(url)

    return read, write


@pytest.mark.parametrize("kind", ["revise", "withdraw"])
def test_action_cli_review_approve_execute_round_trip(
    tmp_path: Path, monkeypatch: Any, capsys: Any, kind: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_utcnow", _now)
    _stage(tmp_path)
    path = tmp_path / "scratch/evidence/drafts/11111111.json"
    ledger = Ledger(tmp_path / "scratch/evidence/ledger.jsonl", now=_now)
    ledger.append(LedgerEntry(_now(), "submitted", "11111111", "SID1", body_hash(_draft_body()), "old", {"amount": 2000}))
    posts: list[tuple[str, Any, Any]] = []
    state = {"revision": 2}
    read, write = _network(kind, posts, state)
    kwargs = {"environ": {"SAVETAX_TOKEN": TOKEN}, "transport": read, "write_transport": write}
    assert cli.main(["review", "--kind", kind, "--item", "11111111", "--draft", str(path)], **kwargs) == 0
    output = capsys.readouterr().out
    assert "old=2000 new=2000" in output
    challenge = next(line.split("challenge=")[1] for line in output.splitlines() if "challenge=" in line)
    assert all(url.endswith("/preview") for url, _, _ in posts)
    assert cli.main(["approve", "--kind", kind, "--item", "11111111", "--draft", str(path),
                     "--challenge", challenge, "--note", "approve precisely this action"], **kwargs) == 0
    posts.clear()
    arguments = [kind, "--item", "11111111", "--execute"]
    if kind == "revise":
        arguments += ["--draft", str(path)]
    assert cli.main(arguments, **kwargs) == 0
    assert any(url.endswith("/revisions" if kind == "revise" else "/withdraw") for url, _, _ in posts)
    assert TOKEN not in capsys.readouterr().out


def test_action_review_requires_exact_item(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)
    read, write = _network("withdraw", [], {"revision": 2})
    assert cli.main(["review", "--kind", "withdraw"], environ={"SAVETAX_TOKEN": TOKEN},
                    transport=read, write_transport=write) == 2


def test_revision_identity_and_refusal_exit_codes(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_utcnow", _now)
    _stage(tmp_path)
    path = tmp_path / "scratch/evidence/drafts/11111111.json"
    posts: list[tuple[str, Any, Any]] = []
    read, write = _network("revise", posts, {"revision": 2})
    kwargs = {"environ": {"SAVETAX_TOKEN": TOKEN}, "transport": read, "write_transport": write}
    assert cli.main(["revise", "--item", "99999999", "--draft", str(path), "--execute"], **kwargs) == 2
    assert cli.main(["revise", "--item", "11111111", "--draft", str(path), "--execute"], **kwargs) == 2
    assert posts == []


def test_pending_action_maps_to_pause(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage(tmp_path)

    def pending(*args: Any, **kwargs: Any) -> Any:
        raise SubmissionPending("hidden")

    monkeypatch.setattr(cli, "_run_submit", pending)
    assert cli.main(["submit", "--execute"]) == 3


def test_missing_bundle_and_single_draft_errors_fail_closed(tmp_path: Path) -> None:
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "x.json").write_text(json.dumps(_draft_body()))
    with pytest.raises(ConfigError, match="missing its evidence"):
        cli._workflow_inputs(drafts, tmp_path, _now())
    with pytest.raises(DataIntegrityError):
        cli._single_draft(tmp_path / "missing.json", tmp_path, _now())


@pytest.mark.parametrize(("url", "html", "expected"), [
    ("https://prod.danawa.com/info/?pcode=77", '<script type="application/ld+json">{"@type":"Product","name":"M","offers":{"lowPrice":2500,"offerCount":2}}</script>', 2500),
    ("https://itncctv.kr/product/x", '<meta property="og:title" content="M"><meta property="product:price:currency" content="KRW"><meta property="product:price:amount" content="3500">', 3500),
    ("https://itncctv.kr/product/x", "shipping fee 1000원 without product price", None),
    ("https://savetax.now/listing/x", "물품식별번호 11111111 계약품목관리번호 other 계약 단가 1000원", None),
])
def test_live_price_uses_source_specific_identity_and_price(url: str, html: str, expected: int | None) -> None:
    class Fetcher:
        def get(self, url: str) -> Any:
            return SimpleNamespace(status=200, text=html)

    assert cli._live_price(Fetcher(), url) == expected  # type: ignore[arg-type]


def test_live_price_network_error_pauses_instead_of_revoking_approval() -> None:
    from src.savetax.errors import NetworkError, NetworkStage

    class Fetcher:
        def get(self, url: str) -> Any:
            raise NetworkError(NetworkStage.DNS_RESOLUTION, "savetax.now", "unavailable")

    with pytest.raises(NetworkError):
        cli._live_price(Fetcher(), "https://savetax.now/listing/x")  # type: ignore[arg-type]


def test_readonly_client_has_no_write_endpoint_capability() -> None:
    posts: list[tuple[str, Any, Any]] = []
    read, write = _network("revise", posts, {"revision": 2})
    _, _, client, _ = cli._submission_stack({"SAVETAX_TOKEN": TOKEN}, read, write)
    with pytest.raises(ConfigError, match="read-only workflow"):
        client.submit(_draft_body(), idempotency_key="key")
    assert posts == []
