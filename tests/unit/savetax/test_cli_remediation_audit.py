"""Regression guards for remediation approval freshness and evidence integrity."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from src.savetax.cli import _load_rebuilt, _submitted_hashes, main
from src.savetax.evidence import write_bundles, write_draft
from src.savetax.ledger import Ledger, LedgerEntry
from src.savetax.workflow import body_hash, run_revise
from tests.unit.savetax.test_cli import (
    TOKEN, _evidence_rows, _evidence_transport, _write_dataset,
)
from tests.unit.savetax.test_cli_remediation import (
    _remediate_transports, _resp, _stage_remediation,
)
from tests.unit.savetax.test_workflow import (
    _Submissions, _approve, _body, _bundle, _event, _ledger, _now, _preview,
)


def test_replanning_revokes_existing_approval(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage_remediation(tmp_path, ["11111111"])
    read, write = _remediate_transports([], [], [("SID1", "11111111")])
    kwargs = {"environ": {"SAVETAX_TOKEN": TOKEN}, "transport": read, "write_transport": write}
    assert main(["remediate", "plan"], **kwargs) == 0
    challenge = next(line.split("challenge=")[1].split(" total=")[0]
                     for line in capsys.readouterr().out.splitlines() if line.startswith("revise challenge="))
    assert main(["remediate", "approve", "--kind", "revise", "--challenge", challenge, "--note", "yes"], **kwargs) == 0
    assert main(["remediate", "plan"], **kwargs) == 0
    assert main(["remediate", "run", "--kind", "revise", "--execute"], **kwargs) == 0
    entries = Ledger(Path("scratch/evidence/ledger.jsonl"), now=_now).entries()
    assert entries[-1].data["reasons"] == ["not_approved"]
    assert not any(e.kind == "revised" for e in entries)


def test_approve_rejects_stale_plan_revision(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage_remediation(tmp_path, ["11111111"])
    read, write = _remediate_transports([], [], [("SID1", "11111111")])
    kwargs = {"environ": {"SAVETAX_TOKEN": TOKEN}, "transport": read, "write_transport": write}
    assert main(["remediate", "plan"], **kwargs) == 0
    challenge = next(line.split("challenge=")[1].split(" total=")[0]
                     for line in capsys.readouterr().out.splitlines() if line.startswith("revise challenge="))

    def newer(url: str, headers: object, timeout: float) -> Any:
        response = read(url, headers, timeout)
        if url.endswith("/me/submissions"):
            payload = json.loads(response.body)
            payload["submissions"][0]["current_revision"] = 9
            return _resp(200, payload)
        return response

    assert main(["remediate", "approve", "--kind", "revise", "--challenge", challenge, "--note", "yes"],
                environ={"SAVETAX_TOKEN": TOKEN}, transport=newer, write_transport=write) == 2
    entries = Ledger(Path("scratch/evidence/ledger.jsonl"), now=_now).entries()
    assert not any(e.kind == "approved" for e in entries)


def test_revision_ledger_preserves_body_hash(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    client = _Submissions(_preview(action="revise_existing"))
    _approve(ledger, "11111111", body_hash(_body()), kind="revise")
    assert run_revise("SID1", "11111111", _body(), submission_client=client, ledger=ledger,  # type: ignore[arg-type]
                      now=_now(), ttl=timedelta(hours=6), execute=True,
                      bundle=_bundle(), event=_event(), fetch_price=lambda _: 1000) is not None
    assert _submitted_hashes(ledger)["11111111"] == body_hash(_body())
    assert next(e for e in ledger.entries() if e.kind == "revised").body_sha256 != body_hash(_body())
    ledger.append(LedgerEntry(_now(), "revised", "22222222", "SID2", "legacy-action-hash", None, {}))
    assert "22222222" not in _submitted_hashes(ledger)


@pytest.mark.parametrize("failure", [None, "withheld", "item_api"])
def test_owned_evidence_is_isolated_and_inaccessible_facts_pause(
    tmp_path: Path, failure: str | None,
) -> None:
    dataset = _write_dataset(tmp_path, _evidence_rows())
    calls: list[str] = []
    read = _evidence_transport(calls, item_state="claimed_by_me", fail=failure)

    def transport(url: str, headers: object, timeout: float) -> Any:
        response = read(url, headers, timeout)
        if failure == "withheld" and url.endswith("/items/11111111"):
            payload = json.loads(response.body)
            payload["snapshot"] = None
            payload["facts"] = []
            return _resp(200, payload)
        return response

    out = tmp_path / "evidence"
    assert main(["evidence", "--owned", "--out-dir", str(out), "--data-dir", str(dataset)],
                environ={"SAVETAX_TOKEN": TOKEN}, transport=transport) == (0 if failure is None else 3)
    assert not (out / "drafts").exists()
    assert not (out / "cross-bundles.jsonl").exists()
    if failure is None:
        assert (out / "remediation/drafts/11111111.json").exists()


def test_zero_line_bundle_does_not_reuse_stale_draft(tmp_path: Path) -> None:
    drafts = tmp_path / "drafts"
    path = tmp_path / "cross-bundles.jsonl"
    write_draft(_body(), drafts / "11111111.json")
    write_bundles([replace(_bundle(), lines=())], path)
    assert _load_rebuilt(drafts, path, _now()) == {}


def test_review_remediation_drafts_without_item(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _stage_remediation(tmp_path, ["11111111", "22222222"])
    read, write = _remediate_transports([], [], [("SID1", "11111111"), ("SID2", "22222222")])
    assert main(["review", "--kind", "revise", "--drafts-dir", "scratch/evidence/remediation/drafts"],
                environ={"SAVETAX_TOKEN": TOKEN}, transport=read, write_transport=write) == 0
    assert "APPROVE 2 4000" in capsys.readouterr().out
