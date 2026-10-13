"""Fresh-state and approval-expiry guards for action workflows."""

from __future__ import annotations

import io
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from src.savetax.errors import ConfigError
from src.savetax.workflow import action_review, body_hash, run_action_batch, run_revise, run_withdraw
from tests.unit.savetax.test_workflow import (
    _Submissions, _approve, _body, _bundle, _event, _ledger, _now, _preview,
)


@pytest.mark.parametrize("kind", ["review", "revise", "withdraw"])
def test_withdrawn_fresh_target_never_transmits(tmp_path: Path, kind: str) -> None:
    ledger = _ledger(tmp_path)
    client = _Submissions(_preview(action="revise_existing"))
    active = client.get_submission("SID1")
    client.my_submissions = [active]
    client.details = [replace(active, state="withdrawn")]
    def execute() -> None:
        if kind == "review":
            action_review([("11111111", _body(), _bundle())], client, "revise", out=io.StringIO())  # type: ignore[arg-type]
        elif kind == "revise":
            run_revise("SID1", "11111111", _body(), submission_client=client, ledger=ledger,  # type: ignore[arg-type]
                       now=_now(), ttl=timedelta(hours=6), execute=True)
        else:
            run_withdraw("SID1", "11111111", _body(), submission_client=client, ledger=ledger,  # type: ignore[arg-type]
                         now=_now(), ttl=timedelta(hours=6), execute=True)
    with pytest.raises(ConfigError, match="withdrawn"):
        execute()
    assert client.revises == client.withdraws == []


def test_batch_uses_live_clock_for_initial_approval_check(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    client = _Submissions(_preview(action="revise_existing"))
    _approve(ledger, "11111111", body_hash(_body()), kind="revise")
    done, skipped = run_action_batch(
        "revise", [("11111111", "SID1", _body(), _bundle())],
        submission_client=client, ledger=ledger, now=_now(), ttl=timedelta(hours=6),  # type: ignore[arg-type]
        execute=True, pause_seconds=0, stop_after_consecutive_suspect=2,
        fetch_price=lambda _: 1000, event_provider=_event,
        clock=lambda: _now() + timedelta(hours=7), out=io.StringIO(),
    )
    assert (done, skipped) == (0, 1)
    assert client.previews == 0
    assert client.revises == []
