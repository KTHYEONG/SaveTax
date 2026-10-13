"""Adversarial guards for approval causality, replay, and evidence identity."""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from src.savetax.approval import build_review_set, verify_approval
from src.savetax.config import ApprovalConfig
from src.savetax.errors import ConfigError, DataIntegrityError
from src.savetax.evidence import write_bundles
from src.savetax.ledger import Ledger, LedgerEntry
from src.savetax.submissions import RequestOutcome
from src.savetax.workflow import (
    action_hash, action_review, bind_bundle, body_hash, load_bundles, preflight,
    run_approve, run_review, run_revise, run_status, run_submit, run_withdraw,
)
from tests.unit.savetax.test_workflow import (
    _approve, _body, _bundle, _Competition, _drafts, _event, _ledger,
    _now, _preview, _status, _Submissions,
)


def _submit(ledger: Ledger, subs: Any, **over: Any) -> Any:
    kwargs = {"event": _event(), "status": _status(), "competition": _Competition(), "submission_client": subs,
              "ledger": ledger, "fetch_price": lambda _: 1000, "now": _now(), "ttl": timedelta(hours=6),
              "execute": True, "sleep": lambda _: None, "out": io.StringIO()}
    kwargs.update(over)
    return run_submit(kwargs.pop("drafts", _drafts()), **kwargs)


def test_reserve_is_consumed_across_the_whole_run(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    drafts = _drafts() + _drafts("22222222")
    for item, body, _ in drafts:
        _approve(ledger, item, body_hash(body))
    receipts, code = _submit(ledger, subs, drafts=drafts, status=_status(total=6))
    assert len(receipts) == 1
    assert len(subs.submits) == 1
    assert code == 2
    assert ledger.entries()[-1].data["reasons"] == ["limit_reserve"]


def test_missing_page_price_fails_closed() -> None:
    plan = preflight(_body(), _bundle(), event=_event(), status=_status(), lookup_state=_Competition()._state,
                     preview=_preview(), page_prices={}, reserve=5, now=_now())
    assert plan.reasons == ("evidence_changed",)


def test_approval_is_rechecked_after_page_fetch(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    _approve(ledger, "11111111", body_hash(_body()))
    moments = iter([_now(), _now() + timedelta(hours=7)])

    class Competition(_Competition):
        def get_me(self) -> Any:
            return _status()

    _, code = _submit(ledger, subs, competition=Competition(), clock=lambda: next(moments))
    assert code == 2
    assert subs.submits == []
    assert "not_approved" in ledger.entries()[-1].data["reasons"]


def test_refusal_at_same_timestamp_revokes_approval(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    digest = body_hash(_body())
    _approve(ledger, "11111111", digest)
    ledger.append(LedgerEntry(_now(), "refused", "11111111", None, digest, None, {"reasons": ["x"]}))
    assert not verify_approval(ledger, "11111111", digest, _now(), timedelta(hours=6))
    _approve(ledger, "11111111", digest)
    assert verify_approval(ledger, "11111111", digest, _now(), timedelta(hours=6))


@pytest.mark.parametrize("state", ["receipt", "pending", "not_found"])
def test_crash_recovery_resolves_before_claim_lookup(tmp_path: Path, state: str) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    _approve(ledger, "11111111", body_hash(_body()))
    key = ledger.key_for("submit", "11111111", body_hash(_body()), lambda: "stable")
    receipt = subs.submit(_body(), idempotency_key=key)
    subs.submits.clear()
    subs.resolve_request = lambda k: RequestOutcome(k, state, receipt if state == "receipt" else None)  # type: ignore[method-assign]
    competition = _Competition()
    receipts, code = _submit(ledger, subs, competition=competition)
    if state == "receipt":
        assert receipts == []
        assert subs.submits == []
        assert competition.lookups == 0
        assert len([entry for entry in ledger.entries() if entry.kind == "submitted"]) == 1
        _submit(ledger, subs, competition=competition)
        assert len([entry for entry in ledger.entries() if entry.kind == "submitted"]) == 1
    elif state == "pending":
        assert code == 3
        assert competition.lookups == 0
        assert subs.submits == []
    else:
        assert code == 0
        assert subs.submits[0][1] == key


def test_distinct_requests_cannot_count_one_receipt_twice(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    fixed = subs.submit(_body(), idempotency_key="k")
    subs.submits.clear()
    subs.submit_impl = lambda b, k: fixed
    drafts = _drafts() + _drafts("22222222")
    for item, body, _ in drafts:
        _approve(ledger, item, body_hash(body))
    with pytest.raises(DataIntegrityError, match="duplicate receipt"):
        _submit(ledger, subs, drafts=drafts)


@pytest.mark.parametrize("change", ["item", "empty", "shape", "fact", "duplicate", "price"])
def test_bundle_binding_rejects_draft_drift(change: str) -> None:
    body = _body()
    lines: Any = body["lines"]
    if change == "item":
        body["canonical_product_id"] = "other"
    elif change == "empty":
        body["lines"] = []
    elif change == "shape":
        body["lines"] = [42]
    elif change == "fact":
        lines[0]["fact_id"] = "unknown"
    elif change == "duplicate":
        lines.append(lines[0])
    else:
        lines[0]["price_evidence"][0]["observed_price_krw"] = "999"
    with pytest.raises(DataIntegrityError):
        bind_bundle(body, _bundle())


def test_review_limit_is_applied_after_amount_ordering_and_daily_budget(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    subs.totals = {"11111111": 100, "22222222": 2000, "33333333": 1000}
    drafts = _drafts(total=100) + _drafts("22222222", 2000) + _drafts("33333333", 1000)
    review = run_review(drafts, event=_event(), status=_status(per_day=6), competition=_Competition(),
                        submission_client=subs, ledger=ledger, now=_now(), limit=2, out=io.StringIO())  # type: ignore[arg-type]
    assert [row.item_id for row in review.rows] == ["22222222"]


@pytest.mark.parametrize("kind", ["revise", "withdraw"])
def test_action_approval_round_trip_is_bound_to_target_revision(tmp_path: Path, kind: str) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions(_preview(action="revise_existing"))
    subs.my_submissions = [subs.get_submission("SID1")]
    rows, targets = action_review(_drafts(), subs, kind, out=io.StringIO())  # type: ignore[arg-type]
    challenge = build_review_set(rows, _event(), _now()).challenge
    count = run_approve(_drafts(), event=_event(), status=_status(total=0), competition=_Competition(),
                        submission_client=subs, ledger=ledger, now=_now(), challenge=challenge,
                        note="explicit action approval", kind=kind, item="11111111", out=io.StringIO())  # type: ignore[arg-type]
    assert count == 1
    assert ledger.entries()[-1].submission_id == "SID1"
    assert rows[0].body_sha256 == action_hash(kind, targets["11111111"], _body())
    assert action_hash(kind, replace(targets["11111111"], revision=3), _body()) != rows[0].body_sha256
    if kind == "revise":
        receipt = run_revise("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,
                             now=_now(), ttl=timedelta(hours=6), execute=True, bundle=_bundle(),
                             event=_event(), fetch_price=lambda _: 1000, out=io.StringIO())  # type: ignore[arg-type]
    else:
        receipt = run_withdraw("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,
                               now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO())  # type: ignore[arg-type]
    assert receipt is not None


@pytest.mark.parametrize("case", ["missing", "wrong_item", "bad_preview"])
def test_action_review_fails_closed(case: str) -> None:
    subs = _Submissions()
    if case != "missing":
        subs.my_submissions = [subs.get_submission("SID1")]
    if case == "wrong_item":
        subs.details = [replace(subs.get_submission("SID1"), item_id="other")]
    with pytest.raises(ConfigError):
        action_review(_drafts(), subs, "revise", out=io.StringIO())  # type: ignore[arg-type]


def test_action_approval_requires_item(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="requires --item"):
        run_approve([], event=_event(), status=_status(), competition=_Competition(),
                    submission_client=_Submissions(), ledger=_ledger(tmp_path), now=_now(),
                    challenge="irrelevant", note="yes", kind="withdraw")  # type: ignore[arg-type]


@pytest.mark.parametrize("case", ["preview", "missing", "page", "release", "product", "body"])
def test_revision_guards_and_no_write(tmp_path: Path, case: str) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions(_preview(action="revise_existing"))
    body = _body()
    detail = subs.get_submission("SID1")
    if case == "body":
        body["canonical_product_id"] = "other"
    if case == "product":
        detail = replace(detail, item_id="other")
        subs.details = [detail]
    digest = action_hash("revise", detail, body)
    ledger.append(LedgerEntry(_now(), "approved", "11111111", "SID1", digest, None, {"kind": "revise", "note": "yes"}))
    kwargs: dict[str, Any] = {"bundle": _bundle(), "event": _event(), "fetch_price": lambda _: 1000}
    if case == "preview":
        subs._preview = _preview(ok=False)
    if case == "missing":
        kwargs["bundle"] = None
    if case == "page":
        kwargs["fetch_price"] = lambda _: None
    if case == "release":
        kwargs["event"] = replace(_event(), dataset_release_id="new")
    if case in ("product", "body"):
        assert run_revise("SID1", "11111111", body, submission_client=subs, ledger=ledger,
                          now=_now(), ttl=timedelta(hours=6), execute=True, **kwargs) is None
    else:
        with pytest.raises(ConfigError):
            run_revise("SID1", "11111111", body, submission_client=subs, ledger=ledger,
                       now=_now(), ttl=timedelta(hours=6), execute=True, **kwargs)
    assert subs.revises == []


def test_withdraw_unknown_identity_refused(tmp_path: Path) -> None:
    subs = _Submissions()
    subs.details = [replace(subs.get_submission("SID1"), item_id=None)]
    with pytest.raises(ConfigError, match="product mismatch"):
        run_withdraw("SID1", "11111111", _body(), submission_client=subs, ledger=_ledger(tmp_path),
                     now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO())  # type: ignore[arg-type]
    assert subs.withdraws == []


def test_status_records_return_to_earlier_lint_state(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    clean = subs.get_submission("SID1")
    for detail in (clean, replace(clean, lint_status="suspect"), clean):
        subs.my_submissions = [detail]
        run_status(subs, ledger, _now(), out=io.StringIO())  # type: ignore[arg-type]
    assert [e.data["status"] for e in ledger.entries()] == ["clean", "suspect", "clean"]


@pytest.mark.parametrize("mutation", ["date", "decimal", "boolean", "amount", "duplicate", "field", "metadata"])
def test_bundle_values_fail_closed(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "b.jsonl"
    write_bundles([_bundle()], path)
    data = json.loads(path.read_text())
    line = data["lines"][0]
    if mutation == "date":
        line["request_date"] = "impossible"
    elif mutation == "decimal":
        line["quantity"] = "NaN"
    elif mutation == "boolean":
        line["price_evidence"][0]["vat_included"] = "false"
    elif mutation == "amount":
        line["expected_krw"] = 900000
    elif mutation == "field":
        line["fact_id"] = 42
    elif mutation == "metadata":
        data["caveats"] = "not a list"
    else:
        data["lines"].append(line)
    path.write_text(json.dumps(data))
    with pytest.raises(DataIntegrityError):
        load_bundles([path], now=_now())


def test_conflicting_bundle_sources_are_rejected(tmp_path: Path) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_bundles([_bundle()], first)
    write_bundles([replace(_bundle(), title="different")], second)
    with pytest.raises(DataIntegrityError, match="conflicting"):
        load_bundles([first, second], now=_now())


@pytest.mark.parametrize("data", [{"quote": "source text"}, {"note": "savetax_private-pat"}, {"note": {"quote": "secret"}}])
def test_ledger_rejects_sensitive_or_nested_metadata(tmp_path: Path, data: dict[str, Any]) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(DataIntegrityError):
        ledger.append(LedgerEntry(_now(), "approved", "11111111", None, "h", None, data))
    assert ledger.entries() == []


def test_ledger_naive_timestamp_and_parent_failure(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(DataIntegrityError):
        ledger.append(LedgerEntry(_now().replace(tzinfo=None), "approved", "11111111", None, "h", None, {}))
    occupied = tmp_path / "file"
    occupied.touch()
    bad = Ledger(occupied / "ledger.jsonl", now=_now)
    with pytest.raises(DataIntegrityError):
        bad.append(LedgerEntry(_now(), "drafted", None, None, None, None, {}))
    with pytest.raises(DataIntegrityError):
        bad.key_for("submit", "11111111", "h", lambda: "key")


def test_concurrent_key_creation_reuses_one_durable_key(tmp_path: Path) -> None:
    def create(index: int) -> str:
        return _ledger(tmp_path).key_for("submit", "11111111", "h", lambda: f"key-{index}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        keys = list(pool.map(create, range(8)))
    assert len(set(keys)) == 1
    assert len(_ledger(tmp_path).entries()) == 1


@pytest.mark.parametrize("name", ["ttl_hours", "reserve_claims", "stop_after_consecutive_suspect"])
def test_approval_integer_config_rejects_boolean(name: str) -> None:
    with pytest.raises(ConfigError):
        ApprovalConfig(**{name: True})


def test_invalid_utf8_body_hash_fails_closed() -> None:
    with pytest.raises(DataIntegrityError):
        body_hash({"title": "\ud800"})


@pytest.mark.parametrize(("field", "value"), [("kind", 5), ("at", "impossible"), ("item_id", 42)])
def test_complete_ledger_schema_rejects_invalid_fields(tmp_path: Path, field: str, value: Any) -> None:
    data: dict[str, Any] = {"at": _now().isoformat(), "kind": "drafted", "item_id": "11111111",
                            "submission_id": None, "body_sha256": None, "idempotency_key": None, "data": {}}
    data[field] = value
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(data))
    with pytest.raises(DataIntegrityError):
        Ledger(path, now=_now).entries()


def test_revision_rechecks_approval_after_freshness(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions(_preview(action="revise_existing"))
    _approve(ledger, "11111111", body_hash(_body()), "revise")
    with pytest.raises(ConfigError, match="expired"):
        run_revise("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,
                   now=_now(), ttl=timedelta(hours=6), execute=True, bundle=_bundle(),
                   event=_event(), fetch_price=lambda _: 1000, clock=lambda: _now() + timedelta(hours=7))  # type: ignore[arg-type]
    assert subs.revises == []


def test_nested_drafts_match_their_evidence_lane(tmp_path: Path) -> None:
    from src.savetax.cli import _workflow_inputs
    from src.savetax.evidence import Lane, _line_to_draft

    evidence = tmp_path / "evidence"
    nested = evidence / "drafts" / "retail"
    nested.mkdir(parents=True)
    cross = _bundle()
    retail_line = replace(cross.lines[0], benchmark_unit_price_krw=cross.lines[0].benchmark_unit_price_krw / 2,
                          expected_krw=3000)
    retail = replace(cross, lane=Lane.RETAIL, lines=(retail_line,))
    write_bundles([cross], evidence / "cross-bundles.jsonl")
    write_bundles([retail], evidence / "retail-bundles.jsonl")
    (evidence / "drafts" / "11111111.json").write_text(json.dumps(_body()))
    body = _body()
    body["lines"] = [_line_to_draft(retail_line)]
    (nested / "11111111.json").write_text(json.dumps(body))
    selected = _workflow_inputs(evidence / "drafts", evidence, _now())
    assert len(selected) == 1
    assert selected[0][2].lane == Lane.RETAIL
    assert selected[0][2].expected_total_krw == 3000
    body["lines"] = []
    from src.savetax.workflow import matching_bundle

    with pytest.raises(DataIntegrityError):
        matching_bundle(body, {"other": _bundle("22222222"), cross.item_id: cross})


def test_lint_deadline_includes_network_time(tmp_path: Path) -> None:
    from src.savetax.workflow import track_lint

    subs = _Submissions()
    moment = [0.0]
    pending = replace(subs.get_submission("SID1"), lint_status="pending")

    def read(sid: str) -> Any:
        moment[0] += 600.0
        return pending

    subs.get_submission = read  # type: ignore[method-assign]
    sleeps: list[float] = []
    status = track_lint(subs, _ledger(tmp_path), item_id="11111111", submission_id="SID1",
                        body_sha256="h", now=_now(), sleep=sleeps.append,
                        clock=lambda: moment[0], out=io.StringIO())  # type: ignore[arg-type]
    assert status == "pending"
    assert moment[0] == 600.0
    assert all(delay == 0 for delay in sleeps)


def test_recovered_suspect_pauses_and_bad_receipt_is_rejected(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    _approve(ledger, "11111111", body_hash(_body()))
    ledger.key_for("submit", "11111111", body_hash(_body()), lambda: "recovery")
    receipt = subs.submit(_body(), idempotency_key="recovery")
    subs.resolve_request = lambda k: RequestOutcome(k, "receipt", receipt)  # type: ignore[method-assign]
    subs.details = [replace(subs.get_submission("SID1"), lint_status="suspect")]
    assert _submit(ledger, subs, stop_after_consecutive_suspect=1)[1] == 2
    subs.resolve_request = lambda k: RequestOutcome(k, "receipt", replace(receipt, submission_state="withdrawn"))  # type: ignore[method-assign]
    with pytest.raises(DataIntegrityError, match="unexpected recovered"):
        _submit(ledger, subs)
