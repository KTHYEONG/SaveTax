"""Invariant guards for the approval-gated submission workflow."""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.savetax.approval import build_review_set, challenge_for
from src.savetax.competition import EventInfo, ItemState, LookupResult, ParticipantStatus
from src.savetax.errors import ApiError, ConfigError, DataIntegrityError
from src.savetax.evidence import EvidenceBundle, Lane, LineProposal, PriceEvidence
from src.savetax.ledger import Ledger, LedgerEntry
from src.savetax.submissions import LintHint, LintLine, PreviewResult, Receipt, SubmissionDetail
from src.savetax.workflow import (
    body_hash,
    load_bundles,
    load_drafts,
    make_review_row,
    preflight,
    run_approve,
    run_review,
    run_revise,
    run_status,
    run_submit,
    run_withdraw,
    select_for_budget,
    track_lint,
)

import pytest


def _now() -> datetime:
    return datetime(2026, 10, 13, 12, tzinfo=UTC)


def _event() -> EventInfo:
    return EventInfo(
        dataset_release_id="release-1", ruleset_version="rules-v1", phase="open",
        opens_at=datetime(2026, 10, 1, tzinfo=UTC), closes_at=datetime(2026, 11, 1, tzinfo=UTC),
        evidence_format="v2", item_catalog="catalog",
    )


def _status(total: int | None = 100, per_day: int | None = 50) -> ParticipantStatus:
    return ParticipantStatus(handle="h", registered=True, claims_used=0, claims_today=0,
                             remaining_claims_total=total, remaining_claims_per_day=per_day)


def _evidence(url: str = "https://example.com/a", price: int = 1000) -> PriceEvidence:
    return PriceEvidence(source_url=url, quote="observed model M price tag",
                         observed_price_krw=price, observed_at=date(2026, 10, 1),
                         vat_included=True, shipping_included=True)


def _bundle(item: str = "11111111", total: int = 2000, url: str = "https://example.com/a") -> EvidenceBundle:
    line = LineProposal(
        fact_id="F1", benchmark_unit_price_krw=Decimal("1000"), comparison_basis_id="basis:대:vat",
        unit="대", tax_basis="vat_included", fact_unit_price_krw=Decimal("2000"), quantity=Decimal("2"),
        comparable_amount_krw=Decimal("4000"), request_date=date(2026, 10, 1),
        expected_krw=total, price_evidence=(_evidence(url),),
    )
    return EvidenceBundle(item_id=item, lane=Lane.CROSS_CONTRACT, title=f"title-{item}", claim="claim",
                          comparison_scope="scope", reproduction_notes="notes", lines=(line,), rejected=(),
                          caveats=(), review_required=False, generated_at=_now(), unit_notes=())


def _body(item: str = "11111111") -> dict[str, object]:
    from src.savetax.evidence import _line_to_draft

    return {"dataset_release_id": "release-1", "ruleset_version": "rules-v1",
            "canonical_product_id": item, "title": f"title-{item}",
            "lines": [_line_to_draft(_bundle(item).lines[0])]}


def _preview(total: int = 2000, ok: bool = True, action: str = "submit") -> PreviewResult:
    return PreviewResult(ok=ok, total_amount_krw=total, line_amounts_krw={}, next_action=action,
                         product_already_claimed=False, claimed_by_me=False,
                         existing_submission_id=None, current_revision=None)


class _Competition:
    def __init__(self, state: ItemState = ItemState.AVAILABLE, event: EventInfo | None = None) -> None:
        self._state = state
        self._event = event or _event()
        self.lookups = 0

    def lookup_items(self, ids: object) -> list[LookupResult]:
        self.lookups += 1
        return [LookupResult(input=i, kind="item", item_id=i, state=self._state) for i in ids]  # type: ignore[union-attr]

    def get_event(self) -> EventInfo:
        return self._event


class _Submissions:
    def __init__(self, preview: PreviewResult | None = None) -> None:
        self._preview = preview or _preview()
        self.totals: dict[str, int] = {}
        self.failing: set[str] = set()
        self.previews = 0
        self.submits: list[tuple[Mapping[str, object], str]] = []
        self.revises: list[str] = []
        self.withdraws: list[str] = []
        self.submit_impl: Callable[[Mapping[str, object], str], Receipt] | None = None
        self.details: list[SubmissionDetail] = []
        self.detail_calls = 0
        self.my_submissions: list[SubmissionDetail] = []

    def preview(self, body: Mapping[str, object]) -> PreviewResult:
        self.previews += 1
        item = body.get("canonical_product_id")
        base = self._preview
        if isinstance(item, str) and item in self.failing:
            return PreviewResult(ok=False, total_amount_krw=base.total_amount_krw,
                                 line_amounts_krw=base.line_amounts_krw, next_action=base.next_action,
                                 product_already_claimed=base.product_already_claimed,
                                 claimed_by_me=base.claimed_by_me,
                                 existing_submission_id=base.existing_submission_id,
                                 current_revision=base.current_revision)
        if isinstance(item, str) and item in self.totals:
            return PreviewResult(ok=base.ok, total_amount_krw=self.totals[item],
                                 line_amounts_krw=base.line_amounts_krw, next_action=base.next_action,
                                 product_already_claimed=base.product_already_claimed,
                                 claimed_by_me=base.claimed_by_me,
                                 existing_submission_id=base.existing_submission_id,
                                 current_revision=base.current_revision)
        return self._preview

    def submit(self, body: Mapping[str, object], *, idempotency_key: str) -> Receipt:
        self.submits.append((body, idempotency_key))
        if self.submit_impl is not None:
            return self.submit_impl(body, idempotency_key)
        return Receipt(submission_id="S" + str(body["canonical_product_id"]), claim_id=None, revision=1, submission_state="accepted",
                       admitted_at=None, amount_krw=2000, lint_status="clean", api_path=None)

    def get_submission(self, submission_id: str) -> SubmissionDetail:
        self.detail_calls += 1
        if self.details:
            return self.details[min(self.detail_calls - 1, len(self.details) - 1)]
        return SubmissionDetail(submission_id=submission_id, item_id="11111111", state="accepted",
                                revision=2, amount_krw=2000, held=False, lint_status="clean",
                                lint_lines=(), lint_hints=())

    def list_my_submissions(self) -> list[SubmissionDetail]:
        return self.my_submissions

    def resolve_request(self, key: str) -> Any:
        from src.savetax.submissions import RequestOutcome

        return RequestOutcome(key, "not_found", None)

    def revise(self, submission_id: str, body: Mapping[str, object], *, if_match: int, idempotency_key: str) -> Receipt:
        self.revises.append(submission_id)
        if self.submit_impl is not None:
            return self.submit_impl(body, idempotency_key)
        return Receipt(submission_id=submission_id, claim_id=None, revision=if_match + 1,
                       submission_state="accepted", admitted_at=None, amount_krw=2000,
                       lint_status="clean", api_path=None)

    def withdraw(self, submission_id: str, *, idempotency_key: str) -> Receipt:
        self.withdraws.append(submission_id)
        return Receipt(submission_id=submission_id, claim_id=None, revision=1, submission_state="withdrawn",
                       admitted_at=None, amount_krw=2000, lint_status="clean", api_path=None)


def _ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl", now=_now)


def _approve(ledger: Ledger, item: str, digest: str, kind: str = "submit") -> None:
    if kind in ("revise", "withdraw"):
        from src.savetax.workflow import action_hash

        digest = action_hash(kind, _Submissions().get_submission("SID1"), _body(item))
    ledger.append(LedgerEntry(at=_now(), kind="approved", item_id=item, submission_id=None,
                              body_sha256=digest, idempotency_key=None,
                              data={"kind": kind, "note": "yes", "total": 1}))


def _drafts(item: str = "11111111", total: int = 2000) -> list[tuple[str, Mapping[str, object], EvidenceBundle]]:
    return [(item, _body(item), _bundle(item, total))]


def _clean_detail(sid: str = "SID1", item: str = "11111111") -> SubmissionDetail:
    return SubmissionDetail(submission_id=sid, item_id=item, state="accepted", revision=1,
                            amount_krw=2000, held=False, lint_status="clean", lint_lines=(), lint_hints=())


def test_body_hash_is_canonical() -> None:
    assert body_hash({"b": 1, "a": 1}) == body_hash({"a": 1, "b": 1})
    assert body_hash({"a": 1}) != body_hash({"a": 2})
    with pytest.raises(DataIntegrityError):
        body_hash({"a": object()})


def test_preflight_allows_clean(tmp_path: Path) -> None:
    bundle = _bundle()
    plan = preflight(_body(), bundle, event=_event(), status=_status(), lookup_state=ItemState.AVAILABLE,
                     preview=_preview(), page_prices={"https://example.com/a": 1000},
                     reserve=5, now=_now(), approved=True)
    assert plan.allowed
    assert plan.reasons == ()


def test_preflight_release_change_refused() -> None:
    body = dict(_body())
    body["dataset_release_id"] = "other"
    plan = preflight(body, _bundle(), event=_event(), status=_status(), lookup_state=ItemState.AVAILABLE,
                     preview=_preview(), page_prices={"https://example.com/a": 1000},
                     reserve=5, now=_now())
    assert plan.reasons == ("release_changed",)


def test_preflight_window_closed() -> None:
    plan = preflight(_body(), _bundle(), event=_event(), status=_status(), lookup_state=ItemState.AVAILABLE,
                     preview=_preview(), page_prices={"https://example.com/a": 1000},
                     reserve=5, now=datetime(2026, 12, 1, tzinfo=UTC))
    assert "window_closed" in plan.reasons


def test_preflight_accepts_live_running_phase_and_rejects_other_phases() -> None:
    from dataclasses import replace

    def plan_for(phase: str) -> tuple[str, ...]:
        return preflight(
            _body(), _bundle(), event=replace(_event(), phase=phase), status=_status(),
            lookup_state=ItemState.AVAILABLE, preview=_preview(),
            page_prices={"https://example.com/a": 1000}, reserve=5, now=_now(),
        ).reasons

    assert "window_closed" not in plan_for("RUNNING")
    assert "window_closed" in plan_for("ENDED")


def test_preflight_claimed_and_mine() -> None:
    claimed = preflight(_body(), _bundle(), event=_event(), status=_status(), lookup_state=ItemState.CLAIMED,
                        preview=_preview(), page_prices={"https://example.com/a": 1000},
                        reserve=5, now=_now())
    assert claimed.reasons == ("not_available",)
    mine = preflight(_body(), _bundle(), event=_event(), status=_status(), lookup_state=ItemState.CLAIMED_BY_ME,
                     preview=_preview(), page_prices={"https://example.com/a": 1000},
                     reserve=5, now=_now())
    assert mine.reasons == ("already_mine",)


def test_preflight_preview_and_amount() -> None:
    bad_preview = preflight(_body(), _bundle(), event=_event(), status=_status(),
                            lookup_state=ItemState.AVAILABLE, preview=_preview(ok=False),
                            page_prices={"https://example.com/a": 1000}, reserve=5, now=_now())
    assert "preview_not_submittable" in bad_preview.reasons
    drift = preflight(_body(), _bundle(), event=_event(), status=_status(), lookup_state=ItemState.AVAILABLE,
                      preview=_preview(total=999), page_prices={"https://example.com/a": 1000},
                      reserve=5, now=_now())
    assert drift.reasons == ("amount_changed",)


def test_preflight_evidence_change_and_unreadable() -> None:
    changed = preflight(_body(), _bundle(), event=_event(), status=_status(), lookup_state=ItemState.AVAILABLE,
                        preview=_preview(), page_prices={"https://example.com/a": 5}, reserve=5, now=_now())
    assert changed.reasons == ("evidence_changed",)
    missing = preflight(_body(), _bundle(), event=_event(), status=_status(), lookup_state=ItemState.AVAILABLE,
                        preview=_preview(), page_prices={"https://example.com/a": None}, reserve=5, now=_now())
    assert missing.reasons == ("evidence_changed",)


def test_preflight_reserve_respected() -> None:
    plan = preflight(_body(), _bundle(), event=_event(), status=_status(total=5), lookup_state=ItemState.AVAILABLE,
                     preview=_preview(), page_prices={"https://example.com/a": 1000}, reserve=5, now=_now())
    assert plan.reasons == ("limit_reserve",)


def test_preflight_not_approved() -> None:
    plan = preflight(_body(), _bundle(), event=_event(), status=_status(), lookup_state=ItemState.AVAILABLE,
                     preview=_preview(), page_prices={"https://example.com/a": 1000},
                     reserve=5, now=_now(), approved=False)
    assert plan.reasons == ("not_approved",)


def test_dry_run_sends_nothing(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    receipts, code = run_submit(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                                submission_client=subs, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                                now=_now(), ttl=timedelta(hours=6), execute=False, out=io.StringIO())
    assert receipts == []
    assert code == 0
    assert subs.submits == []


def test_unapproved_body_never_sent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    _approve(ledger, "11111111", "old-hash")
    receipts, _ = run_submit(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                             submission_client=subs, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                             now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO())
    assert receipts == []
    assert subs.submits == []
    assert any(e.kind == "refused" for e in ledger.entries())


def test_expired_approval_refused(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    digest = body_hash(_body())
    ledger.append(LedgerEntry(at=_now() - timedelta(hours=7), kind="approved", item_id="11111111",
                              submission_id=None, body_sha256=digest, idempotency_key=None,
                              data={"kind": "submit", "note": "yes", "total": 1}))
    receipts, _ = run_submit(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                             submission_client=subs, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                             now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO())
    assert receipts == []
    assert subs.submits == []


def test_challenge_mismatch_blocks_approval(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    with pytest.raises(ConfigError):
        run_approve(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                    submission_client=subs, ledger=ledger, now=_now(),  # type: ignore[arg-type]
                    challenge="APPROVE 1 1 deadbeef", note="yes")
    assert ledger.entries() == []


def test_claimed_item_skipped_without_preview(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    _approve(ledger, "11111111", body_hash(_body()))
    receipts, _ = run_submit(_drafts(), event=_event(), status=_status(),  # type: ignore[arg-type]
                             competition=_Competition(state=ItemState.CLAIMED),
                             submission_client=subs, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                             now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO())
    assert receipts == []
    assert subs.previews == 0
    assert ledger.entries()[-1].kind == "refused"


def test_amount_drift_refused(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions(preview=_preview(total=1))
    _approve(ledger, "11111111", body_hash(_body()))
    receipts, _ = run_submit(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                             submission_client=subs, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                             now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO())
    assert receipts == []
    assert ledger.entries()[-1].data["reasons"] == ["amount_changed"]


def test_evidence_page_change_refused(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    _approve(ledger, "11111111", body_hash(_body()))
    receipts, _ = run_submit(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                             submission_client=subs, ledger=ledger, fetch_price=lambda _u: 5,  # type: ignore[arg-type]
                             now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO())
    assert receipts == []
    assert ledger.entries()[-1].data["reasons"] == ["evidence_changed"]


def test_unreadable_page_refused(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    _approve(ledger, "11111111", body_hash(_body()))
    receipts, _ = run_submit(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                             submission_client=subs, ledger=ledger, fetch_price=lambda _u: None,  # type: ignore[arg-type]
                             now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO())
    assert receipts == []
    assert subs.submits == []


def test_reserve_respected_in_submit(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    _approve(ledger, "11111111", body_hash(_body()))
    receipts, _ = run_submit(_drafts(), event=_event(), status=_status(total=5), competition=_Competition(),  # type: ignore[arg-type]
                             submission_client=subs, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                             now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO())
    assert receipts == []
    assert ledger.entries()[-1].data["reasons"] == ["limit_reserve"]


def test_submit_order_by_expected_total(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    subs.details = []
    drafts = [("11111111", _body("11111111"), _bundle("11111111", 100)),
              ("22222222", _body("22222222"), _bundle("22222222", 900)),
              ("33333333", _body("33333333"), _bundle("33333333", 500))]
    subs.totals = {"11111111": 100, "22222222": 900, "33333333": 500}
    for item, body, _ in drafts:
        _approve(ledger, item, body_hash(body))
    receipts, code = run_submit(drafts, event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                                submission_client=subs, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                                now=_now(), ttl=timedelta(hours=6), execute=True,
                                sleep=lambda _s: None, out=io.StringIO())
    assert code == 0
    assert len(receipts) == 3
    sent = [dict(b).get("canonical_product_id") for b, _ in subs.submits]
    assert sent == ["22222222", "33333333", "11111111"]


def test_restart_reuses_key(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    _approve(ledger, "11111111", body_hash(_body()))
    keys = iter(["fixed-key", "other-key"])
    receipts, _ = run_submit(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                             submission_client=subs, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                             now=_now(), ttl=timedelta(hours=6), execute=True,
                             sleep=lambda _s: None, new_key=lambda: next(keys), out=io.StringIO())
    assert receipts
    assert subs.submits[0][1] == "fixed-key"
    subs2 = _Submissions()
    receipts2, _ = run_submit(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                              submission_client=subs2, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                              now=_now(), ttl=timedelta(hours=6), execute=True,
                              sleep=lambda _s: None, new_key=lambda: next(keys), out=io.StringIO())
    assert receipts2
    assert subs2.submits[0][1] == "fixed-key"


def test_limit_error_stops_run(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()

    def fail(_b: Mapping[str, object], _k: str) -> Receipt:
        raise ApiError("competition", 403, "PARTICIPANT_LIMIT", "limit", False)

    subs.submit_impl = fail
    drafts = [("11111111", _body("11111111"), _bundle("11111111", 900)),
              ("22222222", _body("22222222"), _bundle("22222222", 100))]
    subs.totals = {"11111111": 900, "22222222": 100}
    for item, body, _ in drafts:
        _approve(ledger, item, body_hash(body))
    receipts, code = run_submit(drafts, event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                                submission_client=subs, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                                now=_now(), ttl=timedelta(hours=6), execute=True,
                                sleep=lambda _s: None, out=io.StringIO())
    assert receipts == []
    assert code == 2
    assert len(subs.submits) == 1


def test_two_suspects_stop_run(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    suspect = SubmissionDetail(submission_id="S", item_id="11111111", state="accepted", revision=1,
                               amount_krw=1, held=True, lint_status="suspect",
                               lint_lines=(LintLine(fact_id="F1", verdict="suspect", reason="price off"),),
                               lint_hints=(LintHint(fact_id=None, code="PRICE_MISMATCH", message="re-read"),))
    subs.details = [suspect]
    drafts = [("11111111", _body("11111111"), _bundle("11111111", 900)),
              ("22222222", _body("22222222"), _bundle("22222222", 800)),
              ("33333333", _body("33333333"), _bundle("33333333", 700))]
    subs.totals = {"11111111": 900, "22222222": 800, "33333333": 700}
    for item, body, _ in drafts:
        _approve(ledger, item, body_hash(body))
    out = io.StringIO()
    receipts, _ = run_submit(drafts, event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                             submission_client=subs, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                             now=_now(), ttl=timedelta(hours=6), execute=True, poll_seconds=1,
                             timeout_seconds=1, sleep=lambda _s: None, out=out)
    assert len(receipts) == 2
    assert len(subs.submits) == 2
    assert "PRICE_MISMATCH" in out.getvalue()


def test_lint_timeout_reported(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    subs.details = [SubmissionDetail(submission_id="S", item_id="11111111", state="accepted", revision=1,
                                     amount_krw=1, held=False, lint_status="pending",
                                     lint_lines=(), lint_hints=())]
    out = io.StringIO()
    final = track_lint(subs, ledger, item_id="11111111", submission_id="S", body_sha256="h",  # type: ignore[arg-type]
                       now=_now(), poll_seconds=7, timeout_seconds=7, sleep=lambda _s: None, out=out)
    assert final == "pending"
    assert ledger.entries()[-1].data["status"] == "pending"
    assert "pending" in out.getvalue()


def test_revision_needs_own_approval(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    _approve(ledger, "11111111", body_hash(_body()))
    assert run_revise("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                      now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO(),
                      bundle=_bundle(), event=_event(), fetch_price=lambda _: 1000) is None
    assert subs.revises == []


def test_revision_conflict_stops(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()

    def conflict(_b: Mapping[str, object], _k: str) -> Receipt:
        raise ApiError("competition", 409, "REVISION_CONFLICT", "conflict", False)

    subs.submit_impl = conflict
    subs._preview = _preview(action="revise_existing")
    _approve(ledger, "11111111", body_hash(_body()), kind="revise")
    assert run_revise("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                      now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO(),
                      bundle=_bundle(), event=_event(), fetch_price=lambda _: 1000) is None
    assert subs.detail_calls == 2
    assert len(subs.revises) == 1


def test_withdraw_needs_own_approval(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    assert run_withdraw("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                        now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO()) is None
    assert subs.withdraws == []


def test_quotes_never_printed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    out = io.StringIO()
    run_review(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
               submission_client=subs, ledger=ledger, now=_now(), out=out)  # type: ignore[arg-type]
    assert "observed model M price tag" not in out.getvalue()


def test_review_and_approve_round_trip(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    review = run_review(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                        submission_client=subs, ledger=ledger, now=_now(), out=io.StringIO())  # type: ignore[arg-type]
    assert review.challenge.startswith("APPROVE 1 2000 ")
    count = run_approve(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                        submission_client=subs, ledger=ledger, now=_now(),  # type: ignore[arg-type]
                        challenge=review.challenge, note="looks good")
    assert count == 1
    with pytest.raises(ConfigError):
        run_approve(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                    submission_client=subs, ledger=ledger, now=_now(),  # type: ignore[arg-type]
                    challenge=review.challenge, note="  ")


def test_review_excludes_and_defers(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    subs.failing = {"22222222"}
    subs.totals = {"11111111": 2000, "22222222": 2000}
    out = io.StringIO()
    drafts = [("11111111", _body("11111111"), _bundle("11111111", 2000)),
              ("22222222", _body("22222222"), _bundle("22222222", 2000))]
    review = run_review(drafts, event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                        submission_client=subs, ledger=ledger, now=_now(), out=out)  # type: ignore[arg-type]
    assert len(review.rows) == 1
    assert "excluded 22222222" in out.getvalue()
    ledger2 = _ledger(tmp_path / "sub")
    subs2 = _Submissions()
    subs2.totals = {"11111111": 900, "22222222": 100}
    out2 = io.StringIO()
    drafts = [("11111111", _body("11111111"), _bundle("11111111", 900)),
              ("22222222", _body("22222222"), _bundle("22222222", 100))]
    review = run_review(drafts, event=_event(), status=_status(total=6), competition=_Competition(),  # type: ignore[arg-type]
                        submission_client=subs2, ledger=ledger2, now=_now(), out=out2, reserve=5)  # type: ignore[arg-type]
    assert len(review.rows) == 1
    assert "deferred" in out2.getvalue()


def test_status_appends_new_lint(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    subs.my_submissions = [_clean_detail()]
    out = io.StringIO()
    details = run_status(subs, ledger, _now(), out=out)  # type: ignore[arg-type]
    assert len(details) == 1
    assert ledger.entries()[-1].kind == "lint"
    assert "SID1" in out.getvalue()
    before = len(ledger.entries())
    run_status(subs, ledger, _now(), out=io.StringIO())  # type: ignore[arg-type]
    assert len(ledger.entries()) == before


def test_revise_and_withdraw_execute(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    subs._preview = _preview(action="revise_existing")
    _approve(ledger, "11111111", body_hash(_body()), kind="revise")
    receipt = run_revise("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                         now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO(),
                         bundle=_bundle(), event=_event(), fetch_price=lambda _: 1000)
    assert receipt is not None
    assert any(entry.kind == "revised" for entry in ledger.entries())
    dry = run_revise("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                     now=_now(), ttl=timedelta(hours=6), execute=False, out=io.StringIO())
    assert dry is None
    ledger2 = _ledger(tmp_path / "w")
    subs2 = _Submissions()
    from src.savetax.workflow import body_hash as _bh

    _approve(ledger2, "11111111", _bh(_body()), kind="withdraw")
    withdrawn = run_withdraw("SID1", "11111111", _body(), submission_client=subs2, ledger=ledger2,  # type: ignore[arg-type]
                             now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO())
    assert withdrawn is not None
    assert ledger2.entries()[-1].kind == "withdrawn"
    assert run_withdraw("SID9", "11111111", _body(), submission_client=subs2, ledger=ledger2,  # type: ignore[arg-type]
                        now=_now(), ttl=timedelta(hours=6), execute=False, out=io.StringIO()) is None


def test_revise_product_mismatch(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    subs.details = [SubmissionDetail(submission_id="SID1", item_id="99999999", state="accepted", revision=1,
                                     amount_krw=1, held=False, lint_status="clean", lint_lines=(), lint_hints=())]
    _approve(ledger, "11111111", body_hash(_body()), kind="revise")
    assert run_revise("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                      now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO()) is None
    ledger2 = _ledger(tmp_path / "m2")
    subs2 = _Submissions()
    _approve(ledger2, "11111111", body_hash(_body("22222222")), kind="revise")
    assert run_revise("SID1", "11111111", _body("22222222"), submission_client=subs2, ledger=ledger2,  # type: ignore[arg-type]
                      now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO()) is None


def test_submit_network_and_pending_and_retryable(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    _approve(ledger, "11111111", body_hash(_body()))

    def net(_b: Mapping[str, object], _k: str) -> Receipt:
        from src.savetax.errors import NetworkError, NetworkStage

        raise NetworkError(NetworkStage.DNS_RESOLUTION, "h", "down")

    subs.submit_impl = net
    _, code = run_submit(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                         submission_client=subs, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                         now=_now(), ttl=timedelta(hours=6), execute=True,
                         sleep=lambda _s: None, out=io.StringIO())
    assert code == 3
    ledger2 = _ledger(tmp_path / "p")
    subs2 = _Submissions()
    _approve(ledger2, "11111111", body_hash(_body()))

    def pending(_b: Mapping[str, object], _k: str) -> Receipt:
        from src.savetax.errors import SubmissionPending

        raise SubmissionPending("k")

    subs2.submit_impl = pending
    _, code2 = run_submit(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                          submission_client=subs2, ledger=ledger2, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                          now=_now(), ttl=timedelta(hours=6), execute=True,
                          sleep=lambda _s: None, out=io.StringIO())
    assert code2 == 3
    ledger3 = _ledger(tmp_path / "r")
    subs3 = _Submissions()
    _approve(ledger3, "11111111", body_hash(_body()))

    def retryable(_b: Mapping[str, object], _k: str) -> Receipt:
        raise ApiError("competition", 500, "TRY_AGAIN", "x", True)

    subs3.submit_impl = retryable
    receipts3, code3 = run_submit(_drafts(), event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                                  submission_client=subs3, ledger=ledger3, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                                  now=_now(), ttl=timedelta(hours=6), execute=True,
                                  sleep=lambda _s: None, out=io.StringIO())
    assert receipts3 == []
    assert code3 == 2


def test_track_lint_suspect_and_clean(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    subs.details = [_clean_detail()]
    assert track_lint(subs, ledger, item_id="11111111", submission_id="SID1", body_sha256="h",  # type: ignore[arg-type]
                      now=_now(), sleep=lambda _s: None, out=io.StringIO()) == "clean"


def test_select_for_budget_and_row() -> None:
    bundle = _bundle()
    row = make_review_row("11111111", _body(), bundle, _preview(), "title-11111111")
    assert row.body_sha256 == body_hash(_body())
    selected, deferred = select_for_budget([row], None, 5)
    assert selected == [row]
    assert deferred == []
    selected2, deferred2 = select_for_budget([row], 5, 5)
    assert selected2 == []
    assert deferred2 == [row]
    assert challenge_for([row], "release-1")


def test_load_drafts_and_bundles(tmp_path: Path) -> None:
    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()
    (drafts_dir / "11111111.json").write_text(json.dumps(_body()), encoding="utf-8")
    (drafts_dir / "note.txt").write_text("ignore", encoding="utf-8")
    assert load_drafts(tmp_path / "missing") == []
    loaded = load_drafts(drafts_dir)
    assert loaded[0][0] == "11111111"
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "x.json").write_text("not json", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_drafts(bad_dir)
    bad_dir2 = tmp_path / "bad2"
    bad_dir2.mkdir()
    (bad_dir2 / "x.json").write_text("[1,2]", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_drafts(bad_dir2)
    bad_dir3 = tmp_path / "bad3"
    bad_dir3.mkdir()
    (bad_dir3 / "x.json").write_text(json.dumps({"title": "t"}), encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_drafts(bad_dir3)
    bundle_path = tmp_path / "bundles.jsonl"
    payload = {
        "item_id": "11111111", "lane": "cross_contract", "title": "title-11111111", "claim": "c",
        "comparison_scope": "s", "reproduction_notes": "r",
        "lines": [{"fact_id": "F1", "benchmark_unit_price_krw": "1000", "comparison_basis_id": "b",
                   "unit": "대", "tax_basis": "vat_included", "fact_unit_price_krw": "2000",
                   "quantity": "2", "comparable_amount_krw": "4000", "request_date": "2026-10-01",
                   "expected_krw": 2000,
                   "price_evidence": [{"source_url": "https://example.com/a", "quote": "q",
                                       "observed_price_krw": 1000, "observed_at": "2026-10-01",
                                       "vat_included": True, "shipping_included": True, "note": None}]}],
        "rejected": [{"fact_id": "F9", "reason": "x"}], "caveats": ["c1"], "review_required": False,
        "generated_at": _now().isoformat(), "expected_total_krw": 2000, "unit_notes": ["u1"],
    }
    bundle_path.write_text(json.dumps(payload) + "\n\n   \n", encoding="utf-8")
    bundles = load_bundles([bundle_path, tmp_path / "absent.jsonl"], now=_now())
    assert bundles["11111111"].expected_total_krw == 2000
    malformed = tmp_path / "mal.jsonl"
    malformed.write_text("nope\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_bundles([malformed], now=_now())
    not_list = tmp_path / "nl.jsonl"
    not_list.write_text(json.dumps({"item_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_bundles([not_list], now=_now())


def test_load_bundles_rejects_bad_shapes(tmp_path: Path) -> None:
    base: dict[str, Any] = {
        "item_id": "11111111", "lane": "cross_contract", "title": "t", "claim": "c",
        "comparison_scope": "s", "reproduction_notes": "r",
        "lines": [{"fact_id": "F1", "benchmark_unit_price_krw": "1000", "comparison_basis_id": "b",
                   "unit": "대", "tax_basis": "vat_included", "fact_unit_price_krw": "2000",
                   "quantity": "2", "comparable_amount_krw": "4000", "request_date": "2026-10-01",
                   "expected_krw": 2000,
                   "price_evidence": [{"source_url": "https://example.com/a", "quote": "q",
                                       "observed_price_krw": 1000, "observed_at": "2026-10-01",
                                       "vat_included": True, "shipping_included": True}]}],
        "rejected": [], "caveats": [], "review_required": False,
        "generated_at": _now().isoformat(), "unit_notes": [],
    }

    def _write(payload: Any) -> Path:
        path = tmp_path / f"b{abs(hash(json.dumps(payload, sort_keys=True, default=str)))}.jsonl"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path

    bad_lane = dict(base, lane="nope")
    with pytest.raises(DataIntegrityError):
        load_bundles([_write(bad_lane)], now=_now())
    bad_lines = dict(base, lines={})
    with pytest.raises(DataIntegrityError):
        load_bundles([_write(bad_lines)], now=_now())
    bad_evidence = json.loads(json.dumps(base))
    bad_evidence["lines"][0]["price_evidence"] = []
    with pytest.raises(DataIntegrityError):
        load_bundles([_write(bad_evidence)], now=_now())
    bad_price = json.loads(json.dumps(base))
    bad_price["lines"][0]["price_evidence"][0]["observed_price_krw"] = "x"
    with pytest.raises(DataIntegrityError):
        load_bundles([_write(bad_price)], now=_now())
    bad_expected = json.loads(json.dumps(base))
    bad_expected["lines"][0]["expected_krw"] = "x"
    with pytest.raises(DataIntegrityError):
        load_bundles([_write(bad_expected)], now=_now())
    bad_rejected = dict(base, rejected={})
    with pytest.raises(DataIntegrityError):
        load_bundles([_write(bad_rejected)], now=_now())
    bad_entry = json.loads(json.dumps(base))
    bad_entry["lines"] = [42]
    with pytest.raises(DataIntegrityError):
        load_bundles([_write(bad_entry)], now=_now())
    bad_item = json.loads(json.dumps(base))
    bad_item["lines"][0]["price_evidence"] = [42]
    with pytest.raises(DataIntegrityError):
        load_bundles([_write(bad_item)], now=_now())
    bad_observed = json.loads(json.dumps(base))
    bad_observed["lines"][0]["price_evidence"][0]["observed_at"] = 5
    with pytest.raises(DataIntegrityError):
        load_bundles([_write(bad_observed)], now=_now())
    bad_title = json.loads(json.dumps(base))
    del bad_title["title"]
    with pytest.raises(DataIntegrityError):
        load_bundles([_write(bad_title)], now=_now())
    with pytest.raises(DataIntegrityError):
        load_bundles([_write([1, 2])], now=_now())
    missing_dir = tmp_path / "unreadable.jsonl"
    missing_dir.mkdir()
    with pytest.raises(DataIntegrityError):
        load_bundles([missing_dir], now=_now())


def test_approve_kind_filter_and_limit(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    drafts = [("11111111", _body("11111111"), _bundle("11111111", 900)),
              ("22222222", _body("22222222"), _bundle("22222222", 100))]
    subs.totals = {"11111111": 900, "22222222": 100}
    review = build_review_set(
        [make_review_row(i, b, bundle, _preview(bundle.expected_total_krw), bundle.title) for i, b, bundle in drafts],
        _event(), _now(),
    )
    from src.savetax.workflow import action_hash
    from dataclasses import replace

    action_row = replace(review.rows[0], body_sha256=action_hash("revise", subs.get_submission("SID1"), _body()))
    subs.my_submissions = [subs.get_submission("SID1")]
    subs._preview = _preview(total=900, action="revise_existing")
    count = run_approve(drafts, event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                        submission_client=subs, ledger=ledger, now=_now(),  # type: ignore[arg-type]
                        challenge=challenge_for([action_row], "release-1"),
                        note="yes", kind="revise", item="11111111")
    assert count == 1
    with pytest.raises(ConfigError):
        run_approve(drafts, event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                    submission_client=subs, ledger=ledger, now=_now(),  # type: ignore[arg-type]
                    challenge=review.challenge, note="yes", kind="bogus")
    ledger2 = _ledger(tmp_path / "l2")
    subs._preview = _preview()
    count2 = run_approve(drafts, event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                         submission_client=subs, ledger=ledger2, now=_now(),  # type: ignore[arg-type]
                         challenge=challenge_for([review.rows[0]], "release-1"),
                         note="yes", limit=1)
    assert count2 == 1


def test_revise_other_api_error_reraises(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()

    def fail(_b: Mapping[str, object], _k: str) -> Receipt:
        raise ApiError("competition", 500, "TRY_AGAIN", "x", True)

    subs.submit_impl = fail
    subs._preview = _preview(action="revise_existing")
    _approve(ledger, "11111111", body_hash(_body()), kind="revise")
    with pytest.raises(ApiError):
        run_revise("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                   now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO(),
                   bundle=_bundle(), event=_event(), fetch_price=lambda _: 1000)


def test_submission_for_item_tracks_latest(tmp_path: Path) -> None:
    from src.savetax.workflow import submission_for_item

    ledger = _ledger(tmp_path)
    assert submission_for_item(ledger, "11111111") is None
    ledger.append(LedgerEntry(at=_now(), kind="submitted", item_id="11111111", submission_id="SID1",
                              body_sha256="h", idempotency_key="k", data={"submission_id": "SID1"}))
    ledger.append(LedgerEntry(at=_now(), kind="submitted", item_id="11111111", submission_id="SID2",
                              body_sha256="h", idempotency_key="k", data={"submission_id": "SID2"}))
    assert submission_for_item(ledger, "11111111") == "SID2"
    assert submission_for_item(ledger, "22222222") is None


def test_submit_limit_truncates_run(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    drafts = [("11111111", _body("11111111"), _bundle("11111111", 900)),
              ("22222222", _body("22222222"), _bundle("22222222", 100))]
    subs.totals = {"11111111": 900, "22222222": 100}
    for item, body, _ in drafts:
        _approve(ledger, item, body_hash(body))
    receipts, code = run_submit(drafts, event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                                submission_client=subs, ledger=ledger, fetch_price=lambda _u: 1000,  # type: ignore[arg-type]
                                now=_now(), ttl=timedelta(hours=6), execute=True, limit=1,
                                sleep=lambda _s: None, out=io.StringIO())
    assert code == 0
    assert len(receipts) == 1
    assert len(subs.submits) == 1


def _revise_event(phase: str = "RUNNING"):
    from dataclasses import replace

    return replace(_event(), phase=phase)


def test_revision_accepts_live_running_phase(tmp_path: Path) -> None:
    from src.savetax.workflow import run_revise

    ledger = _ledger(tmp_path)
    subs = _Submissions(preview=_preview(action="revise_existing"))
    _approve(ledger, "11111111", body_hash(_body()), kind="revise")
    receipt = run_revise("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                         now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO(),
                         bundle=_bundle(), event=_revise_event("RUNNING"), fetch_price=lambda _: 1000)
    assert receipt is not None


def test_revision_refuses_closed_phase(tmp_path: Path) -> None:
    from src.savetax.workflow import run_revise

    ledger = _ledger(tmp_path)
    subs = _Submissions(preview=_preview(action="revise_existing"))
    _approve(ledger, "11111111", body_hash(_body()), kind="revise")
    with pytest.raises(ConfigError):
        run_revise("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                   now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO(),
                   bundle=_bundle(), event=_revise_event("ENDED"), fetch_price=lambda _: 1000)
    assert subs.revises == []


def test_batch_dry_run_writes_nothing(tmp_path: Path) -> None:
    from src.savetax.workflow import run_action_batch

    ledger = _ledger(tmp_path)
    subs = _Submissions(preview=_preview(action="revise_existing"))
    targets = [("11111111", "SID1", _body("11111111"), _bundle("11111111")),
               ("22222222", "SID2", _body("22222222"), _bundle("22222222")),
               ("33333333", "SID3", _body("33333333"), _bundle("33333333"))]
    out = io.StringIO()
    done, skipped = run_action_batch("revise", targets, submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                                     now=_now(), ttl=timedelta(hours=6), execute=False,
                                     pause_seconds=0, stop_after_consecutive_suspect=2,
                                     fetch_price=lambda _: 1000, event_provider=_event,
                                     clock=_now, sleep=lambda _: None, out=out)
    assert (done, skipped) == (0, 0)
    assert out.getvalue().count("would-revise") == 3
    assert subs.revises == []


def test_batch_unapproved_skipped_not_fatal(tmp_path: Path) -> None:
    from src.savetax.workflow import action_hash, run_action_batch

    ledger = _ledger(tmp_path)
    subs = _Submissions(preview=_preview(action="revise_existing"))
    bodies = [_body("11111111"), _body("22222222"), _body("33333333")]
    bundles = [_bundle("11111111"), _bundle("22222222"), _bundle("33333333")]
    details = [SubmissionDetail(submission_id=f"SID{i+1}", item_id=item, state="accepted", revision=2,
                                amount_krw=2000, held=False, lint_status="clean",
                                lint_lines=(), lint_hints=())
               for i, item in enumerate(["11111111", "22222222", "33333333"])]
    subs.details = details
    for item, body, detail in [("11111111", bodies[0], details[0]), ("33333333", bodies[2], details[2])]:
        ledger.append(LedgerEntry(at=_now(), kind="approved", item_id=item, submission_id=None,
                                  body_sha256=action_hash("revise", detail, body), idempotency_key=None,
                                  data={"kind": "revise", "note": "yes", "total": 1}))
    targets = [(item, det.submission_id, body, bundle)
               for item, det, body, bundle in zip(["11111111", "22222222", "33333333"], details, bodies, bundles, strict=True)]
    done, skipped = run_action_batch("revise", targets, submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                                     now=_now(), ttl=timedelta(hours=6), execute=True,
                                     pause_seconds=0, stop_after_consecutive_suspect=2,
                                     fetch_price=lambda _: 1000, event_provider=_event,
                                     clock=_now, sleep=lambda _: None, out=io.StringIO())
    assert (done, skipped) == (2, 1)
    assert any(e.kind == "refused" for e in ledger.entries())


def test_batch_two_suspects_stop(tmp_path: Path) -> None:
    from src.savetax.workflow import action_hash, run_action_batch

    ledger = _ledger(tmp_path)
    subs = _Submissions(preview=_preview(action="revise_existing"))
    items = ["11111111", "22222222", "33333333"]
    bodies = [_body(i) for i in items]
    bundles = [_bundle(i) for i in items]
    details = [SubmissionDetail(submission_id=f"SID{i+1}", item_id=item, state="accepted", revision=2,
                                amount_krw=2000, held=False, lint_status="suspect",
                                lint_lines=(LintLine(fact_id="F1", verdict="suspect", reason="x"),),
                                lint_hints=()) for i, item in enumerate(items)]
    subs.details = details
    for item, body, detail in zip(items, bodies, details, strict=True):
        ledger.append(LedgerEntry(at=_now(), kind="approved", item_id=item, submission_id=None,
                                  body_sha256=action_hash("revise", detail, body), idempotency_key=None,
                                  data={"kind": "revise", "note": "yes", "total": 1}))
    targets = list(zip(items, [d.submission_id for d in details], bodies, bundles, strict=True))
    done, _ = run_action_batch("revise", targets, submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                               now=_now(), ttl=timedelta(hours=6), execute=True,
                               pause_seconds=0, stop_after_consecutive_suspect=2,
                               fetch_price=lambda _: 1000, event_provider=_event,
                               clock=_now, sleep=lambda _: None, out=io.StringIO())
    assert done == 2
    assert len(subs.revises) == 2


def test_batch_limit_error_stops(tmp_path: Path) -> None:
    from src.savetax.workflow import action_hash, run_action_batch

    ledger = _ledger(tmp_path)
    subs = _Submissions(preview=_preview(action="revise_existing"))

    def fail(_b: Mapping[str, object], _k: str) -> Receipt:
        raise ApiError("competition", 403, "REVISION_LIMIT", "limit", False)

    subs.submit_impl = fail
    items = ["11111111", "22222222"]
    bodies = [_body(i) for i in items]
    bundles = [_bundle(i) for i in items]
    details = [SubmissionDetail(submission_id=f"SID{i+1}", item_id=item, state="accepted", revision=2,
                                amount_krw=2000, held=False, lint_status="clean",
                                lint_lines=(), lint_hints=()) for i, item in enumerate(items)]
    subs.details = details
    for item, body, detail in zip(items, bodies, details, strict=True):
        ledger.append(LedgerEntry(at=_now(), kind="approved", item_id=item, submission_id=None,
                                  body_sha256=action_hash("revise", detail, body), idempotency_key=None,
                                  data={"kind": "revise", "note": "yes", "total": 1}))
    targets = list(zip(items, [d.submission_id for d in details], bodies, bundles, strict=True))
    done, _ = run_action_batch("revise", targets, submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                               now=_now(), ttl=timedelta(hours=6), execute=True,
                               pause_seconds=0, stop_after_consecutive_suspect=2,
                               fetch_price=lambda _: 1000, event_provider=_event,
                               clock=_now, sleep=lambda _: None, out=io.StringIO())
    assert done == 0
    assert len(subs.revises) == 1


def test_batch_withdraw_sends_no_body(tmp_path: Path) -> None:
    from src.savetax.workflow import action_hash, run_action_batch

    ledger = _ledger(tmp_path)
    subs = _Submissions()
    items = ["11111111", "22222222"]
    bodies = [{"canonical_product_id": i} for i in items]
    details = [SubmissionDetail(submission_id=f"SID{i+1}", item_id=item, state="accepted", revision=1,
                                amount_krw=2000, held=False, lint_status="clean",
                                lint_lines=(), lint_hints=()) for i, item in enumerate(items)]
    subs.details = details
    for item, body, detail in zip(items, bodies, details, strict=True):
        ledger.append(LedgerEntry(at=_now(), kind="approved", item_id=item, submission_id=None,
                                  body_sha256=action_hash("withdraw", detail, body), idempotency_key=None,
                                  data={"kind": "withdraw", "note": "yes", "total": 1}))
    targets = [(item, det.submission_id, body, None) for item, det, body in zip(items, details, bodies, strict=True)]
    done, _ = run_action_batch("withdraw", targets, submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                               now=_now(), ttl=timedelta(hours=6), execute=True,
                               pause_seconds=0, stop_after_consecutive_suspect=2,
                               fetch_price=lambda _: 1000, event_provider=_event,
                               clock=_now, sleep=lambda _: None, out=io.StringIO())
    assert done == 2
    assert subs.withdraws == ["SID1", "SID2"]


def test_approve_multiple_without_item(tmp_path: Path) -> None:
    subs = _Submissions()
    subs.my_submissions = [subs.get_submission("SID1")]
    subs._preview = _preview(total=900, action="revise_existing")
    drafts = [("11111111", _body("11111111"), _bundle("11111111", 900))]
    from src.savetax.workflow import run_approve

    with pytest.raises(ConfigError, match="challenge mismatch"):
        run_approve(drafts * 2, event=_event(), status=_status(), competition=_Competition(),  # type: ignore[arg-type]
                    submission_client=subs, ledger=_ledger(tmp_path), now=_now(),  # type: ignore[arg-type]
                    challenge="APPROVE 0 0 deadbeef", note="yes", kind="revise")


def test_action_review_withdraw_product_mismatch() -> None:
    import io
    from src.savetax.workflow import action_review
    import pytest
    from src.savetax.errors import ConfigError
    subs = _Submissions()
    subs.my_submissions = [_clean_detail()]
    drafts = [("11111111", {"canonical_product_id": "other"}, _bundle("11111111", 2000))]
    with pytest.raises(ConfigError):
        action_review(drafts, subs, "withdraw", out=io.StringIO())  # type: ignore[arg-type]


def test_withdraw_closed_phase_raises(tmp_path) -> None:
    from dataclasses import replace
    from src.savetax.workflow import run_withdraw
    import pytest
    from src.savetax.errors import ConfigError
    ledger = _ledger(tmp_path)
    subs = _Submissions()
    _approve(ledger, "11111111", __import__("src.savetax.workflow", fromlist=["action_hash"]).action_hash("withdraw", subs.get_submission("SID1"), _body()), kind="withdraw")
    with pytest.raises(ConfigError):
        run_withdraw("SID1", "11111111", _body(), submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                     now=_now(), ttl=timedelta(hours=6), execute=True, out=io.StringIO(),
                     event=replace(_event(), phase="ENDED"), clock=_now)


def test_batch_invalid_kind_raises(tmp_path) -> None:
    from src.savetax.workflow import run_action_batch
    import pytest
    from src.savetax.errors import ConfigError
    with pytest.raises(ConfigError):
        run_action_batch("bogus", [], submission_client=_Submissions(), ledger=_ledger(tmp_path),  # type: ignore[arg-type]
                         now=_now(), ttl=timedelta(hours=6), execute=True, pause_seconds=0,
                         stop_after_consecutive_suspect=2, fetch_price=lambda _: 1000,
                         event_provider=_event, clock=_now, sleep=lambda _: None, out=io.StringIO())


def test_batch_none_bundle_skipped_and_pending_stops(tmp_path) -> None:
    from src.savetax.errors import SubmissionPending
    from src.savetax.workflow import run_action_batch
    ledger = _ledger(tmp_path)
    subs = _Submissions(preview=_preview(action="revise_existing"))
    targets = [("11111111", "SID1", _body("11111111"), None)]
    done, skipped = run_action_batch("revise", targets, submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                                     now=_now(), ttl=timedelta(hours=6), execute=True, pause_seconds=0,
                                     stop_after_consecutive_suspect=2, fetch_price=lambda _: 1000,
                                     event_provider=_event, clock=_now, sleep=lambda _: None, out=io.StringIO())
    assert (done, skipped) == (0, 1)
    def boom(body, key):
        raise SubmissionPending(key)
    subs.submit_impl = boom
    from src.savetax.workflow import action_hash
    bodies = [_body("11111111")]
    bundles = [_bundle("11111111")]
    detail = SubmissionDetail(submission_id="SID1", item_id="11111111", state="accepted", revision=2,
                              amount_krw=2000, held=False, lint_status="clean", lint_lines=(), lint_hints=())
    subs.details = [detail]
    ledger2 = _ledger(tmp_path / "p2")
    ledger2.append(LedgerEntry(at=_now(), kind="approved", item_id="11111111", submission_id=None,
                               body_sha256=action_hash("revise", detail, bodies[0]), idempotency_key=None,
                               data={"kind": "revise", "note": "yes", "total": 1}))
    done2, _ = run_action_batch("revise", [("11111111", "SID1", bodies[0], bundles[0])], submission_client=subs, ledger=ledger2,  # type: ignore[arg-type]
                                now=_now(), ttl=timedelta(hours=6), execute=True, pause_seconds=0,
                                stop_after_consecutive_suspect=2, fetch_price=lambda _: 1000,
                                event_provider=_event, clock=_now, sleep=lambda _: None, out=io.StringIO())
    assert done2 == 0


def test_batch_retryable_error_skipped_and_fallback_status(tmp_path) -> None:
    from src.savetax.workflow import run_action_batch, action_hash
    ledger = _ledger(tmp_path)
    subs = _Submissions(preview=_preview(action="revise_existing"))
    def retry(body, key):
        raise ApiError("competition", 500, "TRY_AGAIN", "x", True)
    subs.submit_impl = retry
    items = ["11111111"]
    bodies = [_body(i) for i in items]
    bundles = [_bundle(i) for i in items]
    details = [SubmissionDetail(submission_id="SID1", item_id="11111111", state="accepted", revision=2,
                                amount_krw=2000, held=False, lint_status="clean", lint_lines=(), lint_hints=())]
    subs.details = details
    for item, body, detail in zip(items, bodies, details, strict=True):
        ledger.append(LedgerEntry(at=_now(), kind="approved", item_id=item, submission_id=None,
                                  body_sha256=action_hash("revise", detail, body), idempotency_key=None,
                                  data={"kind": "revise", "note": "yes", "total": 1}))
    targets = list(zip(items, ["SID1"], bodies, bundles, strict=True))
    done, skipped = run_action_batch("revise", targets, submission_client=subs, ledger=ledger,  # type: ignore[arg-type]
                                     now=_now(), ttl=timedelta(hours=6), execute=True, pause_seconds=0,
                                     stop_after_consecutive_suspect=2, fetch_price=lambda _: 1000,
                                     event_provider=_event, clock=_now, sleep=lambda _: None, out=io.StringIO())
    assert (done, skipped) == (0, 1)
