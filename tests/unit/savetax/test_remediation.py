"""Invariant guards for remediation of already-submitted items."""

from __future__ import annotations

import json
import stat
from datetime import date
from decimal import Decimal
from pathlib import Path

from src.savetax.evidence import EvidenceBundle, Lane, LineProposal, PriceEvidence
from src.savetax.remediation import RemediationAction, plan_remediation, write_plan
from src.savetax.submissions import PreviewResult, SubmissionDetail
from src.savetax.workflow import body_hash


def _evidence(price: int = 1000) -> PriceEvidence:
    return PriceEvidence(source_url="https://example.com/a", quote="q tag",
                         observed_price_krw=price, observed_at=date(2026, 10, 1),
                         vat_included=True, shipping_included=True)


def _bundle(item: str = "11111111", total: int = 2000, price: int = 1000) -> EvidenceBundle:
    from datetime import datetime, UTC

    line = LineProposal(
        fact_id="F1", benchmark_unit_price_krw=Decimal(str(price)), comparison_basis_id="b",
        unit="대", tax_basis="vat_included", fact_unit_price_krw=Decimal("2000"),
        quantity=Decimal("2"), comparable_amount_krw=Decimal("4000"),
        request_date=date(2026, 10, 1), expected_krw=total, price_evidence=(_evidence(price),),
    )
    return EvidenceBundle(item_id=item, lane=Lane.CROSS_CONTRACT, title=f"t-{item}", claim="c",
                          comparison_scope="s", reproduction_notes="r", lines=(line,),
                          rejected=(), caveats=(), review_required=False,
                          generated_at=datetime(2026, 10, 13, 12, tzinfo=UTC), unit_notes=())


def _body(item: str = "11111111", price: str = "1000") -> dict[str, object]:
    return {"dataset_release_id": "r", "ruleset_version": "v", "canonical_product_id": item,
            "title": f"t-{item}",
            "lines": [{"fact_id": "F1", "benchmark_unit_price_krw": price,
                       "comparison_basis_id": "b", "unit": "대", "tax_basis": "vat_included",
                       "period": None,
                       "price_evidence": [{"kind": "price_page", "source_url": "https://example.com/a",
                                           "quote": "q tag", "observed_price_krw": "1000",
                                           "observed_at": "2026-10-01", "vat_included": True,
                                           "shipping_included": True}]}]}


def _detail(sid: str = "SID1", item: str = "11111111", state: str = "accepted",
            revision: int = 2, amount: int | None = 2000) -> SubmissionDetail:
    return SubmissionDetail(submission_id=sid, item_id=item, state=state, revision=revision,
                            amount_krw=amount, held=False, lint_status="clean",
                            lint_lines=(), lint_hints=())


def _preview(total: int = 2000, ok: bool = True, action: str = "revise_existing") -> PreviewResult:
    return PreviewResult(ok=ok, total_amount_krw=total, line_amounts_krw={}, next_action=action,
                         product_already_claimed=False, claimed_by_me=True,
                         existing_submission_id="SID1", current_revision=2)


def test_unchanged_body_is_kept() -> None:
    body = _body()
    bundle = _bundle()
    rows = plan_remediation([_detail()], {"11111111": (body, bundle)},
                            {"11111111": body_hash(body)}, {"11111111": _preview()})
    assert len(rows) == 1
    assert rows[0].action is RemediationAction.KEEP
    assert rows[0].reasons == ("unchanged",)


def test_changed_benchmark_is_revised() -> None:
    old = _body(price="1000")
    new = _body(price="900")
    bundle = _bundle(price=900)
    rows = plan_remediation([_detail()], {"11111111": (new, bundle)},
                            {"11111111": body_hash(old)}, {"11111111": _preview(total=2000)})
    assert rows[0].action is RemediationAction.REVISE
    assert rows[0].new_amount_krw == 2000


def test_same_total_with_another_listing_is_still_revised() -> None:
    old = _body(price="1000")
    new = dict(_body(price="1000"))
    new["lines"] = [{"fact_id": "F1", "benchmark_unit_price_krw": "1000",
                     "comparison_basis_id": "b", "unit": "대", "tax_basis": "vat_included",
                     "period": None,
                     "price_evidence": [{"kind": "price_page", "source_url": "https://example.com/another-listing",
                                         "quote": "q tag", "observed_price_krw": "1000",
                                         "observed_at": "2026-10-01", "vat_included": True,
                                         "shipping_included": True}]}]
    assert body_hash(old) != body_hash(new)
    rows = plan_remediation([_detail()], {"11111111": (new, _bundle())},
                            {"11111111": body_hash(old)}, {"11111111": _preview()})
    assert rows[0].action is RemediationAction.REVISE


def test_no_admissible_line_means_withdraw() -> None:
    rows = plan_remediation([_detail()], {}, {}, {})
    assert rows[0].action is RemediationAction.WITHDRAW
    assert rows[0].new_amount_krw == 0
    assert rows[0].reasons == ("no_admissible_line",)


def test_unconfirmed_preview_blocks_action() -> None:
    body = _body(price="900")
    rows = plan_remediation([_detail()], {"11111111": (body, _bundle(price=900))},
                            {"11111111": "other"}, {"11111111": _preview(total=9999)})
    assert rows[0].action is RemediationAction.KEEP
    assert rows[0].reasons == ("preview_not_admissible",)


def test_withdrawn_submissions_are_ignored() -> None:
    rows = plan_remediation([_detail(state="withdrawn")], {}, {}, {})
    assert rows == []


def test_ordering_is_stable() -> None:
    keep_body = _body()
    revise_body = _body(price="900")
    own = [_detail("S1", "11111111", revision=1, amount=100),
           _detail("S2", "22222222", revision=1, amount=900),
           _detail("S3", "33333333", revision=1, amount=500)]
    rebuilt = {"11111111": (keep_body, _bundle("11111111")),
               "22222222": (revise_body, _bundle("22222222", price=900))}
    hashes = {"11111111": body_hash(keep_body), "22222222": "old"}
    previews = {"11111111": _preview(), "22222222": _preview()}
    rows = plan_remediation(own, rebuilt, hashes, previews)
    assert [r.action for r in rows] == [RemediationAction.REVISE, RemediationAction.WITHDRAW, RemediationAction.KEEP]
    assert rows[1].item_id == "33333333"


def test_plan_file_is_private(tmp_path: Path) -> None:
    rows = plan_remediation([_detail()], {}, {}, {})
    path = write_plan(rows, tmp_path / "plan.jsonl")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["action"] == "withdraw"


def test_bad_item_never_raises() -> None:
    from dataclasses import replace

    bad = _detail()
    none_item = replace(bad, item_id=None)
    assert plan_remediation([none_item], {}, {}, {}) == []  # type: ignore[arg-type]
    assert plan_remediation(["not-a-detail"], {}, {}, {}) == []  # type: ignore[list-item]


def test_zero_lines_means_withdraw() -> None:
    from datetime import UTC, datetime

    empty = EvidenceBundle(item_id="11111111", lane=Lane.CROSS_CONTRACT, title="t", claim="c",
                           comparison_scope="s", reproduction_notes="r", lines=(),
                           rejected=(), caveats=(), review_required=False,
                           generated_at=datetime(2026, 10, 13, 12, tzinfo=UTC), unit_notes=())
    rows = plan_remediation([_detail()], {"11111111": (_body(), empty)}, {}, {})
    assert rows[0].action is RemediationAction.WITHDRAW


def test_unserializable_body_is_kept() -> None:
    rows = plan_remediation([_detail()], {"11111111": ({"x": object()}, _bundle())}, {}, {"11111111": _preview()})
    assert rows[0].action is RemediationAction.KEEP
    assert rows[0].reasons == ("preview_not_admissible",)


def test_broken_bundle_lines_blocks_withdrawal() -> None:
    class _Bad:
        @property
        def lines(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        @property
        def expected_total_krw(self) -> int:
            return 0

    rows = plan_remediation([_detail()], {"11111111": (_body(), _Bad())}, {}, {})  # type: ignore[dict-item]
    assert rows[0].action is RemediationAction.KEEP


def test_unknown_series_is_kept_not_withdrawn() -> None:
    rows = plan_remediation([_detail()], {}, {}, {}, {"11111111"})
    assert len(rows) == 1
    assert rows[0].action is RemediationAction.KEEP
    assert rows[0].reasons == ("series_unresolved",)
    assert rows[0].new_amount_krw == 2000


def test_resolved_series_without_a_line_is_withdrawn() -> None:
    rows = plan_remediation([_detail()], {}, {}, {}, {"22222222"})
    assert rows[0].action is RemediationAction.WITHDRAW
    assert rows[0].reasons == ("no_admissible_line",)


def test_unresolved_flag_does_not_override_a_rebuilt_body() -> None:
    body = _body()
    bundle = _bundle()
    rows = plan_remediation([_detail()], {"11111111": (body, bundle)},
                            {"11111111": body_hash(body)}, {"11111111": _preview()}, {"11111111"})
    assert rows[0].action is RemediationAction.KEEP
    assert rows[0].reasons == ("unchanged",)


def test_broken_bundle_with_unresolved_flag_is_kept() -> None:
    class _Bad:
        @property
        def lines(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        @property
        def expected_total_krw(self) -> int:
            return 0

    rows = plan_remediation([_detail()], {"11111111": (_body(), _Bad())}, {}, {}, {"11111111"})  # type: ignore[dict-item]
    assert rows[0].action is RemediationAction.KEEP
    assert rows[0].reasons == ("preview_not_admissible",)


def test_unproven_price_is_kept_not_withdrawn() -> None:
    rows = plan_remediation([_detail()], {}, {}, {}, {"11111111": "price_unproven"})
    assert len(rows) == 1
    assert rows[0].action is RemediationAction.KEEP
    assert rows[0].reasons == ("price_unproven",)
    assert rows[0].new_amount_krw == 2000


def test_plain_collection_still_means_series() -> None:
    rows = plan_remediation([_detail()], {}, {}, {}, frozenset({"11111111"}))
    assert rows[0].action is RemediationAction.KEEP
    assert rows[0].reasons == ("series_unresolved",)


def test_proven_absence_of_line_is_still_withdrawal_candidate() -> None:
    rows = plan_remediation([_detail()], {}, {}, {}, {})
    assert rows[0].action is RemediationAction.WITHDRAW
    assert rows[0].reasons == ("no_admissible_line",)
