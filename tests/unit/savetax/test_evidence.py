"""Invariant guards for evidence models, draft bodies, and bundle writers."""

import json
import os
import stat
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from src.savetax.competition import EventInfo, Fact
from src.savetax.config import EvidenceConfig
from src.savetax.errors import DomainValidationError
from src.savetax.evidence import (
    EvidenceBundle,
    Lane,
    LineProposal,
    PriceEvidence,
    RejectedFact,
    build_draft_body,
    make_line,
    write_bundles,
    write_draft,
)

_REQUEST = date(2026, 3, 15)


def _fact(**over: object) -> Fact:
    base: dict = {
        "fact_id": "F1",
        "listing_id": "L1",
        "contract_no": "C1",
        "request_no": "R1",
        "line_no": "1",
        "request_date": _REQUEST,
        "unit": "EA",
        "tax_basis": "vat_included",
        "basis_id": "basis:EA:vat_included",
        "period_kind": "none",
        "unit_price_krw": Decimal("1000"),
        "quantity": Decimal("3"),
        "comparable_amount_krw": Decimal("1000"),
        "scorable": True,
        "exclusion_reasons": (),
    }
    base.update(over)
    return Fact(**base)


def _proof(**over: object) -> PriceEvidence:
    base: dict = {
        "source_url": "https://prod.danawa.com/info/?pcode=1",
        "quote": "최저가 300원",
        "observed_price_krw": 300,
        "observed_at": _REQUEST,
        "vat_included": True,
        "shipping_included": True,
    }
    base.update(over)
    return PriceEvidence(**base)


def _line(fact_id: str, expected: int, benchmark: Decimal = Decimal("100"), note: str | None = None) -> LineProposal:
    return LineProposal(
        fact_id=fact_id,
        benchmark_unit_price_krw=benchmark,
        comparison_basis_id="basis:EA:vat_included",
        unit="EA",
        tax_basis="vat_included",
        fact_unit_price_krw=Decimal("1000"),
        quantity=Decimal("1"),
        comparable_amount_krw=Decimal(str(1000 + expected)),
        request_date=_REQUEST,
        expected_krw=expected,
        price_evidence=(_proof(note=note) if note else _proof(),),
    )


def _event() -> EventInfo:
    moment = datetime(2026, 10, 12, tzinfo=UTC)
    return EventInfo(
        dataset_release_id="release-live-20261012",
        ruleset_version="rules-v1",
        phase="open",
        opens_at=moment,
        closes_at=moment,
        evidence_format="v2",
        item_catalog="catalog",
    )


def _bundle(lines: list, **over: object) -> EvidenceBundle:
    base: dict = {
        "item_id": "12345678",
        "lane": Lane.CROSS_CONTRACT,
        "title": "과다 집행 의심 품목",
        "claim": "계약 단가가 시장가 대비 과다하다",
        "comparison_scope": "동일 단위와 부가세 포함 기준",
        "reproduction_notes": "재현 절차",
        "lines": tuple(lines),
        "rejected": (),
        "caveats": (),
        "review_required": False,
        "generated_at": datetime(2026, 10, 12, 12, 0, tzinfo=UTC),
    }
    base.update(over)
    return EvidenceBundle(**base)


def test_line_copies_the_fact_basis_verbatim() -> None:
    line = make_line(_fact(), Decimal("300.5"), [_proof()])
    assert line.unit == "EA"
    assert line.tax_basis == "vat_included"
    assert line.comparison_basis_id == "basis:EA:vat_included"


def test_expected_amount_is_floored_per_line() -> None:
    with pytest.raises(DomainValidationError):
        make_line(_fact(), Decimal("333.4"), [_proof()])
    assert make_line(_fact(), Decimal("300.5"), [_proof()]).expected_krw == 98


def test_non_scorable_fact_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        make_line(_fact(scorable=False), Decimal("100"), [_proof()])


def test_fact_without_request_date_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        make_line(_fact(request_date=None), Decimal("100"), [_proof()])


def test_fact_with_period_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        make_line(_fact(period_kind="monthly"), Decimal("100"), [_proof()])


def test_benchmark_with_seven_fractional_digits_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        make_line(_fact(), Decimal("1.0000001"), [_proof()])


@pytest.mark.parametrize("count", [0, 9])
def test_evidence_count_bounds(count: int) -> None:
    with pytest.raises(DomainValidationError):
        make_line(_fact(), Decimal("100"), [_proof() for _ in range(count)])


def test_quote_bounds() -> None:
    with pytest.raises(DomainValidationError):
        _proof(quote="x" * 2001)
    with pytest.raises(DomainValidationError):
        _proof(quote="   ")


def test_non_https_evidence_url_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        _proof(source_url="http://savetax.now/listing/A")


@pytest.mark.parametrize("url", ["https://", "https://u:p@savetax.now/listing/A", "https://savetax.now/" + "x" * 2048])
def test_malformed_or_oversized_evidence_url_is_rejected(url: str) -> None:
    with pytest.raises(DomainValidationError):
        _proof(source_url=url)


def test_stale_observation_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        _proof(observed_at=date(2014, 12, 31))


def test_non_positive_observed_price_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        _proof(observed_price_krw=0)


def test_oversized_note_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        _proof(note="x" * 501)


def test_non_decimal_benchmark_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        make_line(_fact(), "100", [_proof()])  # type: ignore[arg-type]


def test_non_finite_benchmark_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        make_line(_fact(), Decimal("NaN"), [_proof()])


def test_non_positive_benchmark_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        make_line(_fact(), Decimal("0"), [_proof()])


def test_zero_quantity_fact_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        make_line(_fact(quantity=Decimal("0")), Decimal("100"), [_proof()])


@pytest.mark.parametrize("quantity", [Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity")])
def test_non_finite_quantity_is_rejected(quantity: Decimal) -> None:
    with pytest.raises(DomainValidationError):
        make_line(_fact(quantity=quantity), Decimal("100"), [_proof()])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("claim", "x" * 2001),
        ("comparison_scope", "x" * 1001),
        ("reproduction_notes", "x" * 4001),
    ],
)
def test_draft_text_limits_are_enforced(field: str, value: str) -> None:
    with pytest.raises(DomainValidationError):
        build_draft_body(_bundle([_line("F1", 10)], **{field: value}), _event(), EvidenceConfig())


def test_draft_has_no_unknown_fields() -> None:
    body = build_draft_body(_bundle([_line("F1", 30), _line("F2", 20)]), _event(), EvidenceConfig())
    assert set(body) == {
        "dataset_release_id",
        "ruleset_version",
        "canonical_product_id",
        "title",
        "claim",
        "comparison_scope",
        "reproduction_notes",
        "lines",
    }
    assert body["dataset_release_id"] == "release-live-20261012"
    assert body["canonical_product_id"] == "12345678"
    for line in body["lines"]:
        assert set(line) == {
            "fact_id",
            "benchmark_unit_price_krw",
            "comparison_basis_id",
            "unit",
            "tax_basis",
            "period",
            "price_evidence",
        }
        assert line["period"] is None
        for evidence in line["price_evidence"]:
            assert set(evidence) == {
                "kind",
                "source_url",
                "quote",
                "observed_price_krw",
                "observed_at",
                "vat_included",
                "shipping_included",
            }


def test_draft_numbers_are_strings() -> None:
    body = build_draft_body(_bundle([_line("F1", 10, benchmark=Decimal("2508000.000"))]), _event(), EvidenceConfig())
    assert body["lines"][0]["benchmark_unit_price_krw"] == "2508000"
    assert body["lines"][0]["price_evidence"][0]["observed_price_krw"] == "300"


def test_draft_note_is_emitted_only_when_present() -> None:
    body = build_draft_body(_bundle([_line("F1", 10, note="배송비 포함")]), _event(), EvidenceConfig())
    assert body["lines"][0]["price_evidence"][0]["note"] == "배송비 포함"


def test_draft_lines_are_ordered_by_expected_amount() -> None:
    body = build_draft_body(
        _bundle([_line("F1", 10), _line("F2", 30), _line("F3", 20)]), _event(), EvidenceConfig()
    )
    assert [line["fact_id"] for line in body["lines"]] == ["F2", "F3", "F1"]


def test_draft_respects_the_line_cap() -> None:
    config = EvidenceConfig(max_lines_per_draft=2)
    body = build_draft_body(_bundle([_line("F1", 10), _line("F2", 30), _line("F3", 20)]), _event(), config)
    assert [line["fact_id"] for line in body["lines"]] == ["F2", "F3"]


def test_draft_respects_the_byte_cap() -> None:
    bundle = _bundle([_line("F1", 30, note="x" * 400), _line("F2", 10, note="y" * 400)])
    full = build_draft_body(bundle, _event(), EvidenceConfig())
    one = build_draft_body(_bundle([_line("F1", 30, note="x" * 400)]), _event(), EvidenceConfig())
    full_size = len(json.dumps(full, ensure_ascii=False).encode())
    one_size = len(json.dumps(one, ensure_ascii=False).encode())
    assert one_size < full_size
    capped = build_draft_body(bundle, _event(), EvidenceConfig(max_draft_bytes=one_size))
    assert [line["fact_id"] for line in capped["lines"]] == ["F1"]
    with pytest.raises(DomainValidationError):
        build_draft_body(bundle, _event(), EvidenceConfig(max_draft_bytes=1))
    with pytest.raises(DomainValidationError):
        build_draft_body(_bundle([]), _event(), EvidenceConfig())


def test_title_length_is_enforced() -> None:
    with pytest.raises(DomainValidationError):
        build_draft_body(_bundle([_line("F1", 10)], title="x" * 201), _event(), EvidenceConfig())
    with pytest.raises(DomainValidationError):
        build_draft_body(_bundle([_line("F1", 10)], title=""), _event(), EvidenceConfig())


def test_bundle_file_is_private_and_atomic(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "bundles.jsonl"
    bundle = _bundle(
        [_line("F1", 30)],
        rejected=(RejectedFact("F9", "terms_mismatch"),),
        caveats=("설치비 별도 확인 필요",),
    )
    assert bundle.expected_total_krw == 30
    assert write_bundles([bundle], target) == target
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    rows = target.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    decoded = json.loads(rows[0])
    assert decoded["expected_total_krw"] == 30
    assert decoded["lane"] == "cross_contract"
    assert [entry for entry in tmp_path.rglob("*") if entry.is_file()] == [target]


def test_write_draft_round_trip(tmp_path: Path) -> None:
    body = build_draft_body(_bundle([_line("F1", 30)]), _event(), EvidenceConfig())
    target = tmp_path / "draft.json"
    assert write_draft(body, target) == target
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    assert json.loads(target.read_text(encoding="utf-8")) == body
