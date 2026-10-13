"""Evidence models, draft bodies, and atomic bundle writers."""

from __future__ import annotations

import enum
import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.savetax.audit import expected_amount_krw
from src.savetax.competition import EventInfo, Fact
from src.savetax.config import EvidenceConfig, _check_url
from src.savetax.errors import ConfigError, DomainValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "EvidenceBundle",
    "Lane",
    "LineProposal",
    "PriceEvidence",
    "RejectedFact",
    "build_draft_body",
    "make_line",
    "write_bundles",
    "write_draft",
]

_MIN_OBSERVED_DATE = date(2015, 1, 1)


class Lane(enum.StrEnum):
    """Which benchmark lane an evidence bundle scores under."""

    CROSS_CONTRACT = "cross_contract"
    RETAIL = "retail"
    AGENCY = "agency"
    RECIPE = "recipe"


@dataclass(frozen=True)
class PriceEvidence:
    """One price page citation; price in whole KRW."""

    source_url: str
    quote: str
    observed_price_krw: int
    observed_at: date
    vat_included: bool
    shipping_included: bool
    note: str | None = None

    def __post_init__(self) -> None:
        try:
            _check_url(self.source_url, "source_url")
        except ConfigError:
            raise DomainValidationError("[DATA] invalid source_url: must be a safe https URL") from None
        if not self.source_url.startswith("https://") or len(self.source_url) > 2048:
            raise DomainValidationError("[DATA] invalid source_url: must be an https URL within 2048 chars")
        if not self.quote.strip() or len(self.quote) > 2000:
            raise DomainValidationError("[DATA] invalid quote: must be non-blank within 2000 chars")
        if (
            isinstance(self.observed_price_krw, bool)
            or not isinstance(self.observed_price_krw, int)
            or self.observed_price_krw <= 0
        ):
            raise DomainValidationError("[DATA] invalid observed_price_krw: must be a positive int")
        if self.note is not None and len(self.note) > 500:
            raise DomainValidationError("[DATA] invalid note: must be within 500 chars")
        if self.observed_at < _MIN_OBSERVED_DATE:
            raise DomainValidationError("[DATA] invalid observed_at: must be on or after 2015-01-01")


@dataclass(frozen=True)
class LineProposal:
    """One scorable fact with its benchmark unit price and expected score in KRW."""

    fact_id: str
    benchmark_unit_price_krw: Decimal
    comparison_basis_id: str
    unit: str
    tax_basis: str
    fact_unit_price_krw: Decimal
    quantity: Decimal
    comparable_amount_krw: Decimal
    request_date: date
    expected_krw: int
    price_evidence: tuple[PriceEvidence, ...]


@dataclass(frozen=True)
class RejectedFact:
    """A fact left out of the bundle and the stable reason code."""

    fact_id: str
    reason: str


@dataclass(frozen=True)
class EvidenceBundle:
    """All admissible lines for one item under one lane."""

    item_id: str
    lane: Lane
    title: str
    claim: str
    comparison_scope: str
    reproduction_notes: str
    lines: tuple[LineProposal, ...]
    rejected: tuple[RejectedFact, ...]
    caveats: tuple[str, ...]
    review_required: bool
    generated_at: datetime
    unit_notes: tuple[str, ...] = ()

    @property
    def expected_total_krw(self) -> int:
        """Return the summed expected score over every line."""
        return sum(line.expected_krw for line in self.lines)


def _require_benchmark(value: object) -> Decimal:
    if not isinstance(value, Decimal) or isinstance(value, bool):
        raise DomainValidationError("[DATA] invalid benchmark_unit_price_krw: expected Decimal")
    if not value.is_finite():
        raise DomainValidationError("[DATA] invalid benchmark_unit_price_krw: non-finite amount")
    if value <= 0:
        raise DomainValidationError("[DATA] invalid benchmark_unit_price_krw: must be > 0")
    exponent = value.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -6:
        raise DomainValidationError("[DATA] invalid benchmark_unit_price_krw: more than 6 decimals")
    return value


def make_line(
    fact: Fact, benchmark_unit_price_krw: Decimal, price_evidence: Sequence[PriceEvidence]
) -> LineProposal:
    """Return the proposal for one fact; raises DomainValidationError when the fact or evidence is inadmissible."""
    if not fact.scorable:
        raise DomainValidationError("[DATA] invalid fact: not scorable")
    if fact.request_date is None:
        raise DomainValidationError("[DATA] invalid fact: missing request date")
    if fact.period_kind != "none":
        raise DomainValidationError("[DATA] invalid fact: only period_kind none is admissible")
    if not fact.quantity.is_finite() or fact.quantity <= 0:
        raise DomainValidationError("[DATA] invalid fact: quantity must be > 0")
    benchmark = _require_benchmark(benchmark_unit_price_krw)
    items = tuple(price_evidence)
    if not 1 <= len(items) <= 8:
        raise DomainValidationError("[DATA] invalid price_evidence: need 1 to 8 items")
    expected = expected_amount_krw([(fact.comparable_amount_krw, fact.quantity)], benchmark)
    if expected == 0:
        raise DomainValidationError("[DATA] invalid line: expected amount is zero")
    return LineProposal(
        fact_id=fact.fact_id,
        benchmark_unit_price_krw=benchmark,
        comparison_basis_id=fact.basis_id,
        unit=fact.unit,
        tax_basis=fact.tax_basis,
        fact_unit_price_krw=fact.unit_price_krw,
        quantity=fact.quantity,
        comparable_amount_krw=fact.comparable_amount_krw,
        request_date=fact.request_date,
        expected_krw=expected,
        price_evidence=items,
    )


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _evidence_to_draft(evidence: PriceEvidence) -> dict[str, object]:
    item: dict[str, object] = {
        "kind": "price_page",
        "source_url": evidence.source_url,
        "quote": evidence.quote,
        "observed_price_krw": str(evidence.observed_price_krw),
        "observed_at": evidence.observed_at.isoformat(),
        "vat_included": evidence.vat_included,
        "shipping_included": evidence.shipping_included,
    }
    if evidence.note is not None:
        item["note"] = evidence.note
    return item


def _line_to_draft(line: LineProposal) -> dict[str, object]:
    return {
        "fact_id": line.fact_id,
        "benchmark_unit_price_krw": _format_decimal(line.benchmark_unit_price_krw),
        "comparison_basis_id": line.comparison_basis_id,
        "unit": line.unit,
        "tax_basis": line.tax_basis,
        "period": None,
        "price_evidence": [_evidence_to_draft(evidence) for evidence in line.price_evidence],
    }


def _draft_body(bundle: EvidenceBundle, event: EventInfo, lines: Sequence[LineProposal]) -> dict[str, object]:
    return {
        "dataset_release_id": event.dataset_release_id,
        "ruleset_version": event.ruleset_version,
        "canonical_product_id": bundle.item_id,
        "title": bundle.title,
        "claim": bundle.claim,
        "comparison_scope": bundle.comparison_scope,
        "reproduction_notes": bundle.reproduction_notes,
        "lines": [_line_to_draft(line) for line in lines],
    }


def _draft_size_bytes(body: Mapping[str, object]) -> int:
    return len(json.dumps(body, ensure_ascii=False).encode("utf-8"))


def build_draft_body(bundle: EvidenceBundle, event: EventInfo, config: EvidenceConfig) -> dict[str, object]:
    """Return the submission body (schema v2) for a bundle; raises DomainValidationError when no line remains."""
    if not 1 <= len(bundle.title) <= 200:
        raise DomainValidationError("[DATA] invalid title: must be 1 to 200 chars")
    if len(bundle.claim) > 2000:
        raise DomainValidationError("[DATA] invalid claim: must be within 2000 chars")
    if len(bundle.comparison_scope) > 1000:
        raise DomainValidationError("[DATA] invalid comparison_scope: must be within 1000 chars")
    if len(bundle.reproduction_notes) > 4000:
        raise DomainValidationError("[DATA] invalid reproduction_notes: must be within 4000 chars")
    ordered = sorted(bundle.lines, key=lambda line: (-line.expected_krw, line.fact_id))
    kept = ordered[: config.max_lines_per_draft]
    body = _draft_body(bundle, event, kept)
    while kept and _draft_size_bytes(body) > config.max_draft_bytes:
        kept = kept[:-1]
        body = _draft_body(bundle, event, kept)
    if not kept:
        raise DomainValidationError("[DATA] invalid draft: no line remains within the size budget")
    return body


def _line_to_json(line: LineProposal) -> dict[str, Any]:
    return {
        "fact_id": line.fact_id,
        "benchmark_unit_price_krw": _format_decimal(line.benchmark_unit_price_krw),
        "comparison_basis_id": line.comparison_basis_id,
        "unit": line.unit,
        "tax_basis": line.tax_basis,
        "fact_unit_price_krw": _format_decimal(line.fact_unit_price_krw),
        "quantity": _format_decimal(line.quantity),
        "comparable_amount_krw": _format_decimal(line.comparable_amount_krw),
        "request_date": line.request_date.isoformat(),
        "expected_krw": line.expected_krw,
        "price_evidence": [
            {
                "source_url": evidence.source_url,
                "quote": evidence.quote,
                "observed_price_krw": evidence.observed_price_krw,
                "observed_at": evidence.observed_at.isoformat(),
                "vat_included": evidence.vat_included,
                "shipping_included": evidence.shipping_included,
                "note": evidence.note,
            }
            for evidence in line.price_evidence
        ],
    }


def _bundle_to_json(bundle: EvidenceBundle) -> dict[str, Any]:
    return {
        "item_id": bundle.item_id,
        "lane": bundle.lane.value,
        "title": bundle.title,
        "claim": bundle.claim,
        "comparison_scope": bundle.comparison_scope,
        "reproduction_notes": bundle.reproduction_notes,
        "lines": [_line_to_json(line) for line in bundle.lines],
        "rejected": [{"fact_id": item.fact_id, "reason": item.reason} for item in bundle.rejected],
        "caveats": list(bundle.caveats),
        "review_required": bundle.review_required,
        "generated_at": bundle.generated_at.isoformat(),
        "expected_total_krw": bundle.expected_total_krw,
        "unit_notes": list(bundle.unit_notes),
    }


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        try:
            handle.write(text)
            handle.close()
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return path


def write_bundles(bundles: Sequence[EvidenceBundle], path: Path) -> Path:
    """Write bundles as JSONL atomically and return the path."""
    payload = "".join(json.dumps(_bundle_to_json(bundle), ensure_ascii=False) + "\n" for bundle in bundles)
    return _atomic_write_text(path, payload)


def write_draft(body: Mapping[str, object], path: Path) -> Path:
    """Write one draft body as UTF-8 JSON atomically and return the path."""
    return _atomic_write_text(path, json.dumps(body, ensure_ascii=False))
