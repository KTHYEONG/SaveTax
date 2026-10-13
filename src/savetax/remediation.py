"""Remediation planning for already-submitted items."""

from __future__ import annotations

import enum
import json
import os
import tempfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from src.savetax.evidence import EvidenceBundle
from src.savetax.submissions import PreviewResult, SubmissionDetail
from src.savetax.workflow import body_hash

__all__ = ["RemediationAction", "RemediationRow", "plan_remediation", "write_plan"]


class RemediationAction(enum.StrEnum):
    """Planned outcome for one own submission."""

    KEEP = "keep"
    REVISE = "revise"
    WITHDRAW = "withdraw"


@dataclass(frozen=True)
class RemediationRow:
    """Planned action for one own submission; amounts in KRW."""

    item_id: str
    submission_id: str
    revision: int
    action: RemediationAction
    old_amount_krw: int | None
    new_amount_krw: int
    old_lines: int | None
    new_lines: int
    reasons: tuple[str, ...]


_ORDER = {RemediationAction.REVISE: 0, RemediationAction.WITHDRAW: 1, RemediationAction.KEEP: 2}


def _has_rebuilt_lines(entry: tuple[Mapping[str, object], EvidenceBundle] | None) -> bool:
    """Return True when the rebuilt entry carries at least one line; undeterminable counts as True."""
    if entry is None:
        return False
    try:
        return len(entry[1].lines) > 0
    except Exception:  # noqa: BLE001
        return True


def plan_remediation(
    own: Sequence[SubmissionDetail],
    rebuilt: Mapping[str, tuple[Mapping[str, object], EvidenceBundle]],
    submitted_hashes: Mapping[str, str],
    previews: Mapping[str, PreviewResult],
    unresolved: Mapping[str, str] | Collection[str] = frozenset(),
) -> list[RemediationRow]:
    """Classify every active own submission; never raises for a single bad item."""
    import logging as _logging

    rows: list[RemediationRow] = []
    if isinstance(unresolved, Mapping):
        unresolved_map: Mapping[str, str] = dict(unresolved)
    else:
        unresolved_map = dict.fromkeys(unresolved, "series_unresolved")
    for detail in own:
        try:
            row = _classify(detail, rebuilt, submitted_hashes, previews, unresolved_map)
        except Exception as exc:  # noqa: BLE001
            _logging.getLogger(__name__).info("[SUBMIT] remediation skip reason=%s", type(exc).__name__)
            continue
        if row is not None:
            rows.append(replace(row, old_lines=len(detail.lint_lines) if detail.lint_lines else None))
    rows.sort(key=lambda r: (_ORDER[r.action], -(r.old_amount_krw or 0), r.item_id))
    return rows


def _classify(
    detail: SubmissionDetail,
    rebuilt: Mapping[str, tuple[Mapping[str, object], EvidenceBundle]],
    submitted_hashes: Mapping[str, str],
    previews: Mapping[str, PreviewResult],
    unresolved: Mapping[str, str] | None = None,
) -> RemediationRow | None:
    if detail.state == "withdrawn":
        return None
    item = detail.item_id
    if not isinstance(item, str) or not item:
        return None
    entry = rebuilt.get(item)
    old_amount = detail.amount_krw
    unresolved_reason = unresolved.get(item) if unresolved else None
    if unresolved_reason is not None and not _has_rebuilt_lines(entry):
        current = old_amount if isinstance(old_amount, int) else 0
        return RemediationRow(item, detail.submission_id, detail.revision, RemediationAction.KEEP,
                              old_amount, current, None, 0, (unresolved_reason,))
    if entry is None:
        return RemediationRow(item, detail.submission_id, detail.revision, RemediationAction.WITHDRAW,
                              old_amount, 0, None, 0, ("no_admissible_line",))
    body, bundle = entry
    try:
        new_lines = len(bundle.lines)
    except Exception:
        return RemediationRow(item, detail.submission_id, detail.revision, RemediationAction.KEEP,
                              old_amount, old_amount or 0, None, 0, ("preview_not_admissible",))
    if new_lines == 0:
        return RemediationRow(item, detail.submission_id, detail.revision, RemediationAction.WITHDRAW,
                              old_amount, 0, None, 0, ("no_admissible_line",))
    try:
        digest = body_hash(body)
    except Exception:
        return RemediationRow(item, detail.submission_id, detail.revision, RemediationAction.KEEP,
                              old_amount, old_amount if isinstance(old_amount, int) else 0,
                              None, new_lines, ("preview_not_admissible",))
    if item in submitted_hashes and digest == submitted_hashes[item]:
        current = old_amount if isinstance(old_amount, int) else 0
        return RemediationRow(item, detail.submission_id, detail.revision, RemediationAction.KEEP,
                              old_amount, current, None, new_lines, ("unchanged",))
    preview = previews.get(item)
    if (preview is None or not preview.ok or preview.next_action != "revise_existing"
            or preview.total_amount_krw != bundle.expected_total_krw):
        current = old_amount if isinstance(old_amount, int) else 0
        return RemediationRow(item, detail.submission_id, detail.revision, RemediationAction.KEEP,
                              old_amount, current, None, new_lines, ("preview_not_admissible",))
    return RemediationRow(item, detail.submission_id, detail.revision, RemediationAction.REVISE,
                          old_amount, preview.total_amount_krw, None, new_lines, ("changed",))


def _row_to_json(row: RemediationRow) -> dict[str, object]:
    return {
        "item_id": row.item_id,
        "submission_id": row.submission_id,
        "revision": row.revision,
        "action": row.action.value,
        "old_amount_krw": row.old_amount_krw,
        "new_amount_krw": row.new_amount_krw,
        "old_lines": row.old_lines,
        "new_lines": row.new_lines,
        "reasons": list(row.reasons),
    }


def write_plan(rows: Sequence[RemediationRow], path: Path) -> Path:
    """Write the plan as JSONL with mode 0o600 and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(_row_to_json(row), ensure_ascii=False) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(payload)
            handle.close()
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    os.chmod(path, 0o600)
    return path
