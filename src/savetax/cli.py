"""Command line entry for candidate screening."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsWrite

from src.savetax.competition import CompetitionClient, ItemState
from src.savetax.config import AuditConfig, EvidenceConfig, SaveTaxConfig, UnitConflictPolicy, load_token
from src.savetax.crosscontract import build_cross_bundle, find_cross_candidates
from src.savetax.dataset import Dataset
from src.savetax.errors import ApiError, ConfigError, DataAccessPending, NetworkError, SaveTaxError
from src.savetax.evidence import EvidenceBundle, build_draft_body, write_bundles, write_draft
from src.savetax.http import HttpClient, Transport, urllib_transport
from src.savetax.publicweb import PublicFetcher
from src.savetax.rawapi import RawApiClient
from src.savetax.redaction import redact
from src.savetax.retail import build_retail_bundle
from src.savetax.screening import (
    annotate_availability,
    screen_batch,
    shuffle_key,
    verify_benchmark_conditions,
    write_candidates,
)

logger = logging.getLogger(__name__)

__all__ = ["main"]


class _SafeParser(argparse.ArgumentParser):
    def _print_message(self, message: str, file: SupportsWrite[str] | None = None) -> None:
        super()._print_message(redact(message), file)


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeParser(prog="cli")
    sub = parser.add_subparsers(dest="command", required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("--batch", type=int, required=True)
    screen.add_argument("--batches", type=int, default=1)
    screen.add_argument("--batch-size", type=int)
    screen.add_argument("--handle")
    screen.add_argument("--data-dir", type=Path)
    screen.add_argument("--out-dir", type=Path)
    screen.add_argument("--online", action="store_true")
    evidence = sub.add_parser("evidence")
    evidence.add_argument("--lane", choices=("cross", "retail", "all"), default="cross")
    evidence.add_argument("--limit", type=int, default=50)
    evidence.add_argument("--items")
    evidence.add_argument("--batch", type=int, default=0)
    evidence.add_argument("--batches", type=int, default=1)
    evidence.add_argument("--data-dir", type=Path)
    evidence.add_argument("--out-dir", type=Path)
    evidence.add_argument("--include-review", action="store_true")
    evidence.add_argument("--unit-policy", choices=("annotate", "reject"), default="annotate")
    return parser


def _pause_log(reason: str, condition: str) -> None:
    logger.warning(
        "[SYS] Candidate search paused: 자료 접근 대기 (Reason: %s, Resume condition: %s)",
        redact(reason),
        redact(condition),
    )


def _run_screen(args: argparse.Namespace, environ: Mapping[str, str] | None, transport: Transport | None) -> int:
    if args.batch < 0 or args.batches < 1:
        raise ConfigError("[SYS] batch must be >= 0 and batches must be >= 1")
    defaults = SaveTaxConfig()
    config = replace(
        defaults,
        dataset_dir=args.data_dir if args.data_dir is not None else defaults.dataset_dir,
        candidates_dir=args.out_dir if args.out_dir is not None else defaults.candidates_dir,
    )
    audit_defaults = AuditConfig()
    audit = replace(
        audit_defaults,
        batch_size=args.batch_size if args.batch_size is not None else audit_defaults.batch_size,
    )
    version = config.dataset_version
    handle = args.handle if args.handle is not None else config.participant_handle
    if handle:
        shuffle_key(handle, version, "")
    competition: CompetitionClient | None = None
    raw: RawApiClient | None = None
    if args.online:
        token = load_token(config, os.environ if environ is None else environ)
        http = HttpClient(
            config, token, transport=urllib_transport if transport is None else transport, sleep=time.sleep
        )
        competition = CompetitionClient(http, config, sleep=time.sleep)
        raw = RawApiClient(http, config, sleep=time.sleep)
        if not handle:
            handle = competition.get_me().handle
            shuffle_key(handle, version, "")
    elif not handle:
        raise ConfigError("[SYS] handle required offline")
    with Dataset(config, audit) as dataset:
        for batch in range(args.batch, args.batch + args.batches):
            candidates = screen_batch(dataset, audit, handle, batch)
            if not candidates:
                break
            try:
                if competition is not None and raw is not None:
                    candidates = annotate_availability(candidates, competition)
                    processed = 0
                    for position, candidate in enumerate(candidates):
                        benchmark = candidate.benchmark
                        if benchmark is None or candidate.availability is not ItemState.AVAILABLE:
                            continue
                        try:
                            candidates[position] = verify_benchmark_conditions(candidate, raw, audit)
                        except ApiError as exc:
                            note = f"api_error:{exc.code}"
                            candidates[position] = replace(
                                candidate,
                                benchmark=replace(benchmark, conditions_verified=False, condition_note=note),
                            )
                            logger.info(
                                "[EVIDENCE] item_id=%s benchmark_listing=%s verified=False note=%s",
                                redact(candidate.item_id),
                                redact(benchmark.listing_id),
                                note,
                            )
                        processed += 1
                        if processed % audit.progress_every == 0:
                            logger.info("[AUDIT] verified=%d batch=%d", processed, batch)
            finally:
                write_candidates(candidates, config.candidates_dir / f"batch-{batch:04d}.jsonl")
    return 0


def _append_unit_conflicts(out_dir: Path, records: Sequence[Mapping[str, str]]) -> None:
    if not records:
        return
    path = Path(out_dir) / "unit-conflicts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _run_cross_lane(
    competition: CompetitionClient,
    raw: RawApiClient,
    fetcher: PublicFetcher,
    evidence: EvidenceConfig,
    audit: AuditConfig,
    out_dir: Path,
    dataset: Dataset,
    wanted: list[str] | None,
    limit: int,
    include_review: bool,
    unit_conflicts: list[dict[str, str]],
) -> tuple[int, int, int]:
    bundles: list[EvidenceBundle] = []
    candidates_n = 0
    available_n = 0
    drafts = 0
    try:
        found = find_cross_candidates(dataset, evidence, audit)
        if wanted is not None:
            keep = set(wanted)
            found = [c for c in found if c.item_id in keep]
        candidates_n = len(found)
        ids = [c.item_id for c in found]
        states = {result.input: result.state for result in competition.lookup_items(ids)}
        available = [c for c in found if states.get(c.item_id) is ItemState.AVAILABLE]
        available_n = len(available)
        event = competition.get_event()
        today = date.today()
        for candidate in available[:limit]:
            try:
                detail = competition.get_item(candidate.item_id)
                bundle = build_cross_bundle(
                    candidate, detail, dataset, raw, fetcher, evidence, audit,
                    today=today, on_unit_conflict=unit_conflicts.append,
                )
                if bundle is None:
                    continue
                bundles.append(bundle)
                if not bundle.lines:
                    continue
                if bundle.review_required and not include_review:
                    continue
                body = build_draft_body(bundle, event, evidence)
                write_draft(body, Path(out_dir) / "drafts" / f"{bundle.item_id}.json")
                drafts += 1
            except ApiError as exc:
                logger.info("[EVIDENCE] item_id=%s skipped=api_error:%s", redact(candidate.item_id), redact(exc.code))
                continue
    finally:
        write_bundles(bundles, Path(out_dir) / "cross-bundles.jsonl")
        expected = sum(b.expected_total_krw for b in bundles)
        logger.info(
            "[EVIDENCE] lane=cross candidates=%d available=%d bundles=%d drafts=%d expected_krw=%d",
            candidates_n,
            available_n,
            len(bundles),
            drafts,
            expected,
        )
    return candidates_n, available_n, drafts


def _run_retail_lane(
    competition: CompetitionClient,
    raw: RawApiClient,
    fetcher: PublicFetcher,
    evidence: EvidenceConfig,
    audit: AuditConfig,
    out_dir: Path,
    dataset: Dataset,
    wanted: list[str] | None,
    limit: int,
    include_review: bool,
    handle: str,
    batch: int,
    batches: int,
    separate_drafts: bool = False,
) -> tuple[int, int, int]:
    bundles: list[EvidenceBundle] = []
    candidates_n = 0
    available_n = 0
    drafts = 0
    try:
        pooled: list[str] = []
        for batch_index in range(batch, batch + batches):
            batch_candidates = screen_batch(dataset, audit, handle, batch_index)
            if not batch_candidates:
                break
            for candidate in batch_candidates:
                if candidate.item_id not in pooled:
                    pooled.append(candidate.item_id)
        if wanted is not None:
            keep = set(wanted)
            pooled = [item for item in pooled if item in keep]
        candidates_n = len(pooled)
        states: dict[str, ItemState] = {}
        if pooled:
            states = {result.input: result.state for result in competition.lookup_items(pooled)}
        available = [item for item in pooled if states.get(item) is ItemState.AVAILABLE]
        available_n = len(available)
        event = competition.get_event()
        today = date.today()
        for item_id in available[:limit]:
            try:
                detail = competition.get_item(item_id)
                bundle = build_retail_bundle(detail, dataset, raw, fetcher, evidence, audit, today=today)
                if bundle is None:
                    continue
                bundles.append(bundle)
                if bundle.review_required and not include_review:
                    continue
                body = build_draft_body(bundle, event, evidence)
                draft_dir = Path(out_dir) / "drafts"
                if separate_drafts:
                    draft_dir /= "retail"
                write_draft(body, draft_dir / f"{bundle.item_id}.json")
                drafts += 1
            except ApiError as exc:
                logger.info("[EVIDENCE] item_id=%s skipped=api_error:%s", redact(item_id), redact(exc.code))
                continue
    finally:
        write_bundles(bundles, Path(out_dir) / "retail-bundles.jsonl")
        expected = sum(b.expected_total_krw for b in bundles)
        logger.info(
            "[EVIDENCE] lane=retail candidates=%d available=%d bundles=%d drafts=%d expected_krw=%d",
            candidates_n,
            available_n,
            len(bundles),
            drafts,
            expected,
        )
    return candidates_n, available_n, drafts


def _run_evidence(args: argparse.Namespace, environ: Mapping[str, str] | None, transport: Transport | None) -> int:
    limit = args.limit
    if limit is None or limit < 1:
        raise ConfigError("[SYS] limit must be >= 1")
    batch = args.batch if args.batch is not None else 0
    batches = args.batches if args.batches is not None else 1
    if batch < 0 or batches < 1:
        raise ConfigError("[SYS] batch must be >= 0 and batches must be >= 1")
    defaults = SaveTaxConfig()
    evidence_defaults = EvidenceConfig()
    config = replace(
        defaults,
        dataset_dir=args.data_dir if args.data_dir is not None else defaults.dataset_dir,
    )
    evidence = replace(
        evidence_defaults,
        unit_conflict_policy=(
            UnitConflictPolicy.REJECT if args.unit_policy == "reject" else UnitConflictPolicy.ANNOTATE
        ),
    )
    audit = AuditConfig()
    out_dir = args.out_dir if args.out_dir is not None else evidence.evidence_dir
    token = load_token(config, os.environ if environ is None else environ)
    http = HttpClient(config, token, transport=urllib_transport if transport is None else transport, sleep=time.sleep)
    competition = CompetitionClient(http, config, sleep=time.sleep)
    raw = RawApiClient(http, config, sleep=time.sleep)
    fetcher = PublicFetcher(config, evidence, transport=urllib_transport if transport is None else transport)
    wanted: list[str] | None = None
    if args.items is not None:
        wanted = [part.strip() for part in str(args.items).split(",") if part.strip()]
        wanted = list(dict.fromkeys(wanted))
    records: list[dict[str, str]] = []
    try:
        with Dataset(config, audit) as dataset:
            if args.lane in ("cross", "all"):
                _run_cross_lane(
                    competition, raw, fetcher, evidence, audit, Path(out_dir),
                    dataset, wanted, limit, args.include_review, records,
                )
            if args.lane in ("retail", "all"):
                handle = config.participant_handle
                if not handle:
                    handle = competition.get_me().handle
                _run_retail_lane(
                    competition, raw, fetcher, evidence, audit, Path(out_dir),
                    dataset, wanted, limit, args.include_review, handle, batch,
                    batches, separate_drafts=args.lane == "all",
                )
    finally:
        _append_unit_conflicts(Path(out_dir), records)
        logger.info(
            "[EVIDENCE] unit_conflicts=%d items=%d policy=%s",
            len(records), len({record["item_id"] for record in records}),
            evidence.unit_conflict_policy.value,
        )
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    transport: Transport | None = None,
) -> int:
    """Run screen or evidence; return 0 on success, 2 on invalid input, or 3 when data access pauses."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    try:
        if args.command == "screen":
            return _run_screen(args, environ, transport)
        return _run_evidence(args, environ, transport)
    except NetworkError as exc:
        _pause_log(exc.stage.value, f"restore connectivity to {exc.host} and rerun the batch")
        return 3
    except DataAccessPending as exc:
        _pause_log("dataset", str(exc))
        return 3
    except SaveTaxError as exc:
        logger.error("%s", redact(str(exc)))
        return 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
