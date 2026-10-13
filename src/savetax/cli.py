"""Command line entry for candidate screening."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from _typeshed import SupportsWrite

    from src.savetax.competition import CompetitionClient
    from src.savetax.publicweb import PublicFetcher
    from src.savetax.submissions import SubmissionClient
    from src.savetax.writehttp import WriteTransport

from src.savetax.competition import CompetitionClient, ItemState
from src.savetax.config import AuditConfig, EvidenceConfig, SaveTaxConfig, UnitConflictPolicy, load_token
from src.savetax.crosscontract import build_cross_bundle, find_cross_candidates
from src.savetax.dataset import Dataset
from src.savetax.errors import ApiError, ConfigError, DataAccessPending, NetworkError, SaveTaxError, SubmissionPending
from src.savetax.evidence import EvidenceBundle, build_draft_body, write_bundles, write_draft
from src.savetax.http import HttpClient, HttpResponse, Transport, urllib_transport
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
    review = sub.add_parser("review")
    review.add_argument("--drafts-dir", type=Path)
    review.add_argument("--limit", type=int)
    review.add_argument("--kind", choices=("submit", "revise", "withdraw"), default="submit")
    review.add_argument("--item")
    review.add_argument("--draft", type=Path)
    approve = sub.add_parser("approve")
    approve.add_argument("--challenge", required=True)
    approve.add_argument("--note", required=True)
    approve.add_argument("--kind", choices=("submit", "revise", "withdraw"), default="submit")
    approve.add_argument("--item")
    approve.add_argument("--draft", type=Path)
    approve.add_argument("--drafts-dir", type=Path)
    approve.add_argument("--limit", type=int)
    submit = sub.add_parser("submit")
    submit.add_argument("--execute", action="store_true")
    submit.add_argument("--limit", type=int)
    sub.add_parser("status")
    revise = sub.add_parser("revise")
    revise.add_argument("--item", required=True)
    revise.add_argument("--draft", type=Path, required=True)
    revise.add_argument("--execute", action="store_true")
    withdraw = sub.add_parser("withdraw")
    withdraw.add_argument("--item", required=True)
    withdraw.add_argument("--execute", action="store_true")
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


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _live_price(fetcher: PublicFetcher, url: str) -> int | None:
    from src.savetax.crosscontract import parse_listing_page
    from src.savetax.danawa import parse_product
    from src.savetax.publicweb import visible_text
    from src.savetax.shops import parse_shop_product

    try:
        page = fetcher.get(url)
        if page.status != 200 or not page.text:
            return None
        parsed = urlparse(url)
        if parsed.hostname == "savetax.now":
            listing = parse_listing_page(visible_text(page.text))
            return listing.unit_price_krw if listing.listing_id == parsed.path.rsplit("/", 1)[-1] else None
        if parsed.hostname == "prod.danawa.com":
            pcode = parse_qs(parsed.query).get("pcode", [""])[0]
            return parse_product(page.text, pcode).low_price_krw
        product = parse_shop_product(page.text, url)
        return product.price_krw if product is not None else None
    except NetworkError:
        raise
    except (SaveTaxError, OSError, ValueError):
        return None


def _refusing_write_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> HttpResponse:
    raise ConfigError("[SYS] refusing write transmission on a read-only command")


def _workflow_inputs(
    drafts_dir: Path, evidence_dir: Path, now: datetime
) -> list[tuple[str, Mapping[str, object], EvidenceBundle]]:
    from src.savetax.workflow import load_bundles, load_drafts

    drafts = load_drafts(drafts_dir)
    bundles = load_bundles([evidence_dir / "cross-bundles.jsonl", evidence_dir / "retail-bundles.jsonl"], now=now)
    from src.savetax.workflow import matching_bundle

    if any(item not in bundles for item, _ in drafts):
        raise ConfigError("[SYS] draft is missing its evidence bundle")
    selected: dict[str, tuple[str, Mapping[str, object], EvidenceBundle]] = {}
    for item, body in drafts:
        bundle = matching_bundle(body, bundles)
        if item not in selected or bundle.expected_total_krw > selected[item][2].expected_total_krw:
            selected[item] = (item, body, bundle)
    return list(selected.values())


def _single_draft(path: Path, evidence_dir: Path, now: datetime) -> list[tuple[str, Mapping[str, object], EvidenceBundle]]:
    from src.savetax.errors import DataIntegrityError
    from src.savetax.workflow import load_bundles, matching_bundle

    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        bundles = load_bundles([evidence_dir / "cross-bundles.jsonl", evidence_dir / "retail-bundles.jsonl"], now=now)
        item = body["canonical_product_id"]
        bundle = matching_bundle(body, bundles)
    except (ValueError, TypeError, KeyError, OSError):
        raise DataIntegrityError("[DATA] draft is missing matching evidence") from None
    return [(item, body, bundle)]


def _submission_stack(
    environ: Mapping[str, str] | None,
    transport: Transport | None,
    write_transport: WriteTransport | None,
    *,
    allow_writes: bool = False,
) -> tuple[SaveTaxConfig, CompetitionClient, SubmissionClient, PublicFetcher]:
    from src.savetax.competition import CompetitionClient as _Competition
    from src.savetax.config import EvidenceConfig as _Evidence
    from src.savetax.http import HttpClient as _Http
    from src.savetax.publicweb import PublicFetcher as _Fetcher
    from src.savetax.submissions import SubmissionClient as _Submissions
    from src.savetax.writehttp import WriteHttpClient as _Write
    from src.savetax.writehttp import urllib_write_transport

    config = SaveTaxConfig()
    token = load_token(config, os.environ if environ is None else environ)
    http = _Http(config, token, transport=urllib_transport if transport is None else transport, sleep=time.sleep)
    competition = _Competition(http, config, sleep=time.sleep)
    sender = urllib_write_transport if write_transport is None else write_transport

    def guarded(method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> HttpResponse:
        if not allow_writes and urlparse(url).path != "/v1/events/current/submissions/preview":
            raise ConfigError("[SYS] read-only workflow cannot transmit to a write endpoint")
        return sender(method, url, headers, body, timeout)

    write = _Write(
        config, token, transport=guarded,
        sleep=time.sleep,
    )
    submissions = _Submissions(write, competition, http, config, sleep=time.sleep)
    fetcher = _Fetcher(config, _Evidence(), transport=urllib_transport if transport is None else transport)
    return config, competition, submissions, fetcher


def _run_review(
    args: argparse.Namespace, environ: Mapping[str, str] | None,
    transport: Transport | None, write_transport: WriteTransport | None,
) -> int:
    from src.savetax.approval import build_review_set
    from src.savetax.config import ApprovalConfig, EvidenceConfig
    from src.savetax.ledger import Ledger
    from src.savetax.workflow import _print_table, action_review, run_review

    evidence = EvidenceConfig()
    approval = ApprovalConfig()
    drafts_dir = args.drafts_dir if args.drafts_dir is not None else evidence.evidence_dir / "drafts"
    if args.limit is not None and args.limit < 1:
        raise ConfigError("[SYS] limit must be >= 1")
    _, competition, submissions, _ = _submission_stack(environ, transport, write_transport)
    now = _utcnow()
    joined = (_single_draft(args.draft, evidence.evidence_dir, now) if args.draft is not None
              else _workflow_inputs(Path(drafts_dir), evidence.evidence_dir, now))
    if args.item is not None:
        joined = [entry for entry in joined if entry[0] == args.item]
    if args.kind != "submit":
        if args.item is None:
            raise ConfigError("[SYS] action review requires --item")
        rows, _ = action_review(joined, submissions, args.kind)
        review = build_review_set(rows, competition.get_event(), now)
        import sys

        _print_table(sys.stdout, rows, review.total_krw, review.challenge)
        return 0
    ledger = Ledger(approval.ledger_path, now=_utcnow)
    run_review(joined, event=competition.get_event(), status=competition.get_me(),
               competition=competition, submission_client=submissions,
               ledger=ledger, now=now, limit=args.limit, reserve=approval.reserve_claims)
    return 0


def _run_approve(
    args: argparse.Namespace, environ: Mapping[str, str] | None,
    transport: Transport | None, write_transport: WriteTransport | None,
) -> int:
    from src.savetax.config import ApprovalConfig, EvidenceConfig
    from src.savetax.ledger import Ledger
    from src.savetax.workflow import run_approve

    evidence = EvidenceConfig()
    approval = ApprovalConfig()
    if not args.note.strip():
        raise ConfigError("[SYS] invalid approval note: must be non-empty")
    _, competition, submissions, _ = _submission_stack(environ, transport, write_transport)
    now = _utcnow()
    joined = (_single_draft(args.draft, evidence.evidence_dir, now) if args.draft is not None
              else _workflow_inputs(args.drafts_dir or evidence.evidence_dir / "drafts", evidence.evidence_dir, now))
    ledger = Ledger(approval.ledger_path, now=_utcnow)
    run_approve(joined, event=competition.get_event(), status=competition.get_me(),
                competition=competition, submission_client=submissions, ledger=ledger, now=now,
                challenge=args.challenge, note=args.note, kind=args.kind, item=args.item,
                reserve=approval.reserve_claims, limit=args.limit)
    return 0


def _run_submit(
    args: argparse.Namespace, environ: Mapping[str, str] | None,
    transport: Transport | None, write_transport: WriteTransport | None,
) -> int:
    import sys

    from src.savetax.config import ApprovalConfig, EvidenceConfig
    from src.savetax.ledger import Ledger
    from src.savetax.workflow import run_submit

    evidence = EvidenceConfig()
    approval = ApprovalConfig()
    if args.limit is not None and args.limit < 1:
        raise ConfigError("[SYS] limit must be >= 1")
    now = _utcnow()
    joined = _workflow_inputs(evidence.evidence_dir / "drafts", evidence.evidence_dir, now)
    ledger = Ledger(approval.ledger_path, now=_utcnow)
    if not args.execute:
        selected = sorted(joined, key=lambda e: (-e[2].expected_total_krw, e[0]))
        for item_id, _body, bundle in selected[:args.limit]:
            sys.stdout.write(f"would-submit {redact(item_id)} expected={bundle.expected_total_krw}\n")
        return 0
    _, competition, submissions, fetcher = _submission_stack(environ, transport, write_transport, allow_writes=True)
    _, code = run_submit(
        joined, event=competition.get_event(), status=competition.get_me(),
        competition=competition, submission_client=submissions, ledger=ledger,
        fetch_price=lambda url: _live_price(fetcher, url), now=now,
        ttl=timedelta(hours=approval.ttl_hours),
        reserve=approval.reserve_claims, execute=True, limit=args.limit, sleep=time.sleep,
        poll_seconds=approval.lint_poll_seconds, timeout_seconds=approval.lint_timeout_seconds,
        pause_seconds=approval.pause_between_submits_seconds,
        stop_after_consecutive_suspect=approval.stop_after_consecutive_suspect,
        clock=_utcnow,
    )
    return code


def _run_status(
    args: argparse.Namespace, environ: Mapping[str, str] | None, transport: Transport | None,
) -> int:
    from src.savetax.config import ApprovalConfig
    from src.savetax.ledger import Ledger
    from src.savetax.workflow import run_status

    approval = ApprovalConfig()
    _, _, submissions, _ = _submission_stack(environ, transport, _refusing_write_transport)
    ledger = Ledger(approval.ledger_path, now=_utcnow)
    run_status(submissions, ledger, _utcnow())
    return 0


def _run_revise(
    args: argparse.Namespace, environ: Mapping[str, str] | None,
    transport: Transport | None, write_transport: WriteTransport | None,
) -> int:
    import sys

    from src.savetax.config import ApprovalConfig
    from src.savetax.errors import DataIntegrityError
    from src.savetax.ledger import Ledger
    from src.savetax.workflow import run_revise

    approval = ApprovalConfig()
    draft_path = Path(args.draft)
    try:
        body = json.loads(draft_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError, OSError):
        raise DataIntegrityError("[DATA] malformed draft body") from None
    if not isinstance(body, dict):
        raise DataIntegrityError("[DATA] malformed draft body")
    if not args.execute:
        from src.savetax.config import EvidenceConfig

        expected = _single_draft(draft_path, EvidenceConfig().evidence_dir, _utcnow())[0][2].expected_total_krw
        sys.stdout.write(f"would-revise {redact(args.item)} expected={expected}\n")
        return 0
    _, competition, submissions, fetcher = _submission_stack(environ, transport, write_transport, allow_writes=True)
    ledger = Ledger(approval.ledger_path, now=_utcnow)
    item_id = str(body.get("canonical_product_id", ""))
    details = [detail for detail in submissions.list_my_submissions() if detail.item_id == args.item]
    if len(details) != 1 or item_id != args.item:
        raise ConfigError("[SYS] revision requires one own submission matching --item")
    from src.savetax.config import EvidenceConfig

    bundle = _single_draft(draft_path, EvidenceConfig().evidence_dir, _utcnow())[0][2]
    receipt = run_revise(details[0].submission_id, item_id, body,
                         submission_client=submissions,
                         ledger=ledger, now=_utcnow(),
                         ttl=timedelta(hours=approval.ttl_hours), execute=True,
                         bundle=bundle, event=competition.get_event(),
                         fetch_price=lambda url: _live_price(fetcher, url),
                         clock=_utcnow, event_provider=competition.get_event)
    return 0 if receipt is not None else 2


def _run_withdraw(
    args: argparse.Namespace, environ: Mapping[str, str] | None,
    transport: Transport | None, write_transport: WriteTransport | None,
) -> int:
    from src.savetax.config import ApprovalConfig, EvidenceConfig
    from src.savetax.ledger import Ledger
    from src.savetax.workflow import load_drafts, run_withdraw, submission_for_item

    approval = ApprovalConfig()
    if not args.execute:
        import sys as _sys

        _sys.stdout.write("withdrawing does not refund the used claim count; others may claim the item\n")
        _sys.stdout.write(f"would-withdraw {redact(str(args.item))}\n")
        return 0
    evidence = EvidenceConfig()
    now = _utcnow()
    ledger = Ledger(approval.ledger_path, now=_utcnow)
    drafts = dict(load_drafts(evidence.evidence_dir / "drafts"))
    body = drafts.get(str(args.item))
    if body is None:
        raise ConfigError("[SYS] unknown withdraw item: no draft")
    submission_id = submission_for_item(ledger, str(args.item))
    if submission_id is None:
        raise ConfigError("[SYS] unknown withdraw item: never submitted")
    _, _, submissions, _ = _submission_stack(environ, transport, write_transport, allow_writes=True)
    receipt = run_withdraw(submission_id, str(args.item), body, submission_client=submissions, ledger=ledger,
                 now=now, ttl=timedelta(hours=approval.ttl_hours), execute=True, clock=_utcnow)
    return 0 if receipt is not None else 2


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    transport: Transport | None = None,
    write_transport: WriteTransport | None = None,
) -> int:
    """Run a sub-command; return 0 on success, 2 on invalid input, or 3 when data access pauses."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    try:
        if args.command == "screen":
            return _run_screen(args, environ, transport)
        if args.command == "review":
            return _run_review(args, environ, transport, write_transport)
        if args.command == "approve":
            return _run_approve(args, environ, transport, write_transport)
        if args.command == "submit":
            return _run_submit(args, environ, transport, write_transport)
        if args.command == "status":
            return _run_status(args, environ, transport)
        if args.command == "revise":
            return _run_revise(args, environ, transport, write_transport)
        if args.command == "withdraw":
            return _run_withdraw(args, environ, transport, write_transport)
        return _run_evidence(args, environ, transport)
    except NetworkError as exc:
        _pause_log(exc.stage.value, f"restore connectivity to {exc.host} and rerun the batch")
        return 3
    except (DataAccessPending, SubmissionPending) as exc:
        _pause_log("dataset", str(exc))
        return 3
    except SaveTaxError as exc:
        logger.error("%s", redact(str(exc)))
        return 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
