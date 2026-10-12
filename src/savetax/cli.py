"""Command line entry for candidate screening."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsWrite

from src.savetax.competition import CompetitionClient, ItemState
from src.savetax.config import AuditConfig, SaveTaxConfig, load_token
from src.savetax.dataset import Dataset
from src.savetax.errors import ApiError, ConfigError, DataAccessPending, NetworkError, SaveTaxError
from src.savetax.http import HttpClient, Transport, urllib_transport
from src.savetax.rawapi import RawApiClient
from src.savetax.redaction import redact
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
    return parser


def _pause_log(reason: str, condition: str) -> None:
    logger.warning(
        "[SYS] Candidate search paused: 자료 접근 대기 (Reason: %s, Resume condition: %s)",
        redact(reason),
        redact(condition),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    transport: Transport | None = None,
) -> int:
    """Run screen; return 0 on success, 2 on invalid input, or 3 when data access pauses."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    try:
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
