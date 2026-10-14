"""Synthetic seller sources for hermetic lane wiring tests."""

from dataclasses import replace
from urllib.parse import quote

from src.savetax.config import EvidenceConfig


def seller_config(**over: object) -> EvidenceConfig:
    config = EvidenceConfig(**over)  # type: ignore[arg-type]
    parts = tuple(
        replace(part, sources=(f"https://compuzone.co.kr/fixture/{quote(part.key, safe='')}",), sources_pending=False)
        if part.sources_pending else part
        for part in config.recipe_parts
    )
    return replace(config, recipe_parts=parts)
