"""Invariant guards for the variant policy configuration."""

from __future__ import annotations

import pytest

from src.savetax.config import EvidenceConfig, VariantPolicy
from src.savetax.errors import ConfigError


def test_variant_policy_defaults_to_strict() -> None:
    assert EvidenceConfig().variant_policy is VariantPolicy.STRICT


def test_invalid_variant_policy_fails_closed() -> None:
    with pytest.raises(ConfigError, match="variant_policy"):
        EvidenceConfig(variant_policy="lenient")  # type: ignore[arg-type]
