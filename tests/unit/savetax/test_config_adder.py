"""Guard tests for recipe adder source URLs."""

from __future__ import annotations

import pytest

from src.savetax.config import EvidenceConfig, RecipeAdder
from src.savetax.errors import ConfigError


def test_rejects_unsafe_adder_url() -> None:
    with pytest.raises(ConfigError, match="invalid recipe_adders: source_url"):
        EvidenceConfig(recipe_adders=(RecipeAdder("k", "라벨", 100, "basis", "http://x"),))


def test_rejects_non_string_adder_url() -> None:
    with pytest.raises(ConfigError, match="invalid recipe_adders: source_url"):
        EvidenceConfig(recipe_adders=(RecipeAdder("k", "라벨", 100, "basis", 123),))  # type: ignore[arg-type]


def test_rejects_malformed_adder_url() -> None:
    with pytest.raises(ConfigError, match="invalid recipe_adders: source_url"):
        EvidenceConfig(recipe_adders=(RecipeAdder("k", "라벨", 100, "basis", "https://savetax.now/a b"),))


def test_default_sourced_adders() -> None:
    cfg = EvidenceConfig()
    by_key = {a.key: a for a in cfg.recipe_adders}
    assert by_key["panel:labor-terminated"].source_url == "https://savetax.now/222"
    assert by_key["panel:labor-mounted"].source_url == "https://savetax.now/621"
    assert by_key["rack:labor"].source_url == "https://savetax.now/176"
