"""Invariant guards for endpoints, thresholds, and credential loading."""

import pickle
from decimal import Decimal
from pathlib import Path

import pytest

from src.savetax.config import AuditConfig, EvidenceConfig, SaveTaxConfig, Secret, ShopSpec, load_token
from src.savetax.errors import ConfigError, CredentialError


def test_secret_never_renders() -> None:
    secret = Secret("savetax_x-1")
    assert repr(secret) == "Secret(***)"
    assert str(secret) == "Secret(***)"
    assert f"{secret}" == "Secret(***)"
    assert "%s" % secret == "Secret(***)"  # noqa: UP031
    assert secret.reveal() == "savetax_x-1"


def test_secret_refuses_pickling() -> None:
    secret = Secret("savetax_x-1")
    with pytest.raises(TypeError):
        pickle.dumps(secret)
    with pytest.raises(TypeError):
        secret.__reduce__()
    with pytest.raises(TypeError):
        secret.__reduce_ex__(4)


def _cfg(**over: object) -> SaveTaxConfig:
    return SaveTaxConfig(**over)  # type: ignore[arg-type]


def test_allowlist_accepts_protected_destinations() -> None:
    cfg = SaveTaxConfig()
    assert cfg.allows_credential("https://hackathon.savetax.now/v1/events/current/me") is True
    assert cfg.allows_credential("https://api.savetax.now/v1/me") is True
    assert cfg.allows_credential(cfg.dataset_zip_url) is True


def test_allowlist_rejects_public_and_third_party() -> None:
    cfg = SaveTaxConfig()
    denied = [
        cfg.guidance_manifest_url,
        "https://hackathon.savetax.now/docs/index.md",
        "https://hackathon.savetax.now/",
        "https://savetax.now/listing/x",
        "https://mcp.ai.cowave.kr/mcp",
        "https://other-host.workers.dev/latest/savetax-products.zip",
        "http://hackathon.savetax.now/v1/me",
        "https://user:pass@hackathon.savetax.now/v1/me",
        "https://hackathon.savetax.now/v1/../docs/x",
        "https://hackathon.savetax.now/v1/me#frag",
    ]
    for url in denied:
        assert cfg.allows_credential(url) is False, url


def test_dataset_version_derives_from_directory() -> None:
    cfg = _cfg(dataset_dir=Path("data/savetax-products-20261010-r5"))
    assert cfg.dataset_version == "20261010-r5"
    assert cfg.dataset_parquet_path == Path("data/savetax-products-20261010-r5/listings.parquet")
    with pytest.raises(ConfigError):
        _ = _cfg(dataset_dir=Path("data/foo")).dataset_version
    with pytest.raises(ConfigError):
        _ = _cfg(dataset_dir=Path("data/savetax-products-")).dataset_version


def test_invalid_configuration_fails_closed() -> None:
    cases = [
        {"competition_base_url": "http://hackathon.savetax.now/v1"},
        {"raw_api_base_url": "http://api.savetax.now/v1"},
        {"dataset_zip_url": "https://hackathon.savetax.now/v1/x#frag"},
        {"guidance_manifest_url": "https://u:p@hackathon.savetax.now/docs/agent-guidance-version.json"},
        {"competition_base_url": "https://hackathon.savetax.now/v1/../x"},
        {"user_agent": ""},
        {"timeout_seconds": 0.0},
        {"retry_cooldown_seconds": -1.0},
        {"rate_limit_max_wait_seconds": 0.0},
        {"dotenv_keys": ()},
    ]
    for kw in cases:
        with pytest.raises(ConfigError):
            _cfg(**kw)
    with pytest.raises(ConfigError):
        AuditConfig(markup_threshold=Decimal("0.9"))
    with pytest.raises(ConfigError):
        AuditConfig(min_item_order_amount_krw=-1)
    with pytest.raises(ConfigError):
        AuditConfig(min_unit_price_krw=Decimal("0"))
    with pytest.raises(ConfigError):
        AuditConfig(min_model_alnum_length=0)
    with pytest.raises(ConfigError):
        AuditConfig(batch_size=0)
    with pytest.raises(ConfigError):
        AuditConfig(max_condition_listings=0)
    with pytest.raises(ConfigError):
        AuditConfig(progress_every=0)
    with pytest.raises(ConfigError):
        AuditConfig(comparable_large_categories=())


def test_environment_variable_wins_over_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("SAVETAX_TOKEN=savetax_dotenv-1\n", encoding="utf-8")
    cfg = _cfg(dotenv_path=dotenv)
    secret = load_token(cfg, {"SAVETAX_TOKEN": "savetax_env-1"})
    assert secret.reveal() == "savetax_env-1"


def test_dotenv_fallback_honours_key_order(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("# comment\ntok='savetax_a-b'\n", encoding="utf-8")
    cfg = _cfg(dotenv_path=dotenv)
    assert load_token(cfg, {}).reveal() == "savetax_a-b"


def test_malformed_dotenv_token_raises_without_echo(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("tok=abc\n", encoding="utf-8")
    cfg = _cfg(dotenv_path=dotenv)
    with pytest.raises(CredentialError) as exc:
        load_token(cfg, {})
    assert "abc" not in str(exc.value)


def test_missing_or_malformed_token_raises_without_echo(tmp_path: Path) -> None:
    cfg = _cfg(dotenv_path=tmp_path / ".env")
    with pytest.raises(CredentialError):
        load_token(cfg, {})
    for bad in ("abc", "savetax_ab cd", 'savetax_ab"cd', "savetax_ab<cd"):
        with pytest.raises(CredentialError) as exc:
            load_token(cfg, {"SAVETAX_TOKEN": bad})
        assert bad not in str(exc.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://hackathon.savetax.now/v1/%2e%2e/docs/x",
        "https://hackathon.savetax.now/v1/%252e%252e/docs/x",
        "https://hackathon.savetax.now/v1/me#",
        "https://hackathon.savetax.now/v1/\nme",
        "https://hackathon.savetax.now/v1/\\..\\docs",
        "https://[invalid/v1/me",
        "https://hackathon.savetax.now:444/v1/me",
    ],
)
def test_unsafe_credential_urls_fail_closed(url: str) -> None:
    assert not SaveTaxConfig().allows_credential(url)
    with pytest.raises(ConfigError):
        SaveTaxConfig(competition_base_url=url)


def test_configuration_cannot_expand_credential_boundary() -> None:
    cfg = SaveTaxConfig(competition_base_url="https://evil.example/v1", dataset_zip_url="https://evil.example/a.zip")
    assert not cfg.allows_credential("https://evil.example/v1/me")
    assert not cfg.allows_credential(cfg.dataset_zip_url)
    cfg = SaveTaxConfig(competition_base_url="https://hackathon.savetax.now/docs")
    assert not cfg.allows_credential("https://hackathon.savetax.now/docs/x")


@pytest.mark.parametrize("field", ["timeout_seconds", "retry_cooldown_seconds", "rate_limit_max_wait_seconds"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_timeouts_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(ConfigError):
        SaveTaxConfig(**{field: value})


@pytest.mark.parametrize("field", ["markup_threshold", "min_unit_price_krw"])
@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity")])
def test_thresholds_must_be_finite(field: str, value: Decimal) -> None:
    with pytest.raises(ConfigError):
        AuditConfig(**{field: value})


def test_empty_first_dotenv_key_does_not_fall_through(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("SAVETAX_TOKEN=\ntok=savetax_fallback\n", encoding="utf-8")
    with pytest.raises(CredentialError, match="dotenv"):
        load_token(SaveTaxConfig(dotenv_path=dotenv), {})


def test_unreadable_dotenv_is_a_credential_error(tmp_path: Path) -> None:
    with pytest.raises(CredentialError, match="dotenv"):
        load_token(SaveTaxConfig(dotenv_path=tmp_path), {})


def _evidence(**over: object) -> EvidenceConfig:
    return EvidenceConfig(**over)  # type: ignore[arg-type]


def test_evidence_defaults_are_valid() -> None:
    cfg = EvidenceConfig()
    assert cfg.cross_markup_threshold == Decimal("1.10")
    assert cfg.min_item_expected_krw == 300000
    hosts = cfg.public_hosts()
    for host in ("savetax.now", "prod.danawa.com", "search.danawa.com", "itncctv.kr"):
        assert host in hosts


def test_evidence_threshold_below_one_is_rejected() -> None:
    with pytest.raises(ConfigError):
        _evidence(cross_markup_threshold=Decimal("0.99"))


def test_evidence_template_without_placeholder_is_rejected() -> None:
    with pytest.raises(ConfigError):
        _evidence(listing_page_url_template="https://savetax.now/listing/")


def test_evidence_plain_http_template_is_rejected() -> None:
    with pytest.raises(ConfigError):
        _evidence(danawa_search_url_template="http://search.danawa.com/?query={query}")


def test_evidence_credential_host_cannot_be_a_public_host() -> None:
    with pytest.raises(ConfigError):
        _evidence(listing_page_url_template="https://api.savetax.now/v1/listings/{listing_id}")


def test_evidence_unknown_shop_platform_is_rejected() -> None:
    with pytest.raises(ConfigError):
        _evidence(shops=(ShopSpec("a.example", "shopify"),))


@pytest.mark.parametrize(
    "kw",
    [
        {"cross_markup_threshold": Decimal("NaN")},
        {"retail_markup_threshold": Decimal("0.5")},
        {"install_allowance_rate": Decimal("-0.01")},
        {"min_offer_count": 0},
        {"outlier_ratio": Decimal("0.9")},
        {"single_offer_stability_ratio": Decimal("0.9")},
        {"oem_gap_cap": Decimal("0.9")},
        {"min_item_expected_krw": -1},
        {"max_benchmarks_per_item": 0},
        {"max_lines_per_draft": 0},
        {"max_draft_bytes": 0},
        {"listing_page_url_template": "https:///{listing_id}"},
        {"shops": (ShopSpec("API.SAVETAX.NOW", "cafe24"),)},
        {"shops": (ShopSpec("api.savetax.now.", "cafe24"),)},
        {"shops": (ShopSpec("user@store.example", "cafe24"),)},
        {"shops": (ShopSpec("store.example:443", "cafe24"),)},
        {"public_request_interval_seconds": -0.5},
        {"danawa_product_url_template": "https://prod.danawa.com/info/"},
        {"danawa_history_url_template": "http://prod.danawa.com/info/ajax/getProductPriceList.ajax.php?productCode={pcode}"},
        {"shops": (ShopSpec("", "cafe24"),)},
        {"shops": (ShopSpec("a/b.example", "cafe24"),)},
        {"shops": (ShopSpec("x.workers.dev", "cafe24"),)},
        {"listing_page_url_template": "https://hackathon.savetax.now/v1/listings/{listing_id}"},
    ],
)
def test_evidence_invalid_configuration_fails_closed(kw: object) -> None:
    with pytest.raises(ConfigError):
        _evidence(**kw)  # type: ignore[arg-type]


def test_unit_policy_defaults_to_annotate() -> None:
    from src.savetax.config import UnitConflictPolicy

    assert EvidenceConfig().unit_conflict_policy is UnitConflictPolicy.ANNOTATE


def test_invalid_unit_policy_fails_closed() -> None:
    with pytest.raises(ConfigError, match="unit_conflict_policy"):
        _evidence(unit_conflict_policy="ignore")
@pytest.mark.parametrize("encoded", ["", "../x", "%", "%FF", "has%20space", "A/B", "..", "K?query"])
def test_request_key_path_requires_canonical_encoding(encoded: str) -> None:
    assert not SaveTaxConfig().allows_credential(
        "https://hackathon.savetax.now/v1/events/current/submission-requests/" + encoded
    )
