"""Invariant guards for the domain error hierarchy."""

import pytest

from src.savetax.errors import ApiError, NetworkError, NetworkStage
from src.savetax.errors import (
    ConfigError,
    CredentialError,
    DataAccessPending,
    DataIntegrityError,
    DomainValidationError,
)


@pytest.mark.parametrize(
    ("stage", "sandbox"),
    [
        (NetworkStage.DNS_RESOLUTION, False),
        (NetworkStage.TCP_TLS_CONNECT, False),
        (NetworkStage.SANDBOX_POLICY, True),
    ],
)
def test_sandbox_flag_follows_stage(stage: NetworkStage, sandbox: bool) -> None:
    err = NetworkError(stage, "api.savetax.now", "boom")
    assert err.sandbox is sandbox
    assert err.stage is stage


def test_error_text_never_carries_token() -> None:
    token = "savetax_Ab-Cd_ef12"
    net = NetworkError(NetworkStage.TCP_TLS_CONNECT, "h", f"down {token} x")
    api = ApiError("competition", 429, "RATE_LIMITED", f"slow {token}", True)
    for err in (net, api):
        assert token not in str(err)
        assert "<redacted>" in str(err)


def test_api_error_preserves_code_casing() -> None:
    assert ApiError("competition", 429, "RATE_LIMITED", "m", True).code == "RATE_LIMITED"
    assert ApiError("raw", 404, "not_found", "m", False).code == "not_found"


@pytest.mark.parametrize(
    "error_type", [ConfigError, CredentialError, DataAccessPending, DataIntegrityError, DomainValidationError]
)
def test_leaf_errors_redact_arguments(error_type: type[Exception]) -> None:
    error = error_type("invalid savetax_Ab-cD_12")
    assert error.args == ("invalid <redacted>",)


def test_error_metadata_cannot_leak_tokens() -> None:
    token = "savetax_Ab-cD_12"
    net = NetworkError(NetworkStage.DNS_RESOLUTION, token, token)
    api = ApiError(token, 400, token, token, False)
    for error in (net, api):
        assert token not in repr(error)
        assert token not in str(vars(error))


def test_api_context_retains_structure_and_redacts_nested_tokens() -> None:
    token = "savetax_secret_123"
    error = ApiError("competition", 409, "PRODUCT_ALREADY_CLAIMED", "taken", False,
                     request_id=token, context={"claimed_by_me": True, "nested": [{token: token}]})
    assert error.context == {"claimed_by_me": True, "nested": [{"<redacted>": "<redacted>"}]}
    assert error.request_id == "<redacted>"
    assert token not in str(vars(error))
