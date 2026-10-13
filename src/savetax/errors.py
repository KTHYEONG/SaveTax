"""Domain error hierarchy with credential-safe messages."""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping

from src.savetax import redaction

logger = logging.getLogger(__name__)


class SaveTaxError(Exception):
    """Base class for every domain error raised by this package."""

    def __init__(self, message: str) -> None:
        super().__init__(redaction.redact(message))


class ConfigError(SaveTaxError):
    """A configuration or domain input violates a declared constraint."""


class DomainValidationError(SaveTaxError):
    """A price, quantity, or amount violates a domain invariant."""


class CredentialError(SaveTaxError):
    """The participant token is missing, malformed, or bound for a forbidden destination."""


class DataIntegrityError(SaveTaxError):
    """A dataset or API payload is malformed and was rejected instead of repaired."""


class DataAccessPending(SaveTaxError):  # noqa: N818
    """A required data source is unreachable; candidate search must pause."""


class SubmissionPending(SaveTaxError):  # noqa: N818
    """An idempotent request is still pending; resolve it later with the same key."""

    def __init__(self, key: str) -> None:
        self.key = redaction.redact(key)
        super().__init__("[SUBMIT] request is still pending")


class NetworkStage(enum.Enum):
    """Connection stage at which a request failed."""

    DNS_RESOLUTION = "DNS_RESOLUTION"
    TCP_TLS_CONNECT = "TCP_TLS_CONNECT"
    SANDBOX_POLICY = "SANDBOX_POLICY"


class NetworkError(SaveTaxError):
    """A request failed before any HTTP response; carries stage, host, and sandbox."""

    def __init__(self, stage: NetworkStage, host: str, detail: str) -> None:
        self.stage = stage
        self.host = redaction.redact(host)
        self.sandbox = stage is NetworkStage.SANDBOX_POLICY
        self.detail = redaction.redact(detail)
        super().__init__(f"[{stage.value}] host={self.host}: {self.detail}")

    def __str__(self) -> str:
        return f"[{self.stage.value}] host={self.host}: {self.detail}"


class ApiError(SaveTaxError):
    """A service answered with an error; carries service, status, code, retryable."""

    def __init__(
        self,
        service: str,
        status: int,
        code: str,
        message: str,
        retryable: bool,
        *,
        request_id: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        self.service = redaction.redact(service)
        self.status = status
        self.code = redaction.redact(code)
        self.message = redaction.redact(message)
        self.retryable = retryable
        self.request_id = redaction.redact(request_id) if request_id is not None else None
        self.context = (
            {redaction.redact(key): _redact_context(value) for key, value in context.items()}
            if context is not None else None
        )
        super().__init__(f"[{self.service}] {status} {self.code}: {self.message}")

    def __str__(self) -> str:
        return f"[{self.service}] {self.status} {self.code}: {self.message}"


def _redact_context(value: object) -> object:
    if isinstance(value, str):
        return redaction.redact(value)
    if isinstance(value, dict):
        return {redaction.redact(str(key)): _redact_context(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_context(item) for item in value]
    return value
