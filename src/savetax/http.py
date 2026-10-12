"""Credential-safe GET-only HTTP layer with staged failure classification."""

from __future__ import annotations

import errno
import logging
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from http.client import HTTPException
from urllib.parse import unquote, urlparse

from src.savetax.config import SaveTaxConfig, Secret
from src.savetax.errors import ConfigError, CredentialError, DataIntegrityError, NetworkError, NetworkStage
from src.savetax.redaction import redact

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HttpResponse:
    """One HTTP exchange: final status, lower-cased header map, raw body, elapsed seconds."""

    status: int
    headers: Mapping[str, str]
    body: bytes
    elapsed_seconds: float


Transport = Callable[[str, Mapping[str, str], float], HttpResponse]


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


opener = urllib.request.build_opener(_NoRedirectHandler)


def classify_failure(exc: BaseException) -> NetworkStage:
    """Map a pre-HTTP transport exception to the connection stage where it failed."""
    target: BaseException = exc
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, BaseException):
            target = reason
        else:
            return NetworkStage.TCP_TLS_CONNECT
    if isinstance(target, socket.gaierror):
        return NetworkStage.DNS_RESOLUTION
    if isinstance(target, PermissionError):
        return NetworkStage.SANDBOX_POLICY
    if isinstance(target, OSError) and target.errno in (errno.EPERM, errno.EACCES):
        return NetworkStage.SANDBOX_POLICY
    return NetworkStage.TCP_TLS_CONNECT


def urllib_transport(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    """Perform one GET without following redirects."""
    request = urllib.request.Request(url, headers=dict(headers), method="GET")  # noqa: S310
    started = time.monotonic()
    try:
        with closing(opener.open(request, timeout=timeout)) as response:  # noqa: S310
            return HttpResponse(
                status=int(response.status),
                headers={str(k).lower(): str(v) for k, v in response.headers.items()},
                body=_read_body(response.read),
                elapsed_seconds=time.monotonic() - started,
            )
    except urllib.error.HTTPError as http_err:
        with closing(http_err):
            payload = _read_body(http_err.read)
            hdrs = http_err.headers.items() if http_err.headers is not None else []
            return HttpResponse(
                status=int(http_err.code),
                headers={str(k).lower(): str(v) for k, v in hdrs},
                body=payload,
                elapsed_seconds=time.monotonic() - started,
            )


def _read_body(read: Callable[[], bytes]) -> bytes:
    try:
        return read()
    except (OSError, HTTPException):
        raise DataIntegrityError("[SYS] HTTP response body could not be read completely") from None


def _host_and_path(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    return parsed.hostname or parsed.netloc, parsed.path or "/"


def _log_attempt(stage: NetworkStage, host: str) -> None:
    logger.warning("[SYS] stage=%s host=%s attempt=1", stage.value, host)
    if stage is NetworkStage.SANDBOX_POLICY:
        logger.warning("[SYS] Network blocked: Operation not permitted (local sandbox policy) host=%s", host)


class HttpClient:
    """GET-only client that attaches the token solely to allowlisted URLs."""

    def __init__(
        self,
        config: SaveTaxConfig,
        token: Secret | None = None,
        *,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._token = token
        self._transport = transport
        self._sleep = sleep

    def get(self, url: str, *, authenticated: bool) -> HttpResponse:
        """GET url, attaching the credential only for allowlisted destinations."""
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ConfigError("[SYS] refusing non-https URL")
        headers: dict[str, str] = {
            "User-Agent": self._config.user_agent,
            "Accept": "application/json",
        }
        if authenticated:
            if self._token is None:
                raise CredentialError("[SYS] authenticated request without a configured token")
            if not self._config.allows_credential(url):
                raise CredentialError("[SYS] refusing credential for a non-allowlisted destination")
            headers["Authorization"] = f"Bearer {self._token.reveal()}"
        if self._token is not None and self._token.reveal() in unquote(url):
            raise CredentialError("[SYS] refusing URL that embeds the credential")
        host, path = _host_and_path(url)
        host, path = redact(host), redact(path)
        try:
            return self._transport(url, headers, self._config.timeout_seconds)
        except OSError as first:
            stage = classify_failure(first)
            _log_attempt(stage, host)
            logger.debug("[SYS] host=%s path=%s retrying after cooldown", host, path)
            self._sleep(self._config.retry_cooldown_seconds)
            try:
                return self._transport(url, headers, self._config.timeout_seconds)
            except OSError as second:
                last_stage = classify_failure(second)
                if last_stage is not stage:
                    _log_attempt(last_stage, host)
                raise NetworkError(last_stage, host, f"{type(second).__name__}: {second}") from None
