"""POST-only transport for the submission endpoints."""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import replace
from http.client import HTTPException
from urllib.parse import urlparse

from src.savetax.config import SaveTaxConfig, Secret
from src.savetax.errors import ConfigError, DataIntegrityError, DomainValidationError, NetworkError
from src.savetax.http import HttpResponse, classify_failure
from src.savetax.redaction import redact

logger = logging.getLogger(__name__)

WriteTransport = Callable[[str, str, Mapping[str, str], bytes | None, float], HttpResponse]

_SID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_KEY_RE = re.compile(r"^[\x21-\x7e]{1,128}$")
_MAX_BODY_BYTES = 1_000_000


class _NoRedirect(urllib.request.HTTPRedirectHandler):
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


_opener = urllib.request.build_opener(_NoRedirect)


def _read_body(read: Callable[[], bytes]) -> bytes:
    try:
        return read()
    except (OSError, HTTPException):
        raise DataIntegrityError("[SYS] HTTP response body could not be read completely") from None


def urllib_write_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> HttpResponse:
    """Perform one POST without following redirects."""
    if method != "POST":
        raise ConfigError("[SYS] write transport requires POST")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")  # noqa: S310
    parsed = urlparse(url)
    host = parsed.hostname or parsed.netloc or "unknown"
    started = time.monotonic()
    try:
        with closing(_opener.open(request, timeout=timeout)) as response:  # noqa: S310
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
    except (OSError, HTTPException) as exc:
        token = headers.get("Authorization", "").removeprefix("Bearer ")
        detail = str(exc).replace(token, "<redacted>") if token else str(exc)
        raise NetworkError(classify_failure(exc), host, f"{type(exc).__name__}: {detail}") from None


def _kind_and_log_path(path: str) -> tuple[str, str]:
    if path == "/submissions/preview":
        return "preview", "/submissions/preview"
    if path == "/submissions":
        return "submit", "/submissions"
    if path.startswith("/submissions/") and path.endswith("/revisions"):
        sid = path[len("/submissions/") : -len("/revisions")]
        if sid and _SID_RE.fullmatch(sid):
            return "revise", "/submissions/revisions"
    if path.startswith("/submissions/") and path.endswith("/withdraw"):
        sid = path[len("/submissions/") : -len("/withdraw")]
        if sid and _SID_RE.fullmatch(sid):
            return "withdraw", "/submissions/withdraw"
    raise ConfigError("[SYS] refusing non-submission path")


class WriteHttpClient:
    """POST-only client for the competition submission paths."""

    def __init__(
        self,
        config: SaveTaxConfig,
        token: Secret,
        *,
        transport: WriteTransport = urllib_write_transport,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._token = token
        self._transport = transport
        self._sleep = sleep

    def post_json(
        self,
        path: str,
        body: Mapping[str, object] | None,
        *,
        idempotency_key: str,
        if_match: int | None = None,
    ) -> HttpResponse:
        """POST canonical JSON to an allowlisted submission path."""
        kind, log_path = _kind_and_log_path(path)
        if not _KEY_RE.fullmatch(idempotency_key):
            raise ConfigError("[SYS] invalid idempotency key")
        if if_match is not None and (isinstance(if_match, bool) or not isinstance(if_match, int)):
            raise ConfigError("[SYS] invalid If-Match value")
        payload: bytes | None = None
        if body is not None:
            try:
                text: str = json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)
                payload = text.encode("utf-8")
            except (TypeError, ValueError, UnicodeError):
                raise DataIntegrityError("[DATA] submission body is not JSON serializable") from None
            if len(payload) > _MAX_BODY_BYTES:
                raise DomainValidationError("[DATA] submission body exceeds the size limit")
        base = self._config.competition_base_url.rstrip("/")
        url = base + "/events/current" + path
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "hackathon.savetax.now"
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith("/v1/")
            or not self._config.allows_credential(url)
        ):
            raise ConfigError("[SYS] refusing credential for a non-allowlisted destination")
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._token.reveal()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self._config.user_agent,
        }
        if kind != "preview":
            headers["Idempotency-Key"] = idempotency_key
        if if_match is not None:
            headers["If-Match"] = str(if_match)
        token = headers["Authorization"].removeprefix("Bearer ")
        if token and (token in url or token in idempotency_key or (payload is not None and token in payload.decode())):
            raise ConfigError("[SYS] credential must occur only in the Authorization header")
        host = parsed.hostname or parsed.netloc or "unknown"
        try:
            response = self._transport("POST", url, headers, payload, self._config.timeout_seconds)
        except NetworkError as exc:
            raise NetworkError(exc.stage, host, exc.detail.replace(token, "<redacted>")) from None
        except (OSError, HTTPException) as exc:
            raise NetworkError(
                classify_failure(exc), host, f"{type(exc).__name__}: {str(exc).replace(token, '<redacted>')}"
            ) from None
        response = replace(
            response,
            body=response.body.replace(token.encode("utf-8"), b"<redacted>"),
            headers={
                key.replace(token, "<redacted>"): value.replace(token, "<redacted>")
                for key, value in response.headers.items()
            },
        )
        logger.info(
            "[SUBMIT] POST host=%s path=%s status=%d",
            redact(host),
            log_path,
            response.status,
        )
        return response
