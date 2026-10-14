"""Credential-less page fetcher restricted to configured public hosts."""

from __future__ import annotations

import html as html_module
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse

from src.savetax.config import EvidenceConfig, SaveTaxConfig, _check_url
from src.savetax.errors import ConfigError, NetworkError
from src.savetax.http import Transport, classify_failure, urllib_transport

logger = logging.getLogger(__name__)

__all__ = ["PublicFetcher", "PublicPage", "visible_text"]

_ACCEPT_LANGUAGE = "ko-KR,ko;q=0.9"

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PublicPage:
    """One credential-less page read: status, decoded text, and the UTC fetch time."""

    url: str
    status: int
    text: str
    fetched_at: datetime
    raw: bytes = b""
    content_type: str | None = None


def visible_text(html: str) -> str:
    """Return the page text with scripts, styles and tags removed and whitespace collapsed."""
    without_code = _SCRIPT_STYLE_RE.sub(" ", html)
    without_tags = _TAG_RE.sub(" ", without_code)
    return _WHITESPACE_RE.sub(" ", html_module.unescape(without_tags)).strip()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _response_charset(content_type: str) -> str:
    for part in content_type.split(";"):
        name, separator, value = part.partition("=")
        if separator and name.strip().lower() == "charset" and value.strip():
            return value.strip().strip("\"'")
    return "utf-8"


class PublicFetcher:
    """Credential-less GET client restricted to the configured public hosts."""

    def __init__(
        self,
        config: SaveTaxConfig,
        evidence: EvidenceConfig,
        *,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._config = config
        self._evidence = evidence
        self._transport = transport
        self._sleep = sleep
        self._clock = clock
        self._now = now
        self._last_seen: dict[str, float] = {}

    def get(self, url: str, *, referer: str | None = None) -> PublicPage:
        """GET url without credentials; raises ConfigError for a disallowed URL and NetworkError after one retry."""
        _check_url(url, "public URL")
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if host not in self._evidence.public_hosts():
            raise ConfigError("[SYS] refusing public URL outside the allowlist")
        if self._config.allows_credential(url):
            raise ConfigError("[SYS] refusing credential destination for a public fetch")
        headers: dict[str, str] = {
            "User-Agent": self._evidence.public_user_agent,
            "Accept-Language": _ACCEPT_LANGUAGE,
        }
        if referer is not None:
            headers["Referer"] = referer
            headers["X-Requested-With"] = "XMLHttpRequest"
        for attempt in range(2):
            self._pace(host)
            try:
                response = self._transport(url, headers, self._config.timeout_seconds)
                break
            except (NetworkError, OSError) as error:
                if attempt == 1:
                    if isinstance(error, NetworkError):
                        raise
                    raise NetworkError(classify_failure(error), host, type(error).__name__) from None
                self._sleep(self._config.retry_cooldown_seconds)
        if 300 <= response.status <= 399:
            logger.info("[SYS] public GET host=%s status=%s", host, response.status)
            return PublicPage(url=url, status=response.status, text="", fetched_at=self._now(), raw=b"", content_type=response.headers.get("content-type"))
        try:
            text = response.body.decode(_response_charset(response.headers.get("content-type", "")), errors="replace")
        except LookupError:
            text = response.body.decode("utf-8", errors="replace")
        logger.info("[SYS] public GET host=%s status=%s", host, response.status)
        return PublicPage(url=url, status=response.status, text=text, fetched_at=self._now(), raw=response.body, content_type=response.headers.get("content-type"))

    def _pace(self, host: str) -> None:
        interval = self._evidence.public_request_interval_seconds
        tick = self._clock()
        previous = self._last_seen.get(host)
        if previous is not None:
            wait = interval - (tick - previous)
            if wait > 0:
                self._sleep(wait)
                tick = self._clock()
        self._last_seen[host] = tick
