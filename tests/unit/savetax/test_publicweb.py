"""Invariant guards for the credential-less public page fetcher."""

from __future__ import annotations

import errno
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

import pytest

from src.savetax.config import EvidenceConfig, SaveTaxConfig
from src.savetax.errors import ConfigError, NetworkError, NetworkStage
from src.savetax.http import HttpResponse
from src.savetax.publicweb import PublicFetcher, PublicPage, visible_text

_NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)


def _evidence(**over: object) -> EvidenceConfig:
    return EvidenceConfig(**over)  # type: ignore[arg-type]


def _ok(
    seen: list,
    status: int = 200,
    body: bytes = b"{}",
    headers: Mapping[str, str] | None = None,
) -> Callable:
    def run(url: str, hdrs: Mapping[str, str], timeout: float) -> HttpResponse:
        seen.append((url, dict(hdrs), timeout))
        return HttpResponse(status=status, headers=dict(headers or {}), body=body, elapsed_seconds=0.01)

    return run


def _fetcher(seen: list, config: SaveTaxConfig | None = None, transport: Callable | None = None) -> PublicFetcher:
    return PublicFetcher(
        config or SaveTaxConfig(),
        _evidence(),
        transport=transport or _ok(seen),  # type: ignore[arg-type]
        sleep=lambda _s: None,
        clock=lambda: 0.0,
    )


def test_no_credential_header_is_ever_sent() -> None:
    seen: list = []
    fetch = _fetcher(seen)
    page = fetch.get("https://savetax.now/listing/A")
    assert isinstance(page, PublicPage)
    assert set(seen[0][1]) == {"User-Agent", "Accept-Language"}
    assert "Authorization" not in seen[0][1]


def test_referer_adds_the_xhr_header_pair() -> None:
    seen: list = []
    fetch = _fetcher(seen)
    fetch.get("https://prod.danawa.com/info/?pcode=1", referer="https://prod.danawa.com/info/?pcode=1")
    assert set(seen[0][1]) == {"User-Agent", "Accept-Language", "Referer", "X-Requested-With"}
    assert seen[0][1]["X-Requested-With"] == "XMLHttpRequest"


def test_host_outside_the_allowlist_is_refused() -> None:
    seen: list = []
    fetch = _fetcher(seen)
    with pytest.raises(ConfigError):
        fetch.get("https://www.coupang.com/x")
    assert seen == []


def test_credential_destination_is_refused() -> None:
    seen: list = []

    class _OpenEvidence:
        public_user_agent = "test-agent"
        public_request_interval_seconds = 0.0

        def public_hosts(self) -> frozenset:
            return frozenset({"api.savetax.now"})

    fetch = PublicFetcher(
        SaveTaxConfig(),
        _OpenEvidence(),  # type: ignore[arg-type]
        transport=_ok(seen),  # type: ignore[arg-type]
        sleep=lambda _s: None,
        clock=lambda: 0.0,
        now=lambda: _NOW,
    )
    with pytest.raises(ConfigError):
        fetch.get("https://api.savetax.now/v1/me")
    assert seen == []


def test_plain_http_is_refused() -> None:
    seen: list = []
    fetch = _fetcher(seen)
    with pytest.raises(ConfigError):
        fetch.get("http://savetax.now/listing/A")
    assert seen == []


def test_same_host_is_paced() -> None:
    seen: list = []
    sleeps: list[float] = []
    ticks = iter([100.0, 100.2, 101.0, 105.0])
    fetch = PublicFetcher(
        SaveTaxConfig(),
        _evidence(public_request_interval_seconds=1.0),
        transport=_ok(seen),  # type: ignore[arg-type]
        sleep=sleeps.append,
        clock=lambda: next(ticks),
        now=lambda: _NOW,
    )
    fetch.get("https://savetax.now/listing/A")
    fetch.get("https://savetax.now/listing/B")
    fetch.get("https://savetax.now/listing/C")
    assert sleeps == [pytest.approx(0.8)]


def test_different_hosts_are_not_paced_against_each_other() -> None:
    seen: list = []
    sleeps: list[float] = []
    ticks = iter([100.0, 100.2])
    fetch = PublicFetcher(
        SaveTaxConfig(),
        _evidence(public_request_interval_seconds=1.0),
        transport=_ok(seen),  # type: ignore[arg-type]
        sleep=sleeps.append,
        clock=lambda: next(ticks),
        now=lambda: _NOW,
    )
    fetch.get("https://savetax.now/listing/A")
    fetch.get("https://prod.danawa.com/info/?pcode=1")
    assert sleeps == []


def test_transient_failure_retries_once() -> None:
    calls: list[str] = []
    sleeps: list[float] = []
    attempts = {"n": 0}
    tick = [0.0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        tick[0] += seconds

    def flaky(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise NetworkError(NetworkStage.TCP_TLS_CONNECT, "savetax.now", "boom")
        return HttpResponse(status=200, headers={}, body=b"ok", elapsed_seconds=0.01)

    fetch = PublicFetcher(
        SaveTaxConfig(),
        _evidence(),
        transport=flaky,  # type: ignore[arg-type]
        sleep=sleep,
        clock=lambda: tick[0],
        now=lambda: _NOW,
    )
    page = fetch.get("https://savetax.now/listing/A")
    assert page.status == 200
    assert page.text == "ok"
    assert sleeps == [15.0]
    assert len(calls) == 2


def test_second_failure_propagates() -> None:
    calls: list[str] = []

    def down(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        raise NetworkError(NetworkStage.TCP_TLS_CONNECT, "savetax.now", "down")

    fetch = PublicFetcher(
        SaveTaxConfig(),
        _evidence(),
        transport=down,  # type: ignore[arg-type]
        sleep=lambda _s: None,
        clock=lambda: 0.0,
        now=lambda: _NOW,
    )
    with pytest.raises(NetworkError):
        fetch.get("https://savetax.now/listing/A")
    assert len(calls) == 2


def test_sandbox_os_error_is_classified() -> None:
    calls: list[str] = []

    def blocked(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        raise PermissionError(errno.EPERM, "Operation not permitted")

    fetch = PublicFetcher(
        SaveTaxConfig(),
        _evidence(),
        transport=blocked,  # type: ignore[arg-type]
        sleep=lambda _s: None,
        clock=lambda: 0.0,
        now=lambda: _NOW,
    )
    with pytest.raises(NetworkError) as exc:
        fetch.get("https://savetax.now/listing/A")
    assert exc.value.stage is NetworkStage.SANDBOX_POLICY
    assert len(calls) == 2


def test_non_200_is_returned() -> None:
    seen: list = []
    fetch = _fetcher(seen, transport=_ok(seen, status=404, body=b"missing"))
    page = fetch.get("https://savetax.now/listing/A")
    assert page.status == 404
    assert page.fetched_at.tzinfo is not None


def test_redirect_is_returned_with_empty_text() -> None:
    seen: list = []
    fetch = _fetcher(seen, transport=_ok(seen, status=302))
    page = fetch.get("https://savetax.now/listing/A")
    assert page.status == 302
    assert page.text == ""


def test_charset_is_honoured() -> None:
    seen: list = []
    body = "가격".encode("euc-kr")
    fetch = _fetcher(seen, transport=_ok(seen, body=body, headers={"content-type": "text/html; charset=euc-kr"}))
    assert fetch.get("https://savetax.now/listing/A").text == "가격"


def test_unknown_charset_falls_back_to_utf8() -> None:
    seen: list = []
    fetch = _fetcher(seen, transport=_ok(seen, body="가격".encode(), headers={"content-type": "text/html; charset=x-unknown"}))
    assert fetch.get("https://savetax.now/listing/A").text == "가격"


def test_visible_text_strips_scripts() -> None:
    assert visible_text('<p>가격 <b>1,000</b>원</p><script>var x="9,999원"</script>') == "가격 1,000 원"


@pytest.mark.parametrize("url", ["https://user:pass@savetax.now/listing/A", "https://[broken", "https://savetax.now:444/listing/A"])
def test_unsafe_public_url_is_refused_before_io(url: str) -> None:
    seen: list = []
    with pytest.raises(ConfigError):
        _fetcher(seen).get(url)
    assert seen == []


def test_retry_and_following_request_obey_host_interval() -> None:
    tick = [0.0]
    starts: list[float] = []

    def sleep(seconds: float) -> None:
        tick[0] += seconds

    def transport(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        starts.append(tick[0])
        if len(starts) == 1:
            raise OSError("temporary")
        return HttpResponse(200, {}, b"ok", 0.0)

    fetch = PublicFetcher(
        SaveTaxConfig(retry_cooldown_seconds=0.2),
        EvidenceConfig(),
        transport=transport,
        sleep=sleep,
        clock=lambda: tick[0],
    )
    fetch.get("https://savetax.now/listing/A")
    fetch.get("https://savetax.now/listing/B")
    assert starts == [0.0, 1.0, 2.0]


def test_mixed_transport_failures_are_classified() -> None:
    failures = iter([NetworkError(NetworkStage.DNS_RESOLUTION, "savetax.now", "dns"), PermissionError("blocked")])
    seen: list = []

    def transport(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        seen.append(url)
        raise next(failures)

    with pytest.raises(NetworkError) as error:
        _fetcher(seen, transport=transport).get("https://savetax.now/listing/A")
    assert error.value.stage is NetworkStage.SANDBOX_POLICY
    assert len(seen) == 2
