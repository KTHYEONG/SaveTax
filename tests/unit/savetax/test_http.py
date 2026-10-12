"""Invariant guards for the credential-safe HTTP layer."""

import errno
import io
import socket
import ssl
import urllib.error
import traceback
from urllib.parse import quote

import pytest

import src.savetax.http as http_mod
from src.savetax.config import SaveTaxConfig, Secret
from src.savetax.errors import ConfigError, CredentialError, DataIntegrityError, NetworkError, NetworkStage
from src.savetax.http import HttpClient, HttpResponse, classify_failure

TOKEN = "savetax_test-token_1"


_DEFAULT_TOKEN = Secret(TOKEN)


def _client(token: Secret | None = None, **kw: object) -> tuple[HttpClient, list[tuple], list[float]]:
    token = _DEFAULT_TOKEN if token is None and not kw.get("no_token") else token
    calls: list[tuple] = []
    sleeps: list[float] = []

    def fake_transport(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append((url, dict(headers), timeout))  # type: ignore[arg-type]
        behaviour = kw.get("behaviour", "ok")
        if behaviour == "ok":
            return HttpResponse(status=200, headers={}, body=b"{}", elapsed_seconds=0.01)
        raise AssertionError("unreachable")

    client = HttpClient(
        SaveTaxConfig(),
        token,
        transport=fake_transport,  # type: ignore[arg-type]
        sleep=sleeps.append,
    )
    return client, calls, sleeps


def test_token_reaches_only_allowlisted_hosts() -> None:
    client, calls, _ = _client()
    client.get("https://hackathon.savetax.now/v1/events/current/me", authenticated=True)
    assert calls[0][1]["Authorization"] == f"Bearer {TOKEN}"


def test_unauthenticated_request_carries_no_credential() -> None:
    client, calls, _ = _client()
    client.get("https://hackathon.savetax.now/v1/events/current/me", authenticated=False)
    assert "Authorization" not in calls[0][1]


def test_forbidden_destination_refused_before_io() -> None:
    client, calls, _ = _client()
    for url in (
        "https://hackathon.savetax.now/docs/agent-guidance-version.json",
        "https://prod.danawa.com/info/?pcode=1",
        "https://savetax.now/listing/x",
    ):
        with pytest.raises(CredentialError):
            client.get(url, authenticated=True)
    assert calls == []


def test_authenticated_call_without_token_refused() -> None:
    client, calls, _ = _client(no_token=True)
    with pytest.raises(CredentialError):
        client.get("https://hackathon.savetax.now/v1/me", authenticated=True)
    assert calls == []


def test_plain_http_refused() -> None:
    client, calls, _ = _client()
    with pytest.raises(ConfigError):
        client.get("http://hackathon.savetax.now/v1/me", authenticated=False)
    assert calls == []


def test_token_in_url_refused() -> None:
    client, calls, _ = _client()
    with pytest.raises(CredentialError):
        client.get(f"https://hackathon.savetax.now/v1/me?q={TOKEN}", authenticated=True)
    assert calls == []


def test_single_retry_after_cooldown_then_success() -> None:
    calls: list[tuple] = []
    sleeps: list[float] = []
    attempts = {"n": 0}

    def flaky(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append((url, timeout))
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise socket.gaierror("temporary failure")
        return HttpResponse(status=200, headers={}, body=b"{}", elapsed_seconds=0.01)

    client = HttpClient(SaveTaxConfig(), Secret(TOKEN), transport=flaky, sleep=sleeps.append)  # type: ignore[arg-type]
    resp = client.get("https://hackathon.savetax.now/v1/me", authenticated=True)
    assert resp.status == 200
    assert sleeps == [15.0]
    assert len(calls) == 2


def test_second_failure_raises_staged_error() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def down(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(1)
        raise ConnectionRefusedError("refused")

    client = HttpClient(SaveTaxConfig(), Secret(TOKEN), transport=down, sleep=sleeps.append)  # type: ignore[arg-type]
    with pytest.raises(NetworkError) as exc:
        client.get("https://hackathon.savetax.now/v1/me", authenticated=True)
    assert exc.value.stage is NetworkStage.TCP_TLS_CONNECT
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_sandbox_block_distinguished_from_outage() -> None:
    def blocked(url: str, headers: object, timeout: float) -> HttpResponse:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    client = HttpClient(SaveTaxConfig(), Secret(TOKEN), transport=blocked, sleep=lambda _s: None)  # type: ignore[arg-type]
    with pytest.raises(NetworkError) as exc:
        client.get("https://hackathon.savetax.now/v1/me", authenticated=True)
    assert exc.value.stage is NetworkStage.SANDBOX_POLICY
    assert exc.value.sandbox is True


def test_failure_classification_table() -> None:
    assert classify_failure(socket.gaierror("x")) is NetworkStage.DNS_RESOLUTION
    assert classify_failure(urllib.error.URLError(socket.gaierror("x"))) is NetworkStage.DNS_RESOLUTION
    assert classify_failure(ssl.SSLError("boom")) is NetworkStage.TCP_TLS_CONNECT
    assert classify_failure(TimeoutError("t")) is NetworkStage.TCP_TLS_CONNECT
    assert classify_failure(OSError(errno.EACCES, "denied")) is NetworkStage.SANDBOX_POLICY
    assert classify_failure(urllib.error.URLError("text")) is NetworkStage.TCP_TLS_CONNECT
    errno_only = OSError("sandboxed open")
    errno_only.errno = errno.EPERM
    assert classify_failure(errno_only) is NetworkStage.SANDBOX_POLICY


def test_stage_change_between_attempts_reports_last_stage() -> None:
    attempts = {"n": 0}

    def shifting(url: str, headers: object, timeout: float) -> HttpResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise socket.gaierror("dns down")
        raise PermissionError(errno.EPERM, "Operation not permitted")

    client = HttpClient(SaveTaxConfig(), Secret(TOKEN), transport=shifting, sleep=lambda _s: None)  # type: ignore[arg-type]
    with pytest.raises(NetworkError) as exc:
        client.get("https://hackathon.savetax.now/v1/me", authenticated=True)
    assert exc.value.stage is NetworkStage.SANDBOX_POLICY


def test_redirect_handler_refuses_redirects() -> None:
    handler = http_mod._NoRedirectHandler()
    assert handler.redirect_request(object(), object(), 302, "Found", object(), "https://x/") is None


def test_http_error_statuses_are_responses_not_retries() -> None:
    calls: list[str] = []
    sleeps: list[float] = []
    pending = [401, 403, 404, 500]

    def answering(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append(url)
        return HttpResponse(status=pending[len(calls) - 1], headers={}, body=b"{}", elapsed_seconds=0.01)

    for expected in (401, 403, 404, 500):
        client = HttpClient(
            SaveTaxConfig(),
            Secret(TOKEN),
            transport=answering,  # type: ignore[arg-type]
            sleep=sleeps.append,
        )
        resp = client.get("https://hackathon.savetax.now/v1/me", authenticated=False)
        assert resp.status == expected
    assert len(calls) == 4
    assert sleeps == []


def test_redirect_returned_not_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = {"n": 0}

    def fake_open(req: object, timeout: float = 0) -> object:
        opened["n"] += 1
        raise urllib.error.HTTPError(
            "https://hackathon.savetax.now/v1/me", 302, "Found", {"Location": "https://x/"}, io.BytesIO(b"")
        )

    monkeypatch.setattr(http_mod.opener, "open", fake_open)
    resp = http_mod.urllib_transport("https://hackathon.savetax.now/v1/me", {"Accept": "application/json"}, 5.0)
    assert resp.status == 302
    assert opened["n"] == 1


def test_http_error_with_unreadable_body_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingBody:
        def read(self, *args: object) -> bytes:
            raise OSError("body gone")

        def close(self) -> None:
            self.closed = True

    err = urllib.error.HTTPError("https://h/v1/me", 500, "Err", {}, FailingBody())  # type: ignore[arg-type]

    def fake_open(req: object, timeout: float = 0) -> object:
        raise err

    monkeypatch.setattr(http_mod.opener, "open", fake_open)
    with pytest.raises(DataIntegrityError):
        http_mod.urllib_transport("https://hackathon.savetax.now/v1/me", {}, 5.0)
    assert err.fp.closed


def test_default_transport_maps_normal_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHeaders:
        def items(self) -> list[tuple[str, str]]:
            return [("X-RateLimit-Remaining", "5")]

    class FakeResp:
        status = 200
        headers = FakeHeaders()

        def read(self) -> bytes:
            return b"{}"

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(http_mod.opener, "open", lambda req, timeout=0: FakeResp())
    resp = http_mod.urllib_transport("https://hackathon.savetax.now/v1/me", {}, 5.0)
    assert resp.status == 200
    assert resp.headers["x-ratelimit-remaining"] == "5"
    assert resp.body == b"{}"


def test_error_detail_is_redacted() -> None:
    def failing(url: str, headers: object, timeout: float) -> HttpResponse:
        raise OSError("failed for savetax_ab-cd")

    client = HttpClient(SaveTaxConfig(), Secret(TOKEN), transport=failing, sleep=lambda _s: None)  # type: ignore[arg-type]
    with pytest.raises(NetworkError) as exc:
        client.get("https://hackathon.savetax.now/v1/me", authenticated=False)
    assert "savetax_ab-cd" not in str(exc.value)
    assert "<redacted>" in str(exc.value)


def test_network_traceback_does_not_echo_token() -> None:
    def failing(url: str, headers: object, timeout: float) -> HttpResponse:
        raise OSError(f"failed for {TOKEN}")

    client = HttpClient(SaveTaxConfig(), Secret(TOKEN), transport=failing, sleep=lambda _s: None)
    with pytest.raises(NetworkError) as exc:
        client.get("https://api.savetax.now/v1/me", authenticated=True)
    assert TOKEN not in "".join(traceback.format_exception(exc.value))


def test_encoded_token_in_url_is_refused_before_io() -> None:
    client, calls, _ = _client()
    encoded = quote(TOKEN).replace("s", "%73")
    with pytest.raises(CredentialError):
        client.get(f"https://api.savetax.now/v1/me?q={encoded}", authenticated=True)
    assert calls == []


def test_programming_errors_are_not_retried_or_misclassified() -> None:
    sleeps: list[float] = []

    def broken(url: str, headers: object, timeout: float) -> HttpResponse:
        raise ValueError("transport programming defect")

    client = HttpClient(SaveTaxConfig(), transport=broken, sleep=sleeps.append)
    with pytest.raises(ValueError, match="programming defect"):
        client.get("https://api.savetax.now/v1/me", authenticated=False)
    assert sleeps == []
