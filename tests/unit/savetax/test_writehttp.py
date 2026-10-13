"""Invariant guards for the POST-only submission transport."""

import io
import json
import urllib.error

import pytest

import src.savetax.writehttp as writehttp_mod
from src.savetax.config import SaveTaxConfig, Secret
from src.savetax.errors import ConfigError, DataIntegrityError, DomainValidationError, NetworkError
from src.savetax.http import HttpResponse
from src.savetax.writehttp import WriteHttpClient

TOKEN = "savetax_test-token_1"


def _client(**kw: object) -> tuple[WriteHttpClient, list]:
    calls: list = []
    behaviour = kw.get("behaviour", "ok")
    status = int(kw.get("status", 200))  # type: ignore[arg-type]

    def fake_transport(method: str, url: str, headers: object, body: object, timeout: float) -> HttpResponse:
        calls.append((method, url, dict(headers), body, timeout))  # type: ignore[arg-type]
        if behaviour == "fail_token":
            raise OSError(f"down {TOKEN}")
        return HttpResponse(status=status, headers={}, body=b"{}", elapsed_seconds=0.01)

    client = WriteHttpClient(
        kw.get("config", SaveTaxConfig()),  # type: ignore[arg-type]
        Secret(TOKEN),
        transport=fake_transport,  # type: ignore[arg-type]
        sleep=lambda _s: None,
    )
    return client, calls


@pytest.mark.parametrize(
    "path",
    [
        "/reports",
        "/uploads",
        "/submissions/abc/counterevidence",
        "/../me",
        "/submissions/",
        "/submissions/preview/extra",
        "/submissions/abc/revisions/extra",
        "/submissions//withdraw",
        "/SUBMISSIONS",
        "/submissions/preview?x=1",
        "/submissions/abc//revisions",
        "/submissions/abc//withdraw",
        "/submissions/abc\n/withdraw",
        "/submissions/" + "a" * 65 + "/revisions",
    ],
)
def test_only_submission_paths_are_reachable(path: str) -> None:
    client, calls = _client()
    with pytest.raises(ConfigError):
        client.post_json(path, {}, idempotency_key="K1")
    assert calls == []


def test_credential_destination_is_enforced() -> None:
    evil = SaveTaxConfig(competition_base_url="https://evil.example.com/v1")
    client, calls = _client(config=evil)
    with pytest.raises(ConfigError):
        client.post_json("/submissions", {}, idempotency_key="K1")
    assert calls == []


def test_preview_omits_idempotency_header() -> None:
    client, calls = _client()
    client.post_json("/submissions/preview", {"a": 1}, idempotency_key="K1")
    headers = calls[0][2]
    assert "Idempotency-Key" not in headers
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    assert headers["Content-Type"] == "application/json"


def test_submit_carries_key() -> None:
    client, calls = _client()
    client.post_json("/submissions", {"a": 1}, idempotency_key="K1")
    assert calls[0][2]["Idempotency-Key"] == "K1"


def test_if_match_sent_only_when_given() -> None:
    client, calls = _client()
    client.post_json("/submissions/sid-1/revisions", {"a": 1}, idempotency_key="K1")
    assert "If-Match" not in calls[0][2]
    client.post_json("/submissions/sid-1/revisions", {"a": 1}, idempotency_key="K1", if_match=3)
    assert calls[1][2]["If-Match"] == "3"


def test_invalid_key_is_rejected() -> None:
    client, calls = _client()
    with pytest.raises(ConfigError):
        client.post_json("/submissions", {}, idempotency_key="has space")
    with pytest.raises(ConfigError):
        client.post_json("/submissions", {}, idempotency_key="x" * 129)
    assert calls == []


def test_body_is_canonical_json() -> None:
    client, calls = _client()
    client.post_json("/submissions", {"b": 2, "a": 1}, idempotency_key="K1")
    first = calls[0][3]
    client2, calls2 = _client()
    client2.post_json("/submissions", {"a": 1, "b": 2}, idempotency_key="K1")
    assert first == calls2[0][3]
    assert first == json.dumps({"a": 1, "b": 2}, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def test_oversize_body_is_rejected() -> None:
    client, calls = _client()
    big = "x" * 1_000_001
    with pytest.raises(DomainValidationError):
        client.post_json("/submissions", {"blob": big}, idempotency_key="K1")
    assert calls == []


def test_redirect_is_not_followed() -> None:
    client, calls = _client(status=302)
    resp = client.post_json("/submissions", {}, idempotency_key="K1")
    assert resp.status == 302
    assert len(calls) == 1


def test_token_never_leaks() -> None:
    client, calls = _client(behaviour="fail_token")
    with pytest.raises(NetworkError) as exc:
        client.post_json("/submissions", {}, idempotency_key="K1")
    assert TOKEN not in str(exc.value)
    assert "<redacted>" in str(exc.value)


def test_error_statuses_are_returned() -> None:
    for status in (409, 503):
        client, _ = _client(status=status)
        resp = client.post_json("/submissions", {}, idempotency_key="K1")
        assert resp.status == status


def test_invalid_if_match_rejected() -> None:
    client, calls = _client()
    with pytest.raises(ConfigError):
        client.post_json("/submissions/s1/revisions", {}, idempotency_key="K1", if_match=True)  # type: ignore[arg-type]
    assert calls == []


def test_unserializable_body_rejected() -> None:
    client, calls = _client()
    with pytest.raises(DataIntegrityError):
        client.post_json("/submissions", {"v": object()}, idempotency_key="K1")  # type: ignore[dict-item]
    assert calls == []


def test_withdraw_and_revise_paths_accepted() -> None:
    client, calls = _client()
    client.post_json("/submissions/abc_123-XY/withdraw", None, idempotency_key="K1")
    assert calls[0][3] is None
    client.post_json("/submissions/abc_123-XY/revisions", {"a": 1}, idempotency_key="K1", if_match=1)
    assert calls[1][2]["If-Match"] == "1"


def test_urllib_transport_posts_without_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    class FakeHeaders:
        def items(self) -> list:
            return [("Content-Type", "application/json")]

    class FakeResp:
        status = 201
        headers = FakeHeaders()

        def read(self) -> bytes:
            return b"{}"

        def close(self) -> None:
            seen["closed"] = True

    def fake_open(req: object, timeout: float = 0) -> object:
        seen["method"] = req.get_method()  # type: ignore[attr-defined]
        return FakeResp()

    monkeypatch.setattr(writehttp_mod._opener, "open", fake_open)
    resp = writehttp_mod.urllib_write_transport("POST", "https://h/v1/x", {"A": "b"}, b"{}", 5.0)
    assert resp.status == 201
    assert seen["method"] == "POST"


def test_urllib_transport_http_error_is_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_open(req: object, timeout: float = 0) -> object:
        raise urllib.error.HTTPError("https://h/x", 302, "Found", {"Location": "https://y/"}, io.BytesIO(b"z"))

    monkeypatch.setattr(writehttp_mod._opener, "open", fake_open)
    resp = writehttp_mod.urllib_write_transport("POST", "https://h/x", {}, None, 5.0)
    assert resp.status == 302
    assert resp.body == b"z"


def test_urllib_transport_failure_is_staged(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_open(req: object, timeout: float = 0) -> object:
        raise OSError("boom savetax_hidden-token_x")

    monkeypatch.setattr(writehttp_mod._opener, "open", fake_open)
    with pytest.raises(NetworkError) as exc:
        writehttp_mod.urllib_write_transport("POST", "https://h/x", {}, None, 5.0)
    assert "savetax_hidden-token_x" not in str(exc.value)


def test_urllib_transport_unreadable_body_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class Bad:
        def read(self, *a: object) -> bytes:
            raise OSError("gone")

        def close(self) -> None:
            pass

    err = urllib.error.HTTPError("https://h/x", 500, "E", {}, Bad())  # type: ignore[arg-type]

    def fake_open(req: object, timeout: float = 0) -> object:
        raise err

    monkeypatch.setattr(writehttp_mod._opener, "open", fake_open)
    with pytest.raises(DataIntegrityError):
        writehttp_mod.urllib_write_transport("POST", "https://h/x", {}, None, 5.0)


def test_no_redirect_handler_returns_none() -> None:
    handler = writehttp_mod._NoRedirect()
    assert handler.redirect_request(object(), object(), 302, "Found", object(), "https://x/") is None


@pytest.mark.parametrize("base", ["https://api.savetax.now/v1", "https://hackathon.savetax.now/v1?next="])
def test_write_never_targets_raw_api_or_query_base(base: str) -> None:
    client, calls = _client(config=SaveTaxConfig(competition_base_url=base))
    with pytest.raises(ConfigError):
        client.post_json("/submissions", {}, idempotency_key="K")
    assert calls == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "\ud800"])
def test_invalid_json_values_fail_before_transmission(value: object) -> None:
    client, calls = _client()
    with pytest.raises(DataIntegrityError):
        client.post_json("/submissions", {"value": value}, idempotency_key="K")
    assert calls == []


def test_exact_size_limit_uses_utf8_bytes() -> None:
    client, calls = _client()
    client.post_json("/submissions", {"x": "x" * (1_000_000 - 8)}, idempotency_key="K")
    assert len(calls[0][3]) == 1_000_000
    with pytest.raises(DomainValidationError):
        client.post_json("/submissions", {"x": "한" * 333_331}, idempotency_key="K")
    assert len(calls) == 1


@pytest.mark.parametrize("location", ["body", "key", "path"])
def test_credential_is_confined_to_header(location: str) -> None:
    client, calls = _client()
    path = f"/submissions/{TOKEN}/withdraw" if location == "path" else "/submissions"
    body = {"quote": TOKEN} if location == "body" else {}
    key = TOKEN if location == "key" else "K"
    with pytest.raises(ConfigError):
        client.post_json(path, body, idempotency_key=key)
    assert calls == []


def test_network_error_redacts_exact_secret_and_traceback() -> None:
    import traceback
    from src.savetax.errors import NetworkStage

    token = "private.token-with-punctuation"

    def failing(*args: object) -> HttpResponse:
        raise NetworkError(NetworkStage.DNS_RESOLUTION, token, token)

    client = WriteHttpClient(SaveTaxConfig(), Secret(token), transport=failing)
    with pytest.raises(NetworkError) as caught:
        client.post_json("/submissions", {}, idempotency_key="K")
    assert token not in "".join(traceback.format_exception(caught.value))
    assert token not in str(vars(caught.value))
    assert caught.value.stage is NetworkStage.DNS_RESOLUTION


def test_urllib_rejects_non_post_before_io() -> None:
    with pytest.raises(ConfigError):
        writehttp_mod.urllib_write_transport("GET", "https://h/x", {}, None, 5)


def test_urllib_protocol_disconnect_is_a_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from http.client import BadStatusLine

    def disconnected(*args: object, **kwargs: object) -> object:
        raise BadStatusLine("connection truncated")

    monkeypatch.setattr(writehttp_mod._opener, "open", disconnected)
    with pytest.raises(NetworkError):
        writehttp_mod.urllib_write_transport("POST", "https://h/x", {}, None, 5)


def test_error_response_cannot_echo_exact_credential() -> None:
    token = "private.credential"

    def echo(*args: object) -> HttpResponse:
        return HttpResponse(409, {"x-error": token}, json.dumps({"message": token}).encode(), 0)

    client = WriteHttpClient(SaveTaxConfig(), Secret(token), transport=echo)
    response = client.post_json("/submissions", {}, idempotency_key="K")
    assert token not in repr(response)
    assert response.headers["x-error"] == "<redacted>"
