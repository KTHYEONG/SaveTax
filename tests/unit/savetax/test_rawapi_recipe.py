"""Invariant guards for labelled item attributes."""

import json

import pytest

from src.savetax.config import SaveTaxConfig, Secret
from src.savetax.errors import ConfigError, DataIntegrityError
from src.savetax.http import HttpClient, HttpResponse
from src.savetax.rawapi import RawApiClient

_TOKEN = "savetax_test-token_1"


def _resp(status: int, payload: object) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(status=status, headers={}, body=body, elapsed_seconds=0.01)


def _client(script: list[HttpResponse]) -> RawApiClient:
    queue = list(script)

    def fake_transport(url: str, headers: object, timeout: float) -> HttpResponse:
        assert queue, "unexpected extra request"
        return queue.pop(0)

    http = HttpClient(SaveTaxConfig(), Secret(_TOKEN), transport=fake_transport, sleep=lambda s: None)  # type: ignore[arg-type]
    return RawApiClient(http, SaveTaxConfig(), sleep=lambda s: None)


def _ok(attrs: list[dict] | None = None) -> HttpResponse:
    data: dict = {"id": "123", "attributes": attrs if attrs is not None else [
        {"name": " 중앙처리장치(CPU)규격 ", "value": " Intel Core "},
        {"name": "메모리(RAM)용량", "value": "16"},
    ]}
    return _resp(200, {"data": data})


def test_attributes_returned_by_name() -> None:
    client = _client([_ok()])
    assert client.get_item_attributes("123") == {"중앙처리장치(CPU)규격": "Intel Core", "메모리(RAM)용량": "16"}


def test_unserved_item_returns_none() -> None:
    client = _client([_resp(404, {"error": {"code": "not_found", "message": "x"}})])
    assert client.get_item_attributes("123") is None


def test_identity_mismatch_fails_closed() -> None:
    client = _client([_resp(200, {"data": {"id": "999", "attributes": []}})])
    with pytest.raises(DataIntegrityError):
        client.get_item_attributes("123")


def test_conflicting_duplicate_fails_closed() -> None:
    client = _client([_ok([{"name": "A", "value": "1"}, {"name": "A", "value": "2"}])])
    with pytest.raises(DataIntegrityError):
        client.get_item_attributes("123")


def test_retry_after_rate_limit() -> None:
    client = _client([_resp(429, {"error": {"code": "t", "message": "m"}}), _ok()])
    assert client.get_item_attributes("123") is not None


def test_bad_item_id_rejected() -> None:
    with pytest.raises(ConfigError):
        _client([]).get_item_attributes("bad id!")


def test_non_429_error_raises() -> None:
    from src.savetax.errors import ApiError

    client = _client([_resp(403, {"error": {"code": "forbidden", "message": "no"}})])
    with pytest.raises(ApiError, match="forbidden"):
        client.get_item_attributes("123")


def test_retry_then_not_found_and_error() -> None:
    from src.savetax.errors import ApiError

    client = _client([_resp(429, {"error": {"code": "t", "message": "m"}}), _resp(404, {"error": {"code": "not_found", "message": "x"}})])
    assert client.get_item_attributes("123") is None
    client2 = _client([_resp(429, {"error": {"code": "t", "message": "m"}}), _resp(500, {"error": {"code": "boom", "message": "m"}})])
    with pytest.raises(ApiError, match="boom"):
        client2.get_item_attributes("123")


@pytest.mark.parametrize(
    "payload",
    [b"nope", [1, 2], {"data": "x"}, {"data": {"id": "123", "attributes": {}}}, {"data": {"id": "123", "attributes": ["x"]}}],
)
def test_malformed_bodies_fail_closed(payload: object) -> None:
    client = _client([_resp(200, payload)])
    with pytest.raises(DataIntegrityError):
        client.get_item_attributes("123")


def test_skipped_and_same_duplicate_ok() -> None:
    client = _client([_ok([{"name": "", "value": "1"}, {"name": "A", "value": ""}, {"name": 1, "value": "x"}, {"name": "A", "value": "v"}, {"name": "A", "value": "v"}, {"name": "B", "value": "w"}])])
    assert client.get_item_attributes("123") == {"A": "v", "B": "w"}
    client2 = _client([_resp(200, {"data": {"id": "123"}})])
    assert client2.get_item_attributes("123") == {}
