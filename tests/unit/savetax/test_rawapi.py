"""Invariant guards for the read-only raw data client."""

import json
from datetime import date
from decimal import Decimal

import pytest

from src.savetax.config import SaveTaxConfig, Secret
from src.savetax.errors import ApiError, ConfigError, DataIntegrityError
from src.savetax.http import HttpClient, HttpResponse
from src.savetax.rawapi import RateLimit, RawApiClient

TOKEN = "savetax_test-token_1"


def _resp(status: int, payload: object, headers: dict | None = None) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(status=status, headers=headers or {}, body=body, elapsed_seconds=0.01)


def _data(**over: object) -> dict:
    base: dict = {
        "id": "002370053_1130000008",
        "itemId": "24384969",
        "unit": "㎡",
        "model": "MS-A02-01",
        "price": "218000",
        "priceKind": "unknown",
        "manufacturer": "덕인금속주식회사",
        "category": "금속제패널",
        "deliveryTerms": "현장설치도",
        "supplyRegion": "전지역(제주,도서지역 제외)",
        "vendor": {"id": "CN0100000442008", "name": "덕인금속주식회사"},
        "contract": {"no": "002370053_1", "type": "제3자단가계약", "beginYmd": "20230307", "endYmd": "20270115"},
    }
    base.update(over)
    return base


def _ok(data: dict | None = None, release: str = "r20261011T012137", headers: dict | None = None) -> HttpResponse:
    return _resp(200, {"data": data or _data(), "meta": {"release": release}}, headers)


def _client(script: list[HttpResponse]) -> tuple[RawApiClient, list, list[float]]:
    calls: list = []
    sleeps: list[float] = []
    queue = list(script)

    def fake_transport(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append((url, dict(headers)))  # type: ignore[arg-type]
        assert queue, "unexpected extra request"
        return queue.pop(0)

    http = HttpClient(SaveTaxConfig(), Secret(TOKEN), transport=fake_transport, sleep=sleeps.append)  # type: ignore[arg-type]
    return RawApiClient(http, SaveTaxConfig(), sleep=sleeps.append), calls, sleeps


def test_listing_maps_delivery_conditions() -> None:
    client, calls, _ = _client([_ok()])
    detail = client.get_listing("002370053_1130000008")
    assert detail is not None
    assert detail.price_krw == Decimal("218000")
    assert detail.delivery_terms == "현장설치도"
    assert detail.contract_begin == date(2023, 3, 7)
    assert "Authorization" in calls[0][1]


def test_empty_optional_text_becomes_none() -> None:
    client, _, _ = _client([_ok(_data(deliveryTerms=""))])
    detail = client.get_listing("002370053_1130000008")
    assert detail is not None
    assert detail.delivery_terms is None


def test_unparseable_contract_date_becomes_none_without_clamping() -> None:
    client, _, _ = _client([_ok(_data(contract={"no": "002370053_1", "endYmd": "70270231"}))])
    detail = client.get_listing("002370053_1130000008")
    assert detail is not None
    assert detail.contract_end is None
    client2, _, _ = _client([_ok(_data(contract={"no": "002370053_1", "endYmd": "70270115"}))])
    detail2 = client2.get_listing("002370053_1130000008")
    assert detail2 is not None
    assert detail2.contract_end == date(7027, 1, 15)


def test_not_served_returns_none() -> None:
    client, _, _ = _client([_resp(404, {"error": {"code": "not_found", "message": "없음"}})])
    assert client.get_listing("DOES_NOT_EXIST_0") is None


def test_nested_error_becomes_api_error() -> None:
    client, _, _ = _client([_resp(403, {"error": {"code": "forbidden", "message": "no"}})])
    with pytest.raises(ApiError) as exc:
        client.get_listing("002370053_1130000008")
    assert exc.value.code == "forbidden"
    assert exc.value.retryable is False


def test_missing_required_field_fails_closed() -> None:
    data = _data()
    del data["itemId"]
    client, _, _ = _client([_ok(data)])
    with pytest.raises(DataIntegrityError):
        client.get_listing("002370053_1130000008")


def test_rate_limit_headers_are_exposed() -> None:
    headers = {"x-ratelimit-limit": "3600", "x-ratelimit-remaining": "3596", "x-ratelimit-reset": "649"}
    client, _, _ = _client([_ok(headers=headers)])
    client.get_listing("002370053_1130000008")
    assert client.last_rate_limit == RateLimit(3600, 3596, 649)
    client2, _, _ = _client([_ok()])
    client2.get_listing("002370053_1130000008")
    assert client2.last_rate_limit is None


def test_throttle_waits_bounded_time_then_retries_once() -> None:
    headers = {"x-ratelimit-reset": "649"}
    client, calls, sleeps = _client([_resp(429, {"error": {"code": "throttled", "message": "m"}}, headers), _ok()])
    detail = client.get_listing("002370053_1130000008")
    assert detail is not None
    assert sleeps == [60.0]
    assert len(calls) == 2


def test_throttle_honours_retry_after() -> None:
    headers = {"retry-after": "7"}
    client, _, sleeps = _client([_resp(429, {"error": {"code": "throttled", "message": "m"}}, headers), _ok()])
    client.get_listing("002370053_1130000008")
    assert sleeps == [7.0]


def test_second_throttle_raises() -> None:
    headers = {"x-ratelimit-reset": "5"}
    err = _resp(429, {"error": {"code": "throttled", "message": "m"}}, headers)
    client, calls, _ = _client([err, _resp(429, {"error": {"code": "throttled", "message": "m"}}, headers)])
    with pytest.raises(ApiError) as exc:
        client.get_listing("002370053_1130000008")
    assert exc.value.retryable is True
    assert len(calls) == 2


def test_unsafe_listing_id_is_refused_before_io() -> None:
    client, calls, _ = _client([])
    with pytest.raises(ConfigError):
        client.get_listing("../me")
    assert calls == []


def test_optional_text_edge_cases() -> None:
    client, _, _ = _client([_ok(_data(unit=None, model="  "))])
    detail = client.get_listing("002370053_1130000008")
    assert detail is not None
    assert detail.unit is None
    assert detail.model is None
    bad = _data()
    bad["unit"] = 5  # type: ignore[assignment]
    client2, _, _ = _client([_ok(bad)])
    with pytest.raises(DataIntegrityError):
        client2.get_listing("002370053_1130000008")


def test_price_edge_cases_fail_closed() -> None:
    for bad_price in (True, "abc", "NaN", "Infinity", None):
        data = _data(price=bad_price)  # type: ignore[arg-type]
        client, _, _ = _client([_ok(data)])
        with pytest.raises(DataIntegrityError):
            client.get_listing("002370053_1130000008")


def test_ymd_edge_cases_become_none() -> None:
    for bad in (None, "", "2023", "abcdefgh", "20231301"):
        data = _data(contract={"no": "002370053_1", "beginYmd": bad})
        client, _, _ = _client([_ok(data)])
        detail = client.get_listing("002370053_1130000008")
        assert detail is not None
        assert detail.contract_begin is None


def test_malformed_listing_bodies_fail_closed() -> None:
    cases = [
        b"nope",
        b"[1,2]",
        {"meta": {}},
        {"data": None, "meta": {}},
        {"data": _data(), "meta": {"release": 5}},
        {"data": {**_data(), "priceKind": ""}, "meta": {}},
        {"data": {**_data(), "contract": None}, "meta": {}},
        {"data": {**_data(), "vendor": "x"}, "meta": {}},
        {"data": {**_data(), "vendor": {"name": 5}}, "meta": {}},
        {"data": {k: v for k, v in _data().items() if k != "price"}, "meta": {}},
        {"data": {**_data(), "contract": {"type": "t"}}, "meta": {}},
    ]
    for payload in cases:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        client, _, _ = _client([HttpResponse(status=200, headers={}, body=body, elapsed_seconds=0.01)])
        with pytest.raises(DataIntegrityError):
            client.get_listing("002370053_1130000008")


def test_error_mapping_edges() -> None:
    client, _, _ = _client([HttpResponse(status=500, headers={}, body=b"<html>", elapsed_seconds=0.01)])
    with pytest.raises(ApiError) as exc:
        client.get_listing("002370053_1130000008")
    assert exc.value.retryable is True
    client2, _, _ = _client([_resp(404, {"error": {"code": "other", "message": "m"}})])
    with pytest.raises(ApiError):
        client2.get_listing("002370053_1130000008")
    client3, _, _ = _client([_resp(400, {"error": {"code": "", "message": 5}})])
    with pytest.raises(ApiError) as exc3:
        client3.get_listing("002370053_1130000008")
    assert exc3.value.code == "NON_JSON_ERROR"
    assert exc3.value.retryable is False


def test_throttle_without_headers_uses_cooldown() -> None:
    client, _, sleeps = _client(
        [_resp(429, {"error": {"code": "t", "message": "m"}}, {}), _ok()]
    )
    client.get_listing("002370053_1130000008")
    assert sleeps == [15.0]


def test_bad_rate_headers_yield_none_limit() -> None:
    headers = {"x-ratelimit-limit": "x", "x-ratelimit-remaining": "y", "x-ratelimit-reset": "NaN"}
    client, _, _ = _client([_ok(headers=headers)])
    client.get_listing("002370053_1130000008")
    assert client.last_rate_limit is None


def test_retry_then_not_found_returns_none() -> None:
    client, calls, _ = _client(
        [
            _resp(429, {"error": {"code": "t", "message": "m"}}, {}),
            _resp(404, {"error": {"code": "not_found", "message": "m"}}),
        ]
    )
    assert client.get_listing("002370053_1130000008") is None
    assert len(calls) == 2


def test_404_with_non_json_body_raises() -> None:
    for body in (b"<html>", json.dumps({"error": "x"}).encode(), json.dumps({"x": 1}).encode()):
        client, _, _ = _client([HttpResponse(status=404, headers={}, body=body, elapsed_seconds=0.01)])
        with pytest.raises(ApiError):
            client.get_listing("002370053_1130000008")


def test_retry_after_nan_falls_back() -> None:
    for bad in ("nan", "inf", "-inf", "xyz"):
        client, _, sleeps = _client(
            [_resp(429, {"error": {"code": "t", "message": "m"}}, {"retry-after": bad}), _ok()]
        )
        client.get_listing("002370053_1130000008")
        assert sleeps == [15.0]


def test_newline_listing_id_is_refused_before_io() -> None:
    client, calls, _ = _client([])
    with pytest.raises(ConfigError):
        client.get_listing("a\n")
    assert calls == []


@pytest.mark.parametrize("price", [218000, 218000.1])
def test_raw_price_requires_exact_string_representation(price: object) -> None:
    client, _, _ = _client([_ok(_data(price=price))])
    with pytest.raises(DataIntegrityError, match="price"):
        client.get_listing("002370053_1130000008")


def test_fallback_wait_is_bounded_with_custom_config() -> None:
    config = SaveTaxConfig(retry_cooldown_seconds=90.0, rate_limit_max_wait_seconds=10.0)
    responses = [_resp(429, {"error": {"code": "throttled"}}), _ok()]
    sleeps: list[float] = []
    http = HttpClient(config, Secret(TOKEN), transport=lambda *_args: responses.pop(0))
    client = RawApiClient(http, config, sleep=sleeps.append)
    assert client.get_listing("002370053_1130000008") is not None
    assert sleeps == [10.0]


def test_last_rate_limit_clears_after_headerless_response() -> None:
    headers = {"x-ratelimit-limit": "3600", "x-ratelimit-remaining": "3596", "x-ratelimit-reset": "649"}
    client, _, _ = _client([_ok(headers=headers), _ok()])
    client.get_listing("002370053_1130000008")
    assert client.last_rate_limit == RateLimit(3600, 3596, 649)
    client.get_listing("002370053_1130000008")
    assert client.last_rate_limit is None


def test_huge_reset_seconds_remain_bounded() -> None:
    headers = {"x-ratelimit-reset": "9" * 400}
    client, _, sleeps = _client([_resp(429, {"error": {"code": "throttled"}}, headers), _ok()])
    assert client.get_listing("002370053_1130000008") is not None
    assert sleeps == [60.0]


def test_malformed_release_envelope_fails_closed() -> None:
    client, _, _ = _client([_resp(200, {"data": _data(), "meta": "broken"})])
    with pytest.raises(DataIntegrityError, match="meta"):
        client.get_listing("002370053_1130000008")
