"""Invariant guards for the read-only competition client."""

import json
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from src.savetax.competition import CompetitionClient, ItemState
from src.savetax.config import SaveTaxConfig, Secret
from src.savetax.errors import ApiError, ConfigError, DataIntegrityError
from src.savetax.http import HttpClient, HttpResponse

TOKEN = "savetax_test-token_1"


def _resp(status: int, payload: object, headers: dict | None = None) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(status=status, headers=headers or {}, body=body, elapsed_seconds=0.01)


def _client(
    script: list[HttpResponse],
    config: SaveTaxConfig | None = None,
) -> tuple[CompetitionClient, list, list[float]]:
    calls: list = []
    sleeps: list[float] = []
    queue = list(script)

    def fake_transport(url: str, headers: object, timeout: float) -> HttpResponse:
        calls.append((url, dict(headers)))  # type: ignore[arg-type]
        assert queue, "unexpected extra request"
        return queue.pop(0)

    http = HttpClient(
        config or SaveTaxConfig(),
        Secret(TOKEN),
        transport=fake_transport,  # type: ignore[arg-type]
        sleep=sleeps.append,
    )
    return CompetitionClient(http, config or SaveTaxConfig(), sleep=sleeps.append), calls, sleeps


def _event_body() -> dict:
    return {
        "dataset_release_id": "release-live-20261012",
        "ruleset_version": "rules-v1",
        "phase": "RUNNING",
        "opens_at": "2026-10-11T22:00:00.000Z",
        "closes_at": "2026-10-25T15:00:00.000Z",
        "evidence_format": "text",
        "item_catalog": "live",
    }


def _fact(**over: object) -> dict:
    base: dict = {
        "fact_id": "dl:1:c:R1:1",
        "listing_id": "002370053_1130000008",
        "contract_no": "002370053_1",
        "request_no": "R1",
        "line_no": "1",
        "request_date": "2025-10-14",
        "unit": "㎡",
        "tax_basis": "vat_included",
        "basis_id": "basis:㎡:vat_included",
        "comparison_basis_id": "basis:㎡:vat_included",
        "period_kind": "none",
        "unit_price_krw": "189000",
        "quantity": "4680",
        "comparable_amount_krw": "884520000",
        "scorable": True,
        "exclusion_reasons": [],
    }
    base.update(over)
    return base


def _item_body(**over: object) -> dict:
    base: dict = {
        "item_id": "24384969",
        "snapshot": {
            "name": "n",
            "model": "m",
            "manufacturer": "man",
            "category": "cat",
            "source_release": "r20261011T012137",
            "facts_total": 1,
            "facts_truncated": False,
        },
        "availability": {"state": "available"},
        "facts": [_fact()],
    }
    base.update(over)
    return base


def test_event_is_fetched_without_credentials() -> None:
    client, calls, _ = _client([_resp(200, _event_body())])
    event = client.get_event()
    assert "Authorization" not in calls[0][1]
    assert event.closes_at == datetime(
        2026, 10, 25, 15, 0, tzinfo=timezone.utc  # noqa: UP017
    )
    assert event.closes_at.tzinfo is not None


def test_unlimited_limits_stay_none() -> None:
    body = {
        "handle": "tester",
        "registered": True,
        "claims_used": 3,
        "claims_today": 1,
        "remaining": {"claims_total": 100, "claims_per_day": None},
    }
    client, _, _ = _client([_resp(200, body)])
    me = client.get_me()
    assert me.remaining_claims_total == 100
    assert me.remaining_claims_per_day is None


def test_lookup_chunks_at_100_and_preserves_order() -> None:
    ids = [f"id_{i:04d}" for i in range(230)]
    script = []
    for start in range(0, 230, 100):
        chunk = ids[start : start + 100]
        script.append(
            _resp(
                200,
                {"items": [{"input": i, "kind": "item", "item_id": i, "state": "available"} for i in reversed(chunk)]},
            )
        )
    client, calls, _ = _client(script)
    results = client.lookup_items(ids)
    assert [r.input for r in results] == ids
    assert len(calls) == 3
    assert "ids=" in calls[0][0]
    assert len(calls[2][0].split("ids=")[1].split(",")) == 30


def test_lookup_rejects_unsafe_ids_before_io() -> None:
    client, calls, _ = _client([])
    with pytest.raises(ConfigError):
        client.lookup_items(["a&b"])
    with pytest.raises(ConfigError):
        client.lookup_items(["has space"])
    assert calls == []


def test_lookup_empty_returns_without_request() -> None:
    client, calls, _ = _client([])
    assert client.lookup_items([]) == []
    assert calls == []


def test_lookup_with_missing_id_fails_closed() -> None:
    script = [_resp(200, {"items": [{"input": "a", "kind": "item", "item_id": "a", "state": "available"}]})]
    client, _, _ = _client(script)
    with pytest.raises(DataIntegrityError):
        client.lookup_items(["a", "b"])


def test_lookup_parses_on_hold_state() -> None:
    script = [_resp(200, {"items": [{"input": "a", "kind": "item", "item_id": "a", "state": "on_hold",
                                     "hold_until": "2026-10-17T00:00:00Z"}]})]
    client, _, _ = _client(script)
    assert [r.state for r in client.lookup_items(["a"])] == [ItemState.ON_HOLD]


def test_unknown_state_fails_closed() -> None:
    script = [_resp(200, {"items": [{"input": "a", "kind": "item", "item_id": "a", "state": "frozen"}]})]
    client, _, _ = _client(script)
    with pytest.raises(DataIntegrityError):
        client.lookup_items(["a"])
    client2, _, _ = _client([_resp(200, _item_body(availability={"state": "frozen"}))])
    with pytest.raises(DataIntegrityError):
        client2.get_item("24384969")


def test_facts_parse_exact_decimals() -> None:
    client, _, _ = _client([_resp(200, _item_body())])
    detail = client.get_item("24384969")
    (fact,) = detail.facts
    assert fact.unit_price_krw == Decimal("189000")
    assert fact.quantity == Decimal("4680")
    assert fact.comparable_amount_krw == Decimal("884520000")
    assert fact.basis_id == "basis:㎡:vat_included"


def test_fractional_quantities_keep_precision() -> None:
    body = _item_body(facts=[_fact(quantity="2.500001")])
    client, _, _ = _client([_resp(200, body)])
    (fact,) = client.get_item("24384969").facts
    assert fact.quantity == Decimal("2.500001")


def test_unscorable_fact_keeps_its_reasons() -> None:
    body = _item_body(facts=[_fact(scorable=False, exclusion_reasons=["unit_missing"], unit="미상")])
    client, _, _ = _client([_resp(200, body)])
    (fact,) = client.get_item("24384969").facts
    assert fact.scorable is False
    assert fact.exclusion_reasons == ("unit_missing",)
    assert fact.unit == "미상"


def test_withheld_snapshot_yields_empty_detail() -> None:
    body = {"item_id": "25534748", "snapshot": None, "availability": {"state": "reform_found"}, "facts": []}
    client, _, _ = _client([_resp(200, body)])
    detail = client.get_item("25534748")
    assert detail.state is ItemState.REFORM_FOUND
    assert detail.name is None
    assert detail.facts == ()


def test_missing_required_field_fails_closed() -> None:
    bad = _fact()
    del bad["comparable_amount_krw"]
    client, _, _ = _client([_resp(200, _item_body(facts=[bad]))])
    with pytest.raises(DataIntegrityError, match="comparable_amount_krw"):
        client.get_item("24384969")


def test_non_numeric_amount_fails_closed() -> None:
    for bad_qty in ("abc", "NaN", "Infinity"):
        body = _item_body(facts=[_fact(quantity=bad_qty)])
        client, _, _ = _client([_resp(200, body)])
        with pytest.raises(DataIntegrityError):
            client.get_item("24384969")


def test_flat_error_body_becomes_api_error() -> None:
    err = {"code": "PRODUCT_NOT_FOUND", "message": "없음", "request_id": "r1", "retryable": False}
    client, _, _ = _client([_resp(404, err)])
    with pytest.raises(ApiError) as exc:
        client.get_item("00000000")
    assert exc.value.status == 404
    assert exc.value.code == "PRODUCT_NOT_FOUND"
    assert exc.value.retryable is False


def test_rate_limit_waits_once_then_succeeds() -> None:
    err = {"code": "RATE_LIMITED", "message": "slow", "request_id": "r", "retryable": True}
    client, calls, sleeps = _client([_resp(429, err), _resp(200, _item_body())])
    detail = client.get_item("24384969")
    assert detail.item_id == "24384969"
    assert sleeps == [60.0]
    assert len(calls) == 2


def test_edge_rate_limit_without_json_body_is_retried() -> None:
    client, calls, sleeps = _client([_resp(429, b"<html>busy</html>"), _resp(200, _item_body())])
    assert client.get_item("24384969").item_id == "24384969"
    assert sleeps == [60.0]
    assert len(calls) == 2


def test_rate_limit_twice_raises() -> None:
    err = {"code": "RATE_LIMITED", "message": "slow", "request_id": "r", "retryable": True}
    client, calls, sleeps = _client([_resp(429, err), _resp(429, err)])
    with pytest.raises(ApiError):
        client.get_item("24384969")
    assert len(calls) == 2
    assert sleeps == [60.0]


def test_catalog_outage_retries_once() -> None:
    err = {"code": "CATALOG_UNAVAILABLE", "message": "down", "request_id": "r", "retryable": True}
    client, calls, sleeps = _client([_resp(503, err), _resp(200, _item_body())])
    client.get_item("24384969")
    assert sleeps == [15.0]
    assert len(calls) == 2


def test_auth_and_claim_errors_are_never_retried() -> None:
    for status, code in ((401, "AUTH_REQUIRED"), (403, "NOT_REGISTERED"), (409, "PRODUCT_ALREADY_CLAIMED")):
        err = {"code": code, "message": "m", "request_id": "r", "retryable": False}
        client, calls, sleeps = _client([_resp(status, err)])
        with pytest.raises(ApiError):
            client.get_item("24384969")
        assert len(calls) == 1
        assert sleeps == []


def test_non_json_error_body_is_contained() -> None:
    client, _, _ = _client([HttpResponse(status=502, headers={}, body=b"<html>bad</html>", elapsed_seconds=0.01)])
    with pytest.raises(ApiError) as exc:
        client.get_item("24384969")
    assert exc.value.code == "NON_JSON_ERROR"


def test_request_date_empty_becomes_none() -> None:
    for raw in (None, ""):
        body = _item_body(facts=[_fact(request_date=raw)])
        client, _, _ = _client([_resp(200, body)])
        (fact,) = client.get_item("24384969").facts
        assert fact.request_date is None
    body = _item_body(facts=[_fact(request_date="2025-10-14")])
    client, _, _ = _client([_resp(200, body)])
    (fact,) = client.get_item("24384969").facts
    assert fact.request_date == date(2025, 10, 14)


def test_malformed_success_bodies_fail_closed() -> None:
    bad_bodies: list = [
        b"not json",
        json.dumps(["list"]).encode(),
        json.dumps({"phase": 1}).encode(),
    ]
    for body in bad_bodies:
        client, _, _ = _client([HttpResponse(status=200, headers={}, body=body, elapsed_seconds=0.01)])
        with pytest.raises(DataIntegrityError):
            client.get_event()


def test_event_rejects_bad_moments_and_missing_fields() -> None:
    base = _event_body()
    for field, bad in (("opens_at", ""), ("opens_at", "not-a-date"), ("closes_at", "2026-10-25 15:00:00")):
        body = dict(base)
        body[field] = bad
        client, _, _ = _client([_resp(200, body)])
        with pytest.raises(DataIntegrityError):
            client.get_event()
    body = dict(base)
    del body["phase"]
    client, _, _ = _client([_resp(200, body)])
    with pytest.raises(DataIntegrityError):
        client.get_event()


def test_me_rejects_malformed_shapes() -> None:
    good = {
        "handle": "h",
        "registered": True,
        "claims_used": 0,
        "claims_today": 0,
        "remaining": {"claims_total": None, "claims_per_day": None},
    }
    cases = [
        {**good, "remaining": None},
        {**good, "remaining": {"claims_total": "x", "claims_per_day": None}},
        {**good, "remaining": {"claims_total": True, "claims_per_day": None}},
        {**good, "handle": ""},
        {**good, "registered": "yes"},
        {**good, "claims_used": True},
        {**good, "claims_today": 1.5},
    ]
    for body in cases:
        client, _, _ = _client([_resp(200, body)])
        with pytest.raises(DataIntegrityError):
            client.get_me()


def test_lookup_rejects_malformed_lists() -> None:
    for payload in (
        {"items": None},
        {"items": ["x"]},
        {"items": [{"kind": "item", "item_id": "a", "state": "available"}]},
        {"items": [{"input": "a", "kind": "", "item_id": "a", "state": "available"}]},
        {"items": [{"input": "a", "kind": "item", "item_id": 5, "state": "available"}]},
    ):
        client, _, _ = _client([_resp(200, payload)])
        with pytest.raises(DataIntegrityError):
            client.lookup_items(["a"])


def test_item_rejects_malformed_shapes() -> None:
    good = _item_body()
    cases = [
        {**good, "item_id": ""},
        {**good, "availability": None},
        {**good, "snapshot": "x"},
        {**good, "facts": None},
        {**good, "facts": ["x"]},
        {**good, "snapshot": {"facts_total": True, "facts_truncated": False}},
        {**good, "snapshot": {"facts_total": 1, "facts_truncated": "no"}},
        {**good, "snapshot": {"facts_total": 1, "facts_truncated": False, "name": 5}},
    ]
    for body in cases:
        client, _, _ = _client([_resp(200, body)])
        with pytest.raises(DataIntegrityError):
            client.get_item("24384969")


def test_fact_rejects_bad_shapes() -> None:
    bad_facts = [
        _fact(scorable="yes"),
        _fact(exclusion_reasons="x"),
        _fact(exclusion_reasons=[1]),
        _fact(request_date=5),
        _fact(request_date="not-a-date"),
        _fact(unit_price_krw=True),
        _fact(unit_price_krw="xyz"),
        _fact(unit=""),
        {**_fact(), "basis_id": "", "comparison_basis_id": ""},
    ]
    for bad in bad_facts:
        if isinstance(bad, dict) and "basis_id" in bad and bad["basis_id"] == "":
            bad.pop("basis_id", None)
        client, _, _ = _client([_resp(200, _item_body(facts=[bad]))])
        with pytest.raises(DataIntegrityError):
            client.get_item("24384969")


def test_fact_requires_authoritative_basis_id() -> None:
    fact = _fact()
    del fact["basis_id"]
    client, _, _ = _client([_resp(200, _item_body(facts=[fact]))])
    with pytest.raises(DataIntegrityError, match=r"facts\.basis_id"):
        client.get_item("24384969")


def test_non_retryable_statuses_raise_immediately() -> None:
    for status, code in ((429, "PARTICIPANT_LIMIT"), (503, "OTHER"), (422, "SCHEMA_INVALID")):
        err = {"code": code, "message": "m", "request_id": "r", "retryable": False}
        client, calls, sleeps = _client([_resp(status, err)])
        with pytest.raises(ApiError):
            client.get_item("24384969")
        assert len(calls) == 1
        assert sleeps == []


def test_reform_check_outage_retries() -> None:
    err = {"code": "REFORM_CHECK_UNAVAILABLE", "message": "m", "request_id": "r", "retryable": True}
    client, _, sleeps = _client([_resp(503, err), _resp(200, _item_body())])
    client.get_item("24384969")
    assert sleeps == [15.0]


def test_error_with_empty_code_uses_unknown() -> None:
    client, _, _ = _client([_resp(500, {"message": "m"})])
    with pytest.raises(ApiError) as exc:
        client.get_item("24384969")
    assert exc.value.code == "UNKNOWN_ERROR"


def test_get_item_rejects_unsafe_id() -> None:
    client, calls, _ = _client([])
    with pytest.raises(ConfigError):
        client.get_item("bad id!")
    assert calls == []


def test_snapshot_none_fields_are_allowed() -> None:
    body = _item_body(
        snapshot={
            "name": None,
            "model": None,
            "manufacturer": None,
            "category": None,
            "source_release": None,
            "facts_total": 1,
            "facts_truncated": False,
        }
    )
    client, _, _ = _client([_resp(200, body)])
    detail = client.get_item("24384969")
    assert detail.name is None
    assert detail.source_release is None


@pytest.mark.parametrize("field", ["claims_total", "claims_per_day"])
def test_missing_remaining_budget_is_not_unlimited(field: str) -> None:
    remaining = {"claims_total": None, "claims_per_day": None}
    del remaining[field]
    body = {"handle": "h", "registered": True, "claims_used": 0, "claims_today": 0, "remaining": remaining}
    client, _, _ = _client([_resp(200, body)])
    with pytest.raises(DataIntegrityError, match=f"remaining.{field}"):
        client.get_me()


@pytest.mark.parametrize("field", ["snapshot", "snapshot.name", "facts.request_date"])
def test_missing_nullable_fields_are_rejected(field: str) -> None:
    body = _item_body()
    if field == "snapshot":
        del body["snapshot"]
    elif field == "snapshot.name":
        del body["snapshot"]["name"]
    else:
        del body["facts"][0]["request_date"]
    client, _, _ = _client([_resp(200, body)])
    with pytest.raises(DataIntegrityError, match=field):
        client.get_item("24384969")


@pytest.mark.parametrize("value", [189000, 189000.1])
def test_numeric_json_amounts_do_not_bypass_string_contract(value: object) -> None:
    client, _, _ = _client([_resp(200, _item_body(facts=[_fact(unit_price_krw=value)]))])
    with pytest.raises(DataIntegrityError, match="unit_price_krw"):
        client.get_item("24384969")


def test_newline_ids_are_rejected_before_any_chunk_request() -> None:
    client, calls, _ = _client([])
    with pytest.raises(ConfigError):
        client.lookup_items(["a"] * 100 + ["a\n"])
    assert calls == []


def test_duplicate_lookup_inputs_preserve_histogram(caplog: pytest.LogCaptureFixture) -> None:
    entry = {"input": "a", "kind": "item", "item_id": "a", "state": "available"}
    client, _, _ = _client([_resp(200, {"items": [entry]})])
    with caplog.at_level("INFO"):
        result = client.lookup_items(["a", "a"])
    assert len(result) == 2
    assert "lookup n=2 states=available:2" in caplog.text


def test_mixed_transient_errors_have_separate_retry_budgets() -> None:
    limited = _resp(429, {"code": "RATE_LIMITED", "retryable": True})
    outage = _resp(503, {"code": "CATALOG_UNAVAILABLE", "retryable": True})
    client, calls, sleeps = _client([limited, outage, _resp(200, _item_body())])
    assert client.get_item("24384969").facts
    assert len(calls) == 3
    assert sleeps == [60.0, 15.0]


@pytest.mark.parametrize("value", ["20261011", "2026-W41-7", "2026-02-30"])
def test_request_dates_require_calendar_iso_format(value: str) -> None:
    client, _, _ = _client([_resp(200, _item_body(facts=[_fact(request_date=value)]))])
    with pytest.raises(DataIntegrityError, match="request_date"):
        client.get_item("24384969")


@pytest.mark.parametrize("value", ["2026-02-30T00:00:00.000Z", "2026-10-11T22:00:00+09:00"])
def test_event_timestamps_require_valid_utc(value: str) -> None:
    body = _event_body()
    body["opens_at"] = value
    client, _, _ = _client([_resp(200, body)])
    with pytest.raises(DataIntegrityError, match="opens_at"):
        client.get_event()


def test_present_snapshot_text_wrong_type_is_rejected() -> None:
    body = _item_body()
    body["snapshot"]["name"] = 5
    client, _, _ = _client([_resp(200, body)])
    with pytest.raises(DataIntegrityError, match=r"snapshot\.name"):
        client.get_item("24384969")


def test_conflicting_lookup_duplicates_fail_closed() -> None:
    entry = {"input": "a", "kind": "item", "item_id": "a", "state": "available"}
    client, _, _ = _client([_resp(200, {"items": [entry, {**entry, "state": "claimed"}]})])
    with pytest.raises(DataIntegrityError, match="conflicting duplicate"):
        client.lookup_items(["a"])
