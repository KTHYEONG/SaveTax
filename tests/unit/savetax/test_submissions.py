"""Invariant guards for the idempotent submission workflow."""

import json
import logging

import pytest

from src.savetax.competition import CompetitionClient
from src.savetax.config import SaveTaxConfig, Secret
from src.savetax.errors import ApiError, ConfigError, DataIntegrityError, NetworkError, NetworkStage, SubmissionPending
from src.savetax.http import HttpClient, HttpResponse
from src.savetax.submissions import SubmissionClient
from src.savetax.writehttp import WriteHttpClient

TOKEN = "savetax_test-token_1"


def _resp(status: int, payload: object) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(status=status, headers={}, body=body, elapsed_seconds=0.01)


def _receipt_body(**over: object) -> dict:
    base: dict = {
        "submission_id": "sid-1",
        "claim_id": "claim-1",
        "revision": 1,
        "submission_state": "accepted",
        "admitted_at": None,
        "amount_krw": 100,
        "lint_status": "pending",
    }
    base.update(over)
    return base


def _valid_body() -> dict:
    return {"canonical_product_id": "item-1", "dataset_release_id": "r1", "ruleset_version": "v1"}


def _harness(
    write_script: list,
    http_script: list | None = None,
    config: SaveTaxConfig | None = None,
) -> tuple[SubmissionClient, list, list, list[float]]:
    cfg = config or SaveTaxConfig()
    write_calls: list = []
    http_calls: list = []
    sleeps: list[float] = []
    queue = list(write_script)
    http_queue = list(http_script or [])

    def fake_write(method: str, url: str, headers: object, body: object, timeout: float) -> HttpResponse:
        write_calls.append((method, url, dict(headers), body))  # type: ignore[arg-type]
        assert queue, "unexpected extra write"
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def fake_http(url: str, headers: object, timeout: float) -> HttpResponse:
        http_calls.append((url, dict(headers)))  # type: ignore[arg-type]
        assert http_queue, "unexpected extra http"
        return http_queue.pop(0)

    write = WriteHttpClient(cfg, Secret(TOKEN), transport=fake_write, sleep=sleeps.append)  # type: ignore[arg-type]
    http = HttpClient(cfg, Secret(TOKEN), transport=fake_http, sleep=sleeps.append)  # type: ignore[arg-type]
    read_http = HttpClient(cfg, Secret(TOKEN), transport=fake_http, sleep=sleeps.append)  # type: ignore[arg-type]
    read = CompetitionClient(read_http, cfg, sleep=sleeps.append)
    client = SubmissionClient(write, read, http, cfg, sleep=sleeps.append)
    return client, write_calls, http_calls, sleeps


def _err(code: str, status: int = 409) -> HttpResponse:
    return _resp(status, {"code": code, "message": "m", "request_id": "r", "retryable": False})


def test_preview_parses_dry_run() -> None:
    body = {"ok": True, "total_amount_krw": 50, "lines": [{"fact_id": "f1", "amount_krw": 50}], "next_action": "submit"}
    client, _, _, _ = _harness([_resp(200, body)])
    result = client.preview(_valid_body())
    assert result.ok is True
    assert result.total_amount_krw == 50
    assert result.line_amounts_krw == {"f1": 50}
    assert result.next_action == "submit"


def test_preview_rejects_unknown_next_action() -> None:
    body = {"ok": True, "total_amount_krw": 1, "lines": [], "next_action": "other"}
    client, _, _, _ = _harness([_resp(200, body)])
    with pytest.raises(DataIntegrityError):
        client.preview(_valid_body())


def test_submit_returns_receipt() -> None:
    client, _, _, _ = _harness([_resp(201, _receipt_body())])
    receipt = client.submit(_valid_body(), idempotency_key="K1")
    assert receipt.submission_id == "sid-1"
    assert receipt.lint_status == "pending"


def test_retry_reuses_key_and_bytes() -> None:
    net = NetworkError(NetworkStage.TCP_TLS_CONNECT, "h", "down")
    client, calls, _, sleeps = _harness([net, _resp(201, _receipt_body())])
    client.submit(_valid_body(), idempotency_key="K1")
    assert len(calls) == 2
    assert calls[0][2]["Idempotency-Key"] == calls[1][2]["Idempotency-Key"] == "K1"
    assert calls[0][3] == calls[1][3]
    assert sleeps == [15.0]


def test_gateway_failure_resolves_through_request_log() -> None:
    errs = [_resp(503, {"code": "ADMISSION_UNAVAILABLE", "message": "m", "retryable": True})] * 3
    http_ok = [_resp(200, _receipt_body(submission_id="sid-9"))]
    client, calls, _, _ = _harness(errs, http_ok)
    receipt = client.submit(_valid_body(), idempotency_key="K1")
    assert receipt.submission_id == "sid-9"
    assert len(calls) == 3


def test_pending_request_raises() -> None:
    errs = [_resp(503, {"code": "ADMISSION_UNAVAILABLE", "message": "m", "retryable": True})] * 3
    client, _, _, _ = _harness(errs, [_resp(202, {})])
    with pytest.raises(SubmissionPending):
        client.submit(_valid_body(), idempotency_key="K1")


def test_never_admitted_reraises_last_error() -> None:
    nets = [NetworkError(NetworkStage.TCP_TLS_CONNECT, "h", "d1")] * 3
    not_found = [_resp(404, {"code": "IDEMPOTENCY_KEY_NOT_FOUND", "message": "m", "retryable": False})]
    client, _, _, _ = _harness(nets, not_found)
    with pytest.raises(NetworkError):
        client.submit(_valid_body(), idempotency_key="K1")


def test_duplicate_item_is_not_retried() -> None:
    payload = {
        "code": "PRODUCT_ALREADY_CLAIMED",
        "message": "taken",
        "request_id": "r",
        "retryable": False,
        "context": {"claimed_by_me": True},
    }
    client, calls, _, _ = _harness([_resp(409, payload)])
    with pytest.raises(ApiError) as exc:
        client.submit(_valid_body(), idempotency_key="K1")
    assert len(calls) == 1
    assert exc.value.context == {"claimed_by_me": True}  # type: ignore[attr-defined]


def test_key_reuse_conflict_is_surfaced() -> None:
    client, calls, _, _ = _harness([_err("IDEMPOTENCY_KEY_REUSED")])
    with pytest.raises(ApiError):
        client.submit(_valid_body(), idempotency_key="K1")
    assert len(calls) == 1


def test_revision_conflict_is_surfaced() -> None:
    client, calls, _, _ = _harness([_resp(409, {"code": "REVISION_CONFLICT", "message": "m", "retryable": False})])
    with pytest.raises(ApiError):
        client.revise("sid-1", _valid_body(), if_match=1, idempotency_key="K1")
    assert len(calls) == 1


def test_withdraw_sends_no_body() -> None:
    client, calls, _, _ = _harness([_resp(200, _receipt_body(submission_state="withdrawn"))])
    client.withdraw("sid-1", idempotency_key="K1")
    assert calls[0][3] is None
    assert "If-Match" not in calls[0][2]


def test_revise_sends_if_match() -> None:
    client, calls, _, _ = _harness([_resp(200, _receipt_body())])
    client.revise("sid-1", _valid_body(), if_match=2, idempotency_key="K1")
    assert calls[0][2]["If-Match"] == "2"


def test_held_is_derived() -> None:
    detail = {"submission_id": "sid-1", "state": "accepted", "lint_status": "suspect", "revision": 1}
    client, _, _, _ = _harness(
        [], [HttpResponse(status=200, headers={}, body=json.dumps(detail).encode(), elapsed_seconds=0.01)]
    )
    assert client.get_submission("sid-1").held is True
    clean = {"submission_id": "sid-1", "state": "accepted", "lint_status": "clean", "revision": 1}
    client2, _, _, _ = _harness([], [_resp(200, clean)])
    assert client2.get_submission("sid-1").held is False


def test_detail_tolerates_missing_optionals() -> None:
    client, _, _, _ = _harness([], [_resp(200, {"id": "sid-1", "lint_status": "pending"})])
    detail = client.get_submission("sid-1")
    assert detail.submission_id == "sid-1"
    assert detail.lint_lines == ()
    assert detail.lint_hints == ()


def test_lint_hints_are_exposed() -> None:
    body = {
        "submission_id": "sid-1",
        "lint_status": "suspect",
        "lint": {"hints": [{"code": "page_not_read", "message": "m"}]},
    }
    client, _, _, _ = _harness([], [_resp(200, body)])
    detail = client.get_submission("sid-1")
    assert detail.lint_hints[0].code == "page_not_read"


def test_invalid_submission_id_rejected() -> None:
    client, calls, http_calls, _ = _harness([])
    with pytest.raises(ConfigError):
        client.get_submission("../x")
    with pytest.raises(ConfigError):
        client.withdraw("bad id!", idempotency_key="K1")
    assert calls == []
    assert http_calls == []


def test_no_body_or_key_reaches_log(caplog: pytest.LogCaptureFixture) -> None:
    client, _, _, _ = _harness([_resp(201, _receipt_body(amount_krw=777))])
    body = {**_valid_body(), "quote": "secret-quote-xyz"}
    with caplog.at_level(logging.INFO):
        client.submit(body, idempotency_key="K1-secret")
    assert "secret-quote-xyz" not in caplog.text
    assert "K1-secret" not in caplog.text


def test_submit_rejects_incomplete_body() -> None:
    client, calls, _, _ = _harness([])
    with pytest.raises(ConfigError):
        client.submit({"canonical_product_id": ""}, idempotency_key="K1")
    assert calls == []


def test_revise_rejects_bad_if_match() -> None:
    client, calls, _, _ = _harness([])
    with pytest.raises(ConfigError):
        client.revise("sid-1", _valid_body(), if_match=0, idempotency_key="K1")
    assert calls == []


def test_submit_rejects_wrong_state() -> None:
    client, _, _, _ = _harness([_resp(201, _receipt_body(submission_state="pending"))])
    with pytest.raises(DataIntegrityError):
        client.submit(_valid_body(), idempotency_key="K1")


def test_rate_limited_retries_then_succeeds() -> None:
    limited = _resp(429, {"code": "RATE_LIMITED", "message": "slow", "retryable": True})
    client, calls, _, sleeps = _harness([limited, _resp(201, _receipt_body())])
    client.submit(_valid_body(), idempotency_key="K1")
    assert len(calls) == 2
    assert sleeps == [15.0]


def test_unknown_error_code_raises_without_retry() -> None:
    client, calls, _, _ = _harness([_resp(422, {"code": "SCHEMA_INVALID", "message": "bad", "retryable": False})])
    with pytest.raises(ApiError):
        client.submit(_valid_body(), idempotency_key="K1")
    assert len(calls) == 1


def test_list_parses_multiple_and_logs() -> None:
    payload = {
        "submissions": [
            {"submission_id": "a", "lint_status": "clean", "state": "accepted", "revision": 1},
            {"submission_id": "b", "lint_status": "pending", "revision": 2},
        ]
    }
    client, _, _, _ = _harness([], [_resp(200, payload)])
    details = client.list_my_submissions()
    assert [d.submission_id for d in details] == ["a", "b"]


def test_resolve_handles_all_branches() -> None:
    client, _, _, _ = _harness([], [_resp(200, _receipt_body(submission_id="s1"))])
    assert client.resolve_request("K1").state == "receipt"
    client2, _, _, _ = _harness([], [_resp(202, {})])
    assert client2.resolve_request("K1").state == "pending"
    client3, _, _, _ = _harness([], [_resp(404, {"code": "IDEMPOTENCY_KEY_NOT_FOUND", "message": "m"})])
    assert client3.resolve_request("K1").state == "not_found"
    client4, _, _, _ = _harness([], [_resp(404, {"code": "OTHER", "message": "m"})])
    with pytest.raises(ApiError):
        client4.resolve_request("K1")
    client5, _, _, _ = _harness([], [_resp(500, {"code": "X", "message": "m"})])
    with pytest.raises(ApiError):
        client5.resolve_request("K1")
    client6, _, _, _ = _harness([], [])
    with pytest.raises(ConfigError):
        client6.resolve_request("bad key!")


def test_preview_retries_rate_limit_and_server_errors() -> None:
    ok = {"ok": True, "total_amount_krw": 5, "lines": [], "next_action": "submit"}
    client, calls, _, sleeps = _harness([_resp(429, b"<html>slow down</html>"), _resp(503, b""), _resp(200, ok)])
    assert client.preview(_valid_body()).total_amount_krw == 5
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_preview_gives_up_after_three_rate_limited_attempts() -> None:
    client, calls, _, sleeps = _harness([_resp(429, b"x"), _resp(429, b"x"), _resp(429, b"x")])
    with pytest.raises(ApiError):
        client.preview(_valid_body())
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_preview_retries_network_failure_then_raises() -> None:
    client, calls, _, sleeps = _harness([NetworkError(NetworkStage.TCP_TLS_CONNECT, "h", "down")] * 3)
    with pytest.raises(NetworkError):
        client.preview(_valid_body())
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_preview_error_and_malformed() -> None:
    client, _, _, _ = _harness([_resp(409, {"code": "PRODUCT_ALREADY_CLAIMED", "message": "m"})])
    with pytest.raises(ApiError):
        client.preview(_valid_body())
    client2, _, _, _ = _harness([_resp(200, {"ok": "yes", "total_amount_krw": 1, "next_action": "submit"})])
    with pytest.raises(DataIntegrityError):
        client2.preview(_valid_body())
    client3, _, _, _ = _harness([_resp(200, b"nope")])
    with pytest.raises(DataIntegrityError):
        client3.preview(_valid_body())


def test_lint_lines_parsed() -> None:
    body = {
        "submission_id": "sid-1",
        "lint_status": "clean",
        "lint_lines": [{"fact_id": "f1", "verdict": "ok", "reason": "r"}],
    }
    client, _, _, _ = _harness([], [_resp(200, body)])
    assert client.get_submission("sid-1").lint_lines[0].fact_id == "f1"


def test_helper_branch_coverage() -> None:
    import src.savetax.submissions as sub

    with pytest.raises(DataIntegrityError):
        sub._decode_object(b"nope", "x")
    with pytest.raises(DataIntegrityError):
        sub._decode_object(b"[1,2]", "x")
    err = sub._flat_error("competition", 500, b"nope")
    assert err.code == "UNKNOWN_ERROR"
    err2 = sub._flat_error("competition", 500, b'{"no":"code"}')
    assert err2.code == "UNKNOWN_ERROR"
    assert sub._retryable(429, "RATE_LIMITED") is True
    assert sub._retryable(429, "OTHER") is False
    assert sub._retryable(503, "ADMISSION_UNAVAILABLE") is True
    assert sub._retryable(503, "OTHER") is True
    assert sub._retryable(500, "X") is True
    assert sub._retryable(400, "X") is False
    assert sub._retryable(500, "SCHEMA_INVALID") is False
    with pytest.raises(DataIntegrityError):
        sub._parse_krw(True, "f")
    assert sub._parse_krw("42", "f") == 42
    with pytest.raises(DataIntegrityError):
        sub._parse_krw("abc", "f")
    assert sub._opt_krw({}, ("a",)) is None
    assert sub._parse_moment_or_none(None) is None
    assert sub._parse_moment_or_none("") is None
    with pytest.raises(DataIntegrityError):
        sub._parse_moment_or_none("bad")
    with pytest.raises(DataIntegrityError):
        sub._parse_moment_or_none("2026-02-30T00:00:00.000Z")
    assert sub._parse_moment_or_none("2026-10-11T22:00:00.000Z") is not None
    for bad in ({}, {"submission_id": ""}, {"submission_id": "s"}):
        with pytest.raises(DataIntegrityError):
            sub._parse_receipt(bad, None)
    with pytest.raises(DataIntegrityError):
        sub._parse_receipt({"submission_id": "s", "revision": True, "submission_state": "a", "amount_krw": 1}, None)
    with pytest.raises(DataIntegrityError):
        sub._parse_receipt({"submission_id": "s", "revision": 1, "submission_state": "", "amount_krw": 1}, None)
    with pytest.raises(DataIntegrityError):
        sub._parse_receipt({"submission_id": "s", "revision": 1, "submission_state": "a"}, None)
    with pytest.raises(DataIntegrityError):
        sub._parse_detail({})
    with pytest.raises(DataIntegrityError):
        sub._parse_detail({"submission_id": "s", "lint_status": "pending", "revision": True})
    with pytest.raises(DataIntegrityError):
        sub._parse_detail({"submission_id": "s", "lint_status": "weird"})
    with pytest.raises(DataIntegrityError):
        sub._parse_detail({"submission_id": "s"})
    with pytest.raises(DataIntegrityError):
        sub._parse_lines({"lint_lines": "x"})
    with pytest.raises(DataIntegrityError):
        sub._parse_lines({"lint_lines": ["x"]})
    with pytest.raises(DataIntegrityError):
        sub._parse_lines({"lint_lines": [{}]})
    with pytest.raises(DataIntegrityError):
        sub._parse_hints({"lint_hints": "x"})
    with pytest.raises(DataIntegrityError):
        sub._parse_hints({"lint_hints": ["x"]})
    with pytest.raises(DataIntegrityError):
        sub._parse_hints({"lint_hints": [{}]})
    with pytest.raises(DataIntegrityError):
        sub._parse_hints({"lint": {"hints": [{}]}})
    with pytest.raises(DataIntegrityError):
        SubmissionClient._preview_lines({"lines": "x"})  # type: ignore[dict-item]
    with pytest.raises(DataIntegrityError):
        SubmissionClient._preview_lines({"lines": ["x"]})
    with pytest.raises(DataIntegrityError):
        SubmissionClient._preview_lines({"lines": [{}]})
    assert SubmissionClient._preview_lines({}) == {}
    assert SubmissionClient._preview_lines({"line_amounts_krw": {"f1": "7"}}) == {"f1": 7}
    with pytest.raises(ConfigError):
        sub._check_sid("bad id!")
    with pytest.raises(ConfigError):
        sub._check_key("bad key!")
    assert (
        sub._parse_receipt(
            {
                "submission_id": "s",
                "revision": 1,
                "submission_state": "a",
                "amount_krw": 1,
                "lint": {"status": "clean"},
            },
            None,
        ).lint_status
        == "clean"
    )
    assert sub._parse_detail({"submission_id": "s", "lint": {"status": "clean"}, "amount_krw": "12"}).amount_krw == 12
    client_nested = _harness([], [_resp(200, {"submission_id": "s", "lint": {"status": "clean"}})])[0]
    assert client_nested.get_submission("s").lint_status == "clean"
    bad_preview = {"ok": True, "next_action": "submit", "lines": []}
    client_bad, _, _, _ = _harness([_resp(200, bad_preview)])
    with pytest.raises(DataIntegrityError):
        client_bad.preview(_valid_body())


def test_read_error_paths_raise() -> None:
    client, _, _, _ = _harness([], [_resp(500, {"code": "X", "message": "m"})])
    with pytest.raises(ApiError):
        client.get_submission("sid-1")
    client2, _, _, _ = _harness([], [_resp(500, {"code": "X", "message": "m"})])
    with pytest.raises(ApiError):
        client2.list_my_submissions()
    client3, _, _, _ = _harness([], [_resp(200, {"no": "list"})])
    with pytest.raises(DataIntegrityError):
        client3.list_my_submissions()
    client4, _, _, _ = _harness([], [_resp(200, {"submission": {"submission_id": "s1", "lint_status": "clean"}})])
    assert client4.list_my_submissions()[0].submission_id == "s1"
    client5, _, _, _ = _harness([_resp(200, _receipt_body())])
    with pytest.raises(ConfigError):
        client5.submit(_valid_body(), idempotency_key="bad key!")


def test_documented_api_shapes_and_paths() -> None:
    preview = {
        "ok": True,
        "next_action": "revise_existing",
        "total_amount_krw": "50",
        "lines": [{"fact_id": "dl:1:C:R:1", "score_amount_krw": "50"}],
        "existing_submission_id": "sid-1",
        "current_revision": 2,
    }
    client, _, calls, _ = _harness(
        [_resp(200, preview), _resp(201, _receipt_body(api_path="/v1/receipt"))],
        [_resp(200, {"registered": True, "submissions": []})],
    )
    assert client.preview(_valid_body()).line_amounts_krw == {"dl:1:C:R:1": 50}
    assert client.submit(_valid_body(), idempotency_key="K").api_path == "/v1/receipt"
    assert client.list_my_submissions() == []
    assert calls[0][0].endswith("/me/submissions")
    assert calls[0][1]["Authorization"] == f"Bearer {TOKEN}"


@pytest.mark.parametrize("key", ["A/B?C#D%E\\F", "..", ".", "%", "!" * 128])
def test_request_key_remains_one_opaque_segment(key: str) -> None:
    from urllib.parse import unquote, urlparse

    client, _, calls, _ = _harness([], [_resp(202, {})])
    assert client.resolve_request(key).state == "pending"
    parsed = urlparse(calls[0][0])
    assert not parsed.query
    assert not parsed.fragment
    assert unquote(parsed.path.rsplit("/", 1)[1]) == key
    assert calls[0][1]["Authorization"] == f"Bearer {TOKEN}"


def test_nested_body_is_frozen_across_cooldown() -> None:
    body = {**_valid_body(), "lines": [{"quote": "original"}]}
    client, calls, _, _ = _harness(
        [NetworkError(NetworkStage.TCP_TLS_CONNECT, "h", "down"), _resp(201, _receipt_body())]
    )

    def mutate(_seconds: float) -> None:
        body["lines"][0]["quote"] = "changed"

    client._sleep = mutate
    client.submit(body, idempotency_key="same-key")
    assert calls[0][3] == calls[1][3]
    assert json.loads(calls[1][3])["lines"][0]["quote"] == "original"


@pytest.mark.parametrize("code", ["ADMISSION_UNAVAILABLE", "OTHER_GATEWAY_FAILURE"])
def test_all_gateway_failures_retry_and_recover(code: str) -> None:
    client, calls, reads, sleeps = _harness([_err(code, 503)] * 3, [_resp(200, _receipt_body())])
    assert client.submit(_valid_body(), idempotency_key="same-key").submission_state == "accepted"
    assert len(calls) == 3
    assert len(reads) == 1
    assert len(sleeps) == 2


def test_recovered_receipt_cannot_bypass_state_validation() -> None:
    client, _, _, _ = _harness(
        [_err("ADMISSION_UNAVAILABLE", 503)] * 3, [_resp(200, _receipt_body(submission_state="withdrawn"))]
    )
    with pytest.raises(DataIntegrityError, match="resolved receipt state"):
        client.submit(_valid_body(), idempotency_key="K")


def test_pending_exposes_key_without_rendering_it() -> None:
    key = "private-request-key"
    client, _, _, _ = _harness([_err("ADMISSION_UNAVAILABLE", 503)] * 3, [_resp(202, {})])
    with pytest.raises(SubmissionPending) as caught:
        client.submit(_valid_body(), idempotency_key=key)
    assert caught.value.key == key
    assert key not in str(caught.value)
    assert key not in repr(caught.value)


@pytest.mark.parametrize("over", [{"lint_status": "other"}, {"api_path": []}, {"revision": 0}])
def test_malformed_receipts_fail_closed(over: dict) -> None:
    client, _, _, _ = _harness([_resp(201, _receipt_body(**over))])
    with pytest.raises(DataIntegrityError):
        client.submit(_valid_body(), idempotency_key="K")


@pytest.mark.parametrize("payload", [{"submissions": [None]}, {"submissions": ["bad"]}])
def test_non_object_list_entries_are_domain_errors(payload: dict) -> None:
    client, _, _, _ = _harness([], [_resp(200, payload)])
    with pytest.raises(DataIntegrityError):
        client.list_my_submissions()


@pytest.mark.parametrize("next_action", [[], {}, None])
def test_non_string_next_action_is_domain_error(next_action: object) -> None:
    client, _, _, _ = _harness([_resp(200, {"ok": True, "next_action": next_action})])
    with pytest.raises(DataIntegrityError):
        client.preview(_valid_body())


def test_conflicting_fact_amounts_are_not_overwritten() -> None:
    payload = {
        "ok": True,
        "next_action": "submit",
        "total_amount_krw": 3,
        "lines": [{"fact_id": "f", "score_amount_krw": "1"}, {"fact_id": "f", "score_amount_krw": "2"}],
    }
    client, _, _, _ = _harness([_resp(200, payload)])
    with pytest.raises(DataIntegrityError, match="conflicting"):
        client.preview(_valid_body())


def test_non_json_body_fails_before_io() -> None:
    client, calls, _, _ = _harness([])
    with pytest.raises(DataIntegrityError):
        client.submit({**_valid_body(), "value": object()}, idempotency_key="K")
    assert calls == []


def test_detail_keeps_known_item_and_current_revision() -> None:
    payload = {"id": "s", "item_id": "item-1", "current_revision": 2, "lint_status": "clean"}
    client, _, _, _ = _harness([], [_resp(200, payload)])
    detail = client.get_submission("s")
    assert detail.item_id == "item-1"
    assert detail.revision == 2


@pytest.mark.parametrize("value", ["../x", 1, None, "a\n"])
def test_preview_product_id_is_validated_before_io(value: object) -> None:
    client, calls, _, _ = _harness([])
    with pytest.raises(ConfigError):
        client.preview({**_valid_body(), "canonical_product_id": value})
    assert calls == []
