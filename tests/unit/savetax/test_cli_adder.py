"""Guard tests for adder live prices and recipe lane wiring."""

from __future__ import annotations

from typing import Any

import pytest

from src.savetax.cli import _live_price, _run_recipe_lane
from src.savetax.config import EvidenceConfig, SaveTaxConfig
from src.savetax.http import HttpResponse
from src.savetax.publicweb import PublicFetcher


def _fetcher(body: str, status: int = 200) -> PublicFetcher:
    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        return HttpResponse(status=status, headers={}, body=body.encode(), elapsed_seconds=0.01)

    ev = EvidenceConfig(public_request_interval_seconds=0.0)
    return PublicFetcher(SaveTaxConfig(), ev, transport=transport)  # type: ignore[arg-type]


def test_live_price_recognises_adder_page() -> None:
    fetcher = _fetcher("<html><body>랙 설치 공식 공임 167,310원 · 안내</body></html>")
    assert _live_price(fetcher, "https://savetax.now/176") == 167310


def test_live_price_refuses_page_without_amount() -> None:
    fetcher = _fetcher("<html><body>다른 공임 안내</body></html>")
    assert _live_price(fetcher, "https://savetax.now/176") is None


@pytest.mark.parametrize("adder_body", [None, "공임 안내", "랙 설치 공식 공임 167,310원"])
def test_recipe_lane_skips_unreadable_adder(tmp_path: Any, monkeypatch: Any, caplog: Any, adder_body: str | None) -> None:
    from datetime import date

    from src.savetax.competition import ItemState

    class _Dataset:
        connection: Any = None

    class _Competition:
        def lookup_items(self, ids: list[str]) -> list[Any]:
            from src.savetax.competition import LookupResult

            return [LookupResult(i, "product", i, ItemState.AVAILABLE) for i in ids]

        def get_item(self, item_id: str) -> Any:
            raise AssertionError("detail must not be fetched after adder skip")

        def get_event(self) -> Any:
            from datetime import UTC, datetime

            from src.savetax.competition import EventInfo

            now = datetime.now(UTC)
            return EventInfo("d", "r", "running", now, now, "v2", "catalog")

    class _Raw:
        def get_item_attributes(self, item_id: str) -> dict[str, str] | None:
            return {"외형크기(폭)": "600", "외형크기(높이)": "1200", "외형크기(깊이)": "700"}

        def get_listing(self, lid: str) -> Any:
            raise AssertionError("unexpected")

    def fake_candidates(dataset: Any, category: str, items: Any = None) -> list[str]:
        return ["I1"]

    monkeypatch.setattr("src.savetax.desktoprecipe.find_recipe_candidates", fake_candidates)
    from src.savetax.competition import Fact, ItemDetail
    from decimal import Decimal

    fact = Fact("f1", "L1", "C1", "R1", "1", date(2026, 1, 5), "대", "tax", "B1", "none",
                Decimal(2000000), Decimal(2), Decimal(4000000), True, ())

    class _Competition2(_Competition):
        def get_item(self, item_id: str) -> Any:
            return ItemDetail(item_id, ItemState.AVAILABLE, "허브랙", None, None, None, None, 1, False, (fact,))

        def lookup_items(self, ids: list[str]) -> list[Any]:
            from src.savetax.competition import LookupResult

            return [LookupResult(i, "product", i, ItemState.AVAILABLE) for i in ids]

    from src.savetax.config import AuditConfig

    evidence = EvidenceConfig(public_request_interval_seconds=0.0)

    def lane_transport(url: str, headers: object, timeout: float) -> HttpResponse:
        assert "Authorization" not in headers  # type: ignore[operator]
        if "prod.danawa.com" in url:
            html = (
                "<html><head><script type=\"application/ld+json\">"
                '{"@type": "Product", "name": "LD-SGDR750 H1200", "brand": {"name": "B"}, '
                '"description": "d", "offers": {"lowPrice": 300000, "offerCount": 3}}'
                "</script></head><body><ul class=\"list__mall-price\">"
                '<li><span aria-label="몰A"></span><span class="text__num">300,000</span></li>'
                '<li><span aria-label="몰B"></span><span class="text__num">301,000</span></li>'
                '<li><span aria-label="몰C"></span><span class="text__num">302,000</span></li>'
                "</ul></body></html>"
            )
            return HttpResponse(status=200, headers={}, body=html.encode(), elapsed_seconds=0.01)
        assert url == "https://savetax.now/176"
        return HttpResponse(status=404 if adder_body is None else 200, headers={},
                            body=(adder_body or "").encode(), elapsed_seconds=0.01)

    lane_fetcher = PublicFetcher(SaveTaxConfig(), evidence, transport=lane_transport)  # type: ignore[arg-type]
    out = tmp_path / "out"
    caplog.set_level("INFO")
    n, a, d = _run_recipe_lane(_Competition2(), _Raw(), lane_fetcher, evidence, AuditConfig(), out, _Dataset(), None, 5, False, "rack")
    success = adder_body == "랙 설치 공식 공임 167,310원"
    assert (n, a, d) == (1, 1, int(success))
    draft = out / "drafts" / "recipe" / "I1.json"
    assert draft.exists() == success
    if success:
        import json

        body = json.loads(draft.read_text())
        line = body["lines"][0]
        assert line["benchmark_unit_price_krw"] == "467310"
        assert sum(int(e["observed_price_krw"]) for e in line["price_evidence"]) == 467310
        assert "benchmark_krw=467310" in caplog.text
    else:
        assert "adder_page_unreadable:rack:labor" in caplog.text if adder_body is None else "adder_amount_missing:rack:labor" in caplog.text
