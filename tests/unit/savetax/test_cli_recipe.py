"""Invariant guards for the recipe lane wiring."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest

from src.savetax.cli import _build_parser, _run_recipe_lane
from src.savetax.competition import EventInfo, Fact, ItemDetail, ItemState, LookupResult
from src.savetax.config import AuditConfig, EvidenceConfig, SaveTaxConfig
from src.savetax.errors import ConfigError
from src.savetax.http import HttpResponse
from src.savetax.publicweb import PublicFetcher
from src.savetax.rawapi import ListingDetail


def _attrs() -> dict[str, str]:
    return {
        "중앙처리장치(CPU)규격": "Intel Core Ultra 5 225",
        "메모리(RAM)용량": "16",
        "SSD용량": "512",
        "하드디스크(HDD)용량": "미장착",
        "메인보드": "H810",
        "스위칭모드파워서플라이형식": "500W",
        "컴퓨터형식": "타워",
        "그래픽": "내장그래픽",
        "운영체계": "옵션제공",
        "광디스크(ODD)규격": "옵션제공",
    }


def _product_html(name: str, low: int) -> str:
    items = (
        '<li><span aria-label="몰A"></span><span class="text__num">100000</span></li>'
        '<li><span aria-label="몰B"></span><span class="text__num">101000</span></li>'
        '<li><span aria-label="몰C"></span><span class="text__num">102000</span></li>'
    )
    return (
        "<html><head><script type=\"application/ld+json\">"
        f'{{\"@type": "Product", "name": "{name}", "brand": {{"name": "B"}}, '
        f'"description": "d", "offers": {{"lowPrice": {low}, "offerCount": 3}}}}'
        "</script></head><body>"
        f'<ul class="list__mall-price">{items}</ul></body></html>'
    )


_NAMES = {
    "74123084": "울트라5 225 정품",
    "74123036": "울트라5 235 정품",
    "74122856": "울트라7 265",
    "32038580": "i5 14400 정품",
    "32038256": "i5 14500 정품",
    "32038001": "i7 14700 정품",
    "18965774": "DDR5 5600 16GB",
    "29655680": "T500 500GB",
    "13522751": "WD Blue 2TB",
    "28183691": "H610M",
    "104616128": "H810M",
    "20738630": "B760M",
    "104966246": "B860M",
    "19281344": "500W",
    "18564248": "400 TFX",
    "33990908": "TA20",
    "34740608": "미니멀",
    "21206318": "KM C1",
}


def _fetcher(evidence: EvidenceConfig) -> PublicFetcher:
    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        pcode = url.rsplit("pcode=", 1)[-1]
        body = _product_html(_NAMES.get(pcode, "X"), 100000)
        return HttpResponse(status=200, headers={}, body=body.encode(), elapsed_seconds=0.01)

    return PublicFetcher(SaveTaxConfig(), evidence, transport=transport)  # type: ignore[arg-type]


def _listing(lid: str, item: str, contract: str) -> ListingDetail:
    return ListingDetail(lid, item, "대", Decimal(2000000), "contract", contract, None, None, None,
                         "v", None, None, None, "납품장소도", "서울", None)


class _Raw:
    def __init__(self, attrs: dict | None) -> None:
        self._attrs = attrs

    def get_item_attributes(self, item_id: str) -> dict | None:
        return self._attrs

    def get_listing(self, lid: str) -> ListingDetail | None:
        return _listing(lid, "I1", "C1")


class _Competition:
    def __init__(self, facts: tuple) -> None:
        self._facts = facts

    def lookup_items(self, ids: list[str]) -> list[LookupResult]:
        return [LookupResult(i, "product", i, ItemState.AVAILABLE) for i in ids]

    def get_item(self, item_id: str) -> ItemDetail:
        return ItemDetail(item_id, ItemState.AVAILABLE, "n", "m", None, None, None, len(self._facts), False, self._facts)

    def get_event(self) -> EventInfo:
        now = datetime.now(UTC)
        return EventInfo("d", "r", "running", now, now, "v2", "catalog")


def _fact() -> Fact:
    return Fact("f1", "L1", "C1", "R1", "1", date(2026, 1, 5), "대", "tax", "B1", "none",
                Decimal(2000000), Decimal(2), Decimal(4000000), True, ())


def _dataset(tmp_path: Path) -> Any:
    con = duckdb.connect()
    con.execute("CREATE TABLE listings(listing_id VARCHAR, item_id VARCHAR, detail_category VARCHAR, contract_no VARCHAR, dlv_key INTEGER, dlv_qty INTEGER, dlv_amount INTEGER)")
    con.execute("INSERT INTO listings VALUES ('L1','I1','데스크톱컴퓨터','C1',1,2,4000000)")
    return SimpleNamespace(connection=con)


def test_recipe_lane_writes_bundle_and_draft(tmp_path: Path) -> None:
    evidence = EvidenceConfig()
    out = tmp_path / "out"
    n, avail, drafts = _run_recipe_lane(_Competition((_fact(),)), _Raw(_attrs()), _fetcher(evidence),
                                        evidence, AuditConfig(), out, _dataset(tmp_path), None, 10, False, "desktop")
    assert (n, avail, drafts) == (1, 1, 1)
    assert (out / "recipe-bundles.jsonl").is_file()
    assert (out / "drafts" / "recipe" / "I1.json").is_file()


def test_recipe_lane_skips_without_attributes(tmp_path: Path) -> None:
    evidence = EvidenceConfig()
    out = tmp_path / "out"
    n, avail, drafts = _run_recipe_lane(_Competition((_fact(),)), _Raw(None), _fetcher(evidence),
                                        evidence, AuditConfig(), out, _dataset(tmp_path), None, 10, False, "desktop")
    assert drafts == 0


def test_recipe_lane_skips_unmapped(tmp_path: Path) -> None:
    bad = _attrs()
    bad["중앙처리장치(CPU)규격"] = "AMD X"
    evidence = EvidenceConfig()
    out = tmp_path / "out"
    _, _, drafts = _run_recipe_lane(_Competition((_fact(),)), _Raw(bad), _fetcher(evidence),
                                    evidence, AuditConfig(), out, _dataset(tmp_path), None, 10, False, "desktop")
    assert drafts == 0


def test_parser_has_recipe_lane_and_owned_rejected() -> None:
    parser = _build_parser()
    args = parser.parse_args(["evidence", "--lane", "recipe", "--recipe-kind", "desktop"])
    assert args.lane == "recipe"
    assert args.recipe_kind == "desktop"
    from src.savetax.cli import _run_evidence

    owned_args = parser.parse_args(["evidence", "--lane", "recipe", "--owned"])
    with pytest.raises(ConfigError, match="owned mode rejects the recipe lane"):
        _run_evidence(owned_args, {"SAVETAX_TOKEN": "savetax_test"}, None)


def _fact2(fact_id: str, contract: str, price: int) -> Fact:
    return Fact(fact_id, "L-" + contract, contract, "R1", "1", date(2026, 1, 5), "대", "tax", "B1", "none",
                Decimal(price), Decimal(2), Decimal(price * 2), True, ())


class _RawMixed:
    def get_item_attributes(self, item_id: str) -> dict | None:
        return _attrs()

    def get_listing(self, lid: str) -> ListingDetail | None:
        if lid == "L-C2":
            return ListingDetail(lid, "I1", "대", Decimal(2000000), "contract", "C2", None, None, None,
                                 "v", None, None, None, "현장설치도", "서울", None)
        return ListingDetail(lid, "I1", "대", Decimal(2000000), "contract", "C1", None, None, None,
                             "v", None, None, None, "납품장소도", "서울", None)


class _CompMixed(_Competition):
    pass


def _dataset2(tmp_path: Path) -> Any:
    con = duckdb.connect()
    con.execute("CREATE TABLE listings(listing_id VARCHAR, item_id VARCHAR, detail_category VARCHAR, contract_no VARCHAR, dlv_key INTEGER, dlv_qty INTEGER, dlv_amount INTEGER)")
    con.execute("INSERT INTO listings VALUES ('L-C1','I1','데스크톱컴퓨터','C1',1,2,4000000),('L-C2','I1','데스크톱컴퓨터','C2',2,1,1000000)")
    return SimpleNamespace(connection=con)


def test_recipe_lane_terms_rejected_appended(tmp_path: Path) -> None:
    from src.savetax.workflow import load_bundles as _load
    from datetime import datetime as _dt

    evidence = EvidenceConfig()
    out = tmp_path / "out"
    facts = (_fact2("f1", "C1", 2000000), _fact2("f2", "C2", 2000000))
    n, avail, drafts = _run_recipe_lane(_Competition(facts), _RawMixed(), _fetcher(evidence),
                                        evidence, AuditConfig(), out, _dataset2(tmp_path), None, 10, False, "desktop")
    assert drafts == 1
    bundles = _load([out / "recipe-bundles.jsonl"], now=_dt.now(UTC))
    assert any(r.reason == "terms_include_installation" for r in bundles["I1"].rejected)


def test_recipe_lane_price_skip_and_bundle_none(tmp_path: Path) -> None:
    evidence = EvidenceConfig()
    out = tmp_path / "out"

    def _bad_transport(url: str, headers: object, timeout: float) -> HttpResponse:
        return HttpResponse(status=503, headers={}, body=b"", elapsed_seconds=0.01)

    bad_fetcher = PublicFetcher(SaveTaxConfig(), evidence, transport=_bad_transport)  # type: ignore[arg-type]
    _, _, drafts = _run_recipe_lane(_Competition((_fact(),)), _Raw(_attrs()), bad_fetcher,
                                    evidence, AuditConfig(), out, _dataset(tmp_path), None, 10, False, "desktop")
    assert drafts == 0
    out2 = tmp_path / "out2"
    _, _, drafts2 = _run_recipe_lane(_Competition((_fact2("f1", "C1", 10),)), _Raw(_attrs()), _fetcher(evidence),
                                     evidence, AuditConfig(), out2, _dataset(tmp_path), None, 10, False, "desktop")
    assert drafts2 == 0


def test_recipe_lane_review_gated_and_api_error(tmp_path: Path) -> None:
    from src.savetax.errors import ApiError

    evidence = EvidenceConfig()
    review_attrs = _attrs()
    review_attrs["중앙처리장치(CPU)규격"] = "i5-13500"
    out = tmp_path / "out"
    _, _, drafts = _run_recipe_lane(_Competition((_fact(),)), _Raw(review_attrs), _fetcher(evidence),
                                    evidence, AuditConfig(), out, _dataset(tmp_path), None, 10, False, "desktop")
    assert drafts == 0
    _, _, drafts2 = _run_recipe_lane(_Competition((_fact(),)), _Raw(review_attrs), _fetcher(evidence),
                                     evidence, AuditConfig(), tmp_path / "out3", _dataset(tmp_path), None, 10, True, "desktop")
    assert drafts2 == 1

    class _Boom(_Competition):
        def get_item(self, item_id: str) -> ItemDetail:
            raise ApiError("competition", 500, "boom", "m", False)

    _, _, drafts3 = _run_recipe_lane(_Boom((_fact(),)), _Raw(_attrs()), _fetcher(evidence),
                                     evidence, AuditConfig(), tmp_path / "out4", _dataset(tmp_path), None, 10, False, "desktop")
    assert drafts3 == 0


def test_run_evidence_recipe_branch_and_owned_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    import src.savetax.cli as cli

    base = {
        "limit": 5, "batch": 0, "batches": 1, "data_dir": None, "out_dir": tmp_path / "o",
        "items": None, "include_review": False, "unit_policy": "annotate",
        "variant_policy": "series", "recipe_kind": "desktop",
    }
    with pytest.raises(ConfigError):
        cli._run_evidence(argparse.Namespace(lane="recipe", owned=True, **base), {"SAVETAX_TOKEN": "savetax_x"}, None)
    calls: dict[str, bool] = {}

    def _fake_run(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
        calls["hit"] = True
        return (0, 0, 0)

    class _FakeDataset:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Any:
            return SimpleNamespace()

        def __exit__(self, *args: Any) -> None:
            return None

    monkeypatch.setattr(cli, "_run_recipe_lane", _fake_run)
    monkeypatch.setattr(cli, "Dataset", _FakeDataset)
    code = cli._run_evidence(argparse.Namespace(lane="recipe", owned=False, **base), {"SAVETAX_TOKEN": "savetax_x"}, None)
    assert code == 0
    assert calls.get("hit") is True


def test_recipe_delivery_checks_are_cached_per_contract(tmp_path: Path) -> None:
    class CountingRaw(_Raw):
        calls = 0

        def get_listing(self, lid: str) -> ListingDetail | None:
            self.calls += 1
            return super().get_listing(lid)

    from dataclasses import replace

    raw = CountingRaw(_attrs())
    facts = (_fact(), replace(_fact(), fact_id="f2", request_no="R2"))
    evidence = EvidenceConfig()
    _, _, drafts = _run_recipe_lane(_Competition(facts), raw, _fetcher(evidence), evidence, AuditConfig(), tmp_path / "out", _dataset(tmp_path), None, 10, False, "desktop")
    assert drafts == 1
    assert raw.calls == 1


def test_workflow_loads_recipe_drafts_and_bundles(tmp_path: Path) -> None:
    from src.savetax.cli import _single_draft, _workflow_inputs
    from src.savetax.evidence import Lane

    evidence = EvidenceConfig()
    out = tmp_path / "out"
    _run_recipe_lane(_Competition((_fact(),)), _Raw(_attrs()), _fetcher(evidence), evidence, AuditConfig(), out, _dataset(tmp_path), None, 10, False, "desktop")
    entries = _workflow_inputs(out / "drafts", out, datetime.now(UTC))
    assert len(entries) == 1
    assert entries[0][2].lane is Lane.RECIPE
    single = _single_draft(out / "drafts" / "recipe" / "I1.json", out, datetime.now(UTC))
    assert single[0][2].lines == entries[0][2].lines
