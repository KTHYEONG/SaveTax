"""Invariant guards for panel and rack fixture recipes."""

from __future__ import annotations

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
from src.savetax.fixturerecipe import panel_recipe, rack_recipe
from src.savetax.http import HttpResponse
from src.savetax.publicweb import PublicFetcher
from src.savetax.rawapi import ListingDetail
from tests.unit.savetax.recipe_fixtures import seller_config


def _evidence(**over: Any) -> EvidenceConfig:
    return seller_config(**over)


def _panel_attrs(**over: str) -> dict[str, str]:
    base = {"옵션/기타": "24PORT CAT.6A", "종류": "패치판넬"}
    base.update(over)
    return base


def _panel_name() -> str:
    return "랙캐비닛용패널, 원캐스트, PC6A-R2024G, 패치판넬"


def test_terminated_panel_maps_with_full_floor() -> None:
    recipe = panel_recipe(_panel_name(), _panel_attrs(), _evidence())
    assert not isinstance(recipe, str)
    assert recipe.parts[0][0].key == "panel:24"
    assert recipe.adders[0].key == "panel:labor-terminated"
    assert recipe.floor_krw == 467_649
    assert recipe.review_required is False
    assert recipe.spec_rows[1] == ("등급", "CAT.6A", "CAT.6A")
    assert recipe.spec_rows[2] == ("취부형태", "-", "19인치 랙형")


def test_unterminated_panel_uses_mounted_labour() -> None:
    recipe = panel_recipe("패치판넬 CAT.6, 24Port, 성단제외", {"종류": "패치판넬"}, _evidence())
    assert not isinstance(recipe, str)
    assert recipe.adders[0].key == "panel:labor-mounted"
    assert recipe.floor_krw == 145_164


def test_model_code_port_count_is_not_enough() -> None:
    assert panel_recipe("패치판넬 PC6-R2024 CAT.6A", {"종류": "패치판넬"}, _evidence()) == "ports_unknown"


def test_second_port_count_is_ambiguous() -> None:
    assert panel_recipe("패치", {"종류": "패치판넬", "옵션": "24PORT CAT.6A 48포트"}, _evidence()) == "ports_ambiguous"


def test_small_panel_is_rejected() -> None:
    assert panel_recipe("패치", {"종류": "패치판넬", "옵션": "12PORT CAT.6A"}, _evidence()) == "ports_unknown"


def test_unstated_grade_uses_the_top_copper_reference() -> None:
    recipe = panel_recipe("패치", {"종류": "패치판넬", "옵션": "24PORT"}, _evidence())
    assert not isinstance(recipe, str)
    assert ("등급", "표기 없음", "CAT.6A") in recipe.spec_rows
    assert len(recipe.differences) == 2


@pytest.mark.parametrize("text", ["24PORT CAT.7", "24PORT CAT8", "24PORT 광패치", "24PORT LC duplex"])
def test_grade_above_the_reference_is_rejected(text: str) -> None:
    assert panel_recipe("패치", {"종류": "패치판넬", "옵션": text}, _evidence()) == "category_out_of_class"


def test_non_panel_is_rejected() -> None:
    assert panel_recipe("보호커버 CAT.6 24PORT", {"종류": "보호패널"}, _evidence()) == "not_patch_panel"


def test_mount_attribute_used_in_spec_rows() -> None:
    recipe = panel_recipe(_panel_name(), _panel_attrs(**{"취부형태": "19인치 랙마운트"}), _evidence())
    assert not isinstance(recipe, str)
    assert recipe.spec_rows[2] == ("취부형태", "19인치 랙마운트", "19인치 랙형")


def test_missing_part_and_adder_raise() -> None:
    from src.savetax.errors import ConfigError
    from src.savetax.fixturerecipe import _adder_by_key, _by_key

    with pytest.raises(ConfigError, match="recipe_parts"):
        _by_key((), "panel:24")
    with pytest.raises(ConfigError, match="recipe_adders"):
        _adder_by_key((), "panel:labor-terminated")
    thin = _evidence(recipe_adders=tuple(a for a in _evidence().recipe_adders if a.key.startswith("desktop")))
    with pytest.raises(ConfigError, match="recipe_adders"):
        panel_recipe(_panel_name(), _panel_attrs(), thin)


def _rack_attrs(width: str = "600", height: str = "1195", depth: str = "750", **over: str) -> dict[str, str]:
    base = {"외형크기(폭)": width, "외형크기(높이)": height, "외형크기(깊이)": depth}
    base.update(over)
    return base


def test_rack_maps_to_covering_class() -> None:
    first = rack_recipe("허브랙", _rack_attrs(), _evidence())
    assert not isinstance(first, str)
    assert first.parts[0][0].key == "rack:1200"
    assert first.floor_krw == 563_310
    second = rack_recipe("허브랙", _rack_attrs(height="1421.5"), _evidence())
    assert not isinstance(second, str)
    assert second.parts[0][0].key == "rack:1600"
    assert second.floor_krw == 937_310


def test_rack_boundary_stays_in_class() -> None:
    recipe = rack_recipe("허브랙", _rack_attrs(height="1800"), _evidence())
    assert not isinstance(recipe, str)
    assert recipe.parts[0][0].key == "rack:1800"


@pytest.mark.parametrize(
    ("attrs", "reason"),
    [
        (_rack_attrs(height="2200"), "height_out_of_class"),
        (_rack_attrs(width="800"), "width_out_of_class"),
        (_rack_attrs(depth="1000"), "depth_out_of_class"),
    ],
)
def test_rack_out_of_class_rejected(attrs: dict[str, str], reason: str) -> None:
    assert rack_recipe("허브랙", attrs, _evidence()) == reason


@pytest.mark.parametrize("token", ["내진", "서버랙", "CCTV"])
def test_special_racks_rejected(token: str) -> None:
    assert rack_recipe(f"허브랙 {token}", _rack_attrs(), _evidence()) == "special_rack"


def test_missing_dimension_rejected() -> None:
    attrs = _rack_attrs()
    del attrs["외형크기(깊이)"]
    assert rack_recipe("허브랙", attrs, _evidence()) == "dimensions_unknown"
    assert rack_recipe("허브랙", _rack_attrs(width="-"), _evidence()) == "dimensions_unknown"
    assert rack_recipe("허브랙", _rack_attrs(height="1.2.3"), _evidence()) == "dimensions_unknown"


def test_accessory_wording_forces_review() -> None:
    recipe = rack_recipe("허브랙 측면개폐", _rack_attrs(), _evidence())
    assert not isinstance(recipe, str)
    assert recipe.review_required is True
    plain = rack_recipe("허브랙", _rack_attrs(), _evidence())
    assert not isinstance(plain, str)
    assert plain.review_required is False


def test_dimension_units_stripped() -> None:
    recipe = rack_recipe("허브랙", _rack_attrs(width="600 mm", height="1,800mm"), _evidence())
    assert not isinstance(recipe, str)
    assert recipe.parts[0][0].key == "rack:1800"
    assert recipe.spec_rows[0] == ("폭", "600mm", "600mm")


@pytest.mark.parametrize("dimension", ["외형크기(폭)", "외형크기(높이)", "외형크기(깊이)"])
def test_zero_dimensions_are_unknown(dimension: str) -> None:
    attrs = _rack_attrs()
    attrs[dimension] = "0 mm"
    assert rack_recipe("허브랙", attrs, _evidence()) == "dimensions_unknown"


def test_rack_class_comparisons_preserve_decimal_boundaries() -> None:
    recipe = rack_recipe("허브랙", _rack_attrs(height="1200.0000000000000001"), _evidence())
    assert not isinstance(recipe, str)
    assert recipe.parts[0][0].key == "rack:1400"
    assert recipe.spec_rows[1] == ("높이", "1200.0000000000000001mm", "1400mm")
    assert rack_recipe("허브랙", _rack_attrs(width="650.0000000000000001"), _evidence()) == "width_out_of_class"
    assert rack_recipe("허브랙", _rack_attrs(depth="750.0000000000000001"), _evidence()) == "depth_out_of_class"


_NAMES = {
    "35026826": "IN-50U6A24P 24포트",
    "75600215": "LD-SGDR750 H1200",
    "75600224": "LD-SGDR750 H1400",
    "75600230": "LD-SGDR750 H1600",
    "75600233": "LD-SGDR750 H1800",
    "75600236": "LD-SGDR750 H2000",
}


def _product_html(name: str) -> str:
    items = (
        '<li><span aria-label="몰A"></span><span class="text__num">100000</span></li>'
        '<li><span aria-label="몰B"></span><span class="text__num">101000</span></li>'
        '<li><span aria-label="몰C"></span><span class="text__num">102000</span></li>'
    )
    return (
        "<html><head><script type=\"application/ld+json\">"
        f'{{\"@type": "Product", "name": "{name}", "brand": {{"name": "B"}}, '
        '"description": "d", "offers": {"lowPrice": 100000, "offerCount": 3}}'
        "</script></head><body>"
        f'<ul class="list__mall-price">{items}</ul></body></html>'
    )


def _fetcher(evidence: EvidenceConfig) -> PublicFetcher:
    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        if url in ("https://savetax.now/222", "https://savetax.now/621", "https://savetax.now/176"):
            amounts = {"https://savetax.now/222": "401,649원 표준품셈", "https://savetax.now/621": "79,164원 표준품셈",
                       "https://savetax.now/176": "167,310원 표준품셈"}
            return HttpResponse(status=200, headers={}, body=amounts[url].encode(), elapsed_seconds=0.01)
        if "pcode=" not in url:
            title = "IN-50U6A24P 24포트 LD-SGDR750 H1200 H1400 H1600 H1800 H2000"
            body = f"<html><head><title>{title}</title></head><body><p>판매가 100,000원</p></body></html>"
            return HttpResponse(status=200, headers={}, body=body.encode(), elapsed_seconds=0.01)
        pcode = url.rsplit("pcode=", 1)[-1]
        return HttpResponse(status=200, headers={}, body=_product_html(_NAMES[pcode]).encode(), elapsed_seconds=0.01)

    return PublicFetcher(SaveTaxConfig(), evidence, transport=transport)  # type: ignore[arg-type]


def _fact(price: int = 2000000) -> Fact:
    return Fact("f1", "L1", "C1", "R1", "1", date(2026, 1, 5), "대", "tax", "B1", "none",
                Decimal(price), Decimal(2), Decimal(price * 2), True, ())


class _Raw:
    def __init__(self, attrs: dict | None) -> None:
        self._attrs = attrs

    def get_item_attributes(self, item_id: str) -> dict | None:
        return self._attrs

    def get_listing(self, lid: str) -> ListingDetail | None:
        raise AssertionError("Fixture recipes must include labour without filtering delivery terms")


class _Competition:
    def __init__(self, name: str, facts: tuple = ()) -> None:
        self._name = name
        self._facts = facts or (_fact(),)

    def lookup_items(self, ids: list[str]) -> list[LookupResult]:
        return [LookupResult(i, "product", i, ItemState.AVAILABLE) for i in ids]

    def get_item(self, item_id: str) -> ItemDetail:
        return ItemDetail(item_id, ItemState.AVAILABLE, self._name, "m", None, None, None,
                          len(self._facts), False, self._facts)

    def get_event(self) -> EventInfo:
        now = datetime.now(UTC)
        return EventInfo("d", "r", "running", now, now, "v2", "catalog")


def _dataset(category: str) -> Any:
    con = duckdb.connect()
    con.execute("CREATE TABLE listings(listing_id VARCHAR, item_id VARCHAR, detail_category VARCHAR, contract_no VARCHAR, dlv_key INTEGER, dlv_qty INTEGER, dlv_amount INTEGER)")
    con.execute("INSERT INTO listings VALUES ('L1','I1',?, 'C1',1,2,4000000)", [category])
    return SimpleNamespace(connection=con)


def test_panel_lane_writes_bundle_and_draft(tmp_path: Path) -> None:
    evidence = seller_config()
    out = tmp_path / "out"
    n, avail, drafts = _run_recipe_lane(_Competition(_panel_name()), _Raw(_panel_attrs()), _fetcher(evidence),
                                        evidence, AuditConfig(), out, _dataset("랙캐비닛용패널"), None, 10, False, "panel")
    assert (n, avail, drafts) == (1, 1, 1)
    assert (out / "drafts" / "recipe" / "I1.json").is_file()


def test_rack_lane_writes_bundle_and_draft(tmp_path: Path) -> None:
    evidence = seller_config()
    out = tmp_path / "out"
    n, avail, drafts = _run_recipe_lane(_Competition("허브랙"), _Raw(_rack_attrs()), _fetcher(evidence),
                                        evidence, AuditConfig(), out, _dataset("네트워크시스템장비용랙"), None, 10, False, "rack")
    assert (n, avail, drafts) == (1, 1, 1)


def test_panel_lane_skips_unmapped_and_rack_review_gated(tmp_path: Path) -> None:
    evidence = seller_config()
    _, _, drafts = _run_recipe_lane(_Competition("보호커버"), _Raw({"종류": "보호패널"}), _fetcher(evidence),
                                    evidence, AuditConfig(), tmp_path / "o1", _dataset("랙캐비닛용패널"), None, 10, False, "panel")
    assert drafts == 0
    _, _, gated = _run_recipe_lane(_Competition("허브랙 측면개폐"), _Raw(_rack_attrs()), _fetcher(evidence),
                                   evidence, AuditConfig(), tmp_path / "o2", _dataset("네트워크시스템장비용랙"), None, 10, False, "rack")
    assert gated == 0
    _, _, open_ = _run_recipe_lane(_Competition("허브랙 측면개폐"), _Raw(_rack_attrs()), _fetcher(evidence),
                                   evidence, AuditConfig(), tmp_path / "o3", _dataset("네트워크시스템장비용랙"), None, 10, True, "rack")
    assert open_ == 1


def test_invalid_recipe_kind_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="recipe_kind"):
        _run_recipe_lane(_Competition("x"), _Raw({}), _fetcher(EvidenceConfig()),
                         EvidenceConfig(), AuditConfig(), tmp_path, _dataset("랙캐비닛용패널"), None, 10, False, "bogus")


def test_parser_accepts_panel_and_rack_kinds() -> None:
    parser = _build_parser()
    assert parser.parse_args(["evidence", "--lane", "recipe", "--recipe-kind", "panel"]).recipe_kind == "panel"
    assert parser.parse_args(["evidence", "--lane", "recipe", "--recipe-kind", "rack"]).recipe_kind == "rack"


@pytest.mark.parametrize("kind", ["panel", "rack"])
def test_fixture_lane_preserves_facts_across_contracts(tmp_path: Path, kind: str) -> None:
    from dataclasses import replace

    from src.savetax.workflow import load_bundles

    evidence = seller_config()
    attrs = _panel_attrs() if kind == "panel" else _rack_attrs()
    name = _panel_name() if kind == "panel" else "허브랙"
    category = evidence.panel_category if kind == "panel" else evidence.rack_category
    facts = (_fact(), replace(_fact(), fact_id="f2", listing_id="L2", contract_no="C2", request_no="R2"))
    _, _, drafts = _run_recipe_lane(_Competition(name, facts), _Raw(attrs), _fetcher(evidence),
                                    evidence, AuditConfig(), tmp_path, _dataset(category), None, 10, False, kind)
    assert drafts == 1
    bundle = load_bundles([tmp_path / "recipe-bundles.jsonl"], now=datetime.now(UTC))["I1"]
    assert {line.fact_id for line in bundle.lines} == {"f1", "f2"}
    assert not bundle.rejected


def test_panel_enclosure_selects_enclosure_part() -> None:
    recipe = panel_recipe("패치판넬 함체형 24PORT CAT.6A", {"종류": "패치판넬"}, _evidence())
    assert not isinstance(recipe, str)
    assert recipe.parts[0][0].mount_form == "enclosure"


def test_panel_mount_mismatch_without_matching_part() -> None:
    from src.savetax.config import RecipePart
    from src.savetax.fixturerecipe import _panel_part_for_mount

    rack_only = (RecipePart("panel:99", "패널", ("https://fibermart.co.kr/x",), ("24포트",), "rack"),)
    assert _panel_part_for_mount(rack_only, True) is None
    assert _panel_part_for_mount(rack_only, False) is not None
    custom = (RecipePart("panel:99", "패널", ("https://fibermart.co.kr/x",), ("24포트",), "rack"),)
    assert _panel_part_for_mount(custom, False) == custom[0]
    base = _evidence()
    no_enclosure = tuple(p for p in base.recipe_parts if not (p.key.startswith("panel:") and p.mount_form == "enclosure"))
    assert any(p.key == "panel:24" for p in no_enclosure)
    ev = _evidence(recipe_parts=no_enclosure)
    assert panel_recipe("패치 함체 24PORT", {"종류": "패치판넬"}, ev) == "mount_form_mismatch"
