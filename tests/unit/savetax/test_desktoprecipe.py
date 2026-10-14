"""Invariant guards for the desktop parts-sum mapping."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest

from src.savetax.config import EvidenceConfig
from src.savetax.desktoprecipe import delivery_terms_ok, desktop_recipe, find_recipe_candidates
from src.savetax.rawapi import ListingDetail
from datetime import date


def _evidence(**over: Any) -> EvidenceConfig:
    return EvidenceConfig(**over)  # type: ignore[arg-type]


def _attrs(**over: str) -> dict[str, str]:
    base = {
        "중앙처리장치(CPU)규격": "Intel Core Ultra 5 225",
        "메모리(RAM)용량": "16",
        "SSD용량": "512",
        "하드디스크(HDD)용량": "2",
        "메인보드": "H810",
        "스위칭모드파워서플라이형식": "500W",
        "컴퓨터형식": "타워",
        "그래픽": "내장그래픽",
        "운영체계": "옵션제공",
        "광디스크(ODD)규격": "옵션제공",
    }
    base.update(over)
    return base


def test_full_set_maps_to_eight_parts() -> None:
    recipe = desktop_recipe(_attrs(), _evidence())
    assert not isinstance(recipe, str)
    assert len(recipe.parts) == 8
    assert [p.key for p, _ in recipe.parts] == [
        "cpu:ULTRA 5 225", "ram:16", "ssd:500", "hdd:2", "mb:H810", "psu:500W", "case:tower", "kbm",
    ]
    assert len(recipe.adders) == 1
    assert recipe.review_required is False


def test_32gb_uses_two_modules_and_missing_hdd_rejected() -> None:
    recipe = desktop_recipe(_attrs(**{"메모리(RAM)용량": "32"}), _evidence())
    assert not isinstance(recipe, str)
    assert ((recipe.parts[1][0].key, recipe.parts[1][1])) == ("ram:16", 2)
    attrs = _attrs()
    del attrs["하드디스크(HDD)용량"]
    assert desktop_recipe(attrs, _evidence()) == "hdd_unknown"


def test_hdd_not_installed_adds_no_part() -> None:
    recipe = desktop_recipe(_attrs(**{"하드디스크(HDD)용량": "미장착"}), _evidence())
    assert not isinstance(recipe, str)
    assert len(recipe.parts) == 7


def test_cpu_variants_normalise() -> None:
    for raw in ("IntelCore i5-14400", "Intel® Core i5 14400", "i5-14400(2.5GHz)"):
        recipe = desktop_recipe(_attrs(**{"중앙처리장치(CPU)규격": raw}), _evidence())
        assert not isinstance(recipe, str)
        assert recipe.parts[0][0].key == "cpu:I5 14400"


def test_i5_13500_uses_successor_and_needs_review() -> None:
    recipe = desktop_recipe(_attrs(**{"중앙처리장치(CPU)규격": "i5-13500"}), _evidence())
    assert not isinstance(recipe, str)
    assert recipe.parts[0][0].key == "cpu:I5 14500"
    assert recipe.review_required is True
    assert recipe.differences


def test_unknown_cpu_and_discrete_graphics_rejected() -> None:
    assert desktop_recipe(_attrs(**{"중앙처리장치(CPU)규격": "AMD Ryzen7 PRO 7745"}), _evidence()) == "cpu_unmapped"
    assert desktop_recipe(_attrs(**{"그래픽": "RTX 5060 D7 8GB"}), _evidence()) == "discrete_graphics"


def test_integrated_graphics_spellings_pass() -> None:
    for raw in ("Intel UHD Graphics 770", "인텔 UHD 그래픽 770"):
        assert not isinstance(desktop_recipe(_attrs(**{"그래픽": raw}), _evidence()), str)
    attrs = _attrs()
    del attrs["그래픽"]
    assert not isinstance(desktop_recipe(attrs, _evidence()), str)


def test_bundled_os_and_odd_rejected() -> None:
    assert desktop_recipe(_attrs(**{"운영체계": "Windows 11"}), _evidence()) == "os_included"
    attrs = _attrs()
    del attrs["운영체계"]
    assert desktop_recipe(attrs, _evidence()) == "os_included"
    assert desktop_recipe(_attrs(**{"광디스크(ODD)규격": "DVD-RW"}), _evidence()) == "odd_included"


def test_missing_power_follows_form() -> None:
    attrs = _attrs(**{"컴퓨터형식": "슬림"})
    del attrs["스위칭모드파워서플라이형식"]
    recipe = desktop_recipe(attrs, _evidence())
    assert not isinstance(recipe, str)
    assert recipe.parts[5][0].key == "psu:400W"
    assert recipe.review_required is True


def test_unknown_specs_rejected() -> None:
    assert desktop_recipe(_attrs(**{"메인보드": "Q670"}), _evidence()) == "mainboard_unmapped"
    assert desktop_recipe(_attrs(**{"컴퓨터형식": "빅타워"}), _evidence()) == "form_unmapped"
    assert desktop_recipe(_attrs(**{"SSD용량": "1"}), _evidence()) == "ssd_unmapped"
    assert desktop_recipe(_attrs(**{"메모리(RAM)용량": "8"}), _evidence()) == "ram_unmapped"
    assert desktop_recipe(_attrs(**{"스위칭모드파워서플라이형식": "600W"}), _evidence()) == "psu_unmapped"
    assert desktop_recipe(_attrs(**{"하드디스크(HDD)용량": "4"}), _evidence()) == "hdd_unmapped"


def _mem_dataset() -> Any:
    con = duckdb.connect()
    con.execute("CREATE TABLE listings(listing_id VARCHAR, item_id VARCHAR, detail_category VARCHAR, contract_no VARCHAR, dlv_key INTEGER, dlv_qty INTEGER, dlv_amount INTEGER, delivery_terms VARCHAR)")
    con.execute("INSERT INTO listings VALUES ('L1','A','데스크톱컴퓨터','C1',1,1,100,'납품장소도'),('L2','A','데스크톱컴퓨터','C1',1,1,100,'납품장소도'),('L3','B','데스크톱컴퓨터','C2',2,1,50,'납품장소도')")
    return SimpleNamespace(connection=con)


def test_candidates_ordered_once_per_key() -> None:
    ds = _mem_dataset()
    assert find_recipe_candidates(ds, "데스크톱컴퓨터") == ["A", "B"]
    assert find_recipe_candidates(ds, "데스크톱컴퓨터", items=["B"]) == ["B"]
    assert find_recipe_candidates(ds, "데스크톱컴퓨터", items=[]) == []


def _listing(terms: str | None) -> ListingDetail:
    return ListingDetail("L1", "A", "대", Decimal(100), "contract", "C1", None, None, None, "v", None, None, None, terms, None, None)


class _Raw:
    def __init__(self, mapping: dict) -> None:
        self.mapping = mapping

    def get_listing(self, lid: str) -> ListingDetail | None:
        return self.mapping.get(lid)


def test_installation_terms_fail() -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE listings(listing_id VARCHAR, item_id VARCHAR, contract_no VARCHAR)")
    con.execute("INSERT INTO listings VALUES ('L1','A','C1')")
    ds = SimpleNamespace(connection=con)
    assert delivery_terms_ok(ds, _Raw({"L1": _listing("현장설치도")}), "A", "C1", ("납품장소도",), 12) is False
    assert delivery_terms_ok(ds, _Raw({"L1": None}), "A", "C1", ("납품장소도",), 12) is False
    assert delivery_terms_ok(ds, _Raw({"L1": _listing("납품장소도")}), "A", "C1", ("납품장소도",), 12) is True


def test_terms_limit_blocks() -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE listings(listing_id VARCHAR, item_id VARCHAR, contract_no VARCHAR)")
    con.execute("INSERT INTO listings VALUES ('L1','A','C1'),('L2','A','C1')")
    ds = SimpleNamespace(connection=con)
    assert delivery_terms_ok(ds, _Raw({"L1": _listing("납품장소도"), "L2": _listing("납품장소도")}), "A", "C1", ("납품장소도",), 1) is False


def test_form_and_ssd_variants() -> None:
    assert not isinstance(desktop_recipe(_attrs(**{"컴퓨터형식": "미들타워형", "SSD용량": "500~512", "메모리(RAM)용량": "DDR5 16GB", "메인보드": "Intel H610"}), _evidence()), str)
    assert desktop_recipe(_attrs(**{"하드디스크(HDD)용량": "2TB"}), _evidence()) != "hdd_unmapped"
    assert date.today() is not None


def test_by_key_unknown_raises_and_400w_maps() -> None:
    from src.savetax.errors import ConfigError

    with pytest.raises(ConfigError, match="recipe_parts"):
        desktop_recipe(_attrs(), _evidence(recipe_parts=()))
    recipe = desktop_recipe(_attrs(**{"스위칭모드파워서플라이형식": "400W", "컴퓨터형식": "슬림"}), _evidence())
    assert not isinstance(recipe, str)
    assert recipe.parts[5][0].key == "psu:400W"


def test_candidate_bad_amount_skipped() -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE listings(listing_id VARCHAR, item_id VARCHAR, detail_category VARCHAR, contract_no VARCHAR, dlv_key INTEGER, dlv_qty INTEGER, dlv_amount VARCHAR)")
    con.execute("INSERT INTO listings VALUES ('L1','A','데스크톱컴퓨터','C1',1,1,'bad')")
    from types import SimpleNamespace as _NS

    assert find_recipe_candidates(_NS(connection=con), "데스크톱컴퓨터") == []


@pytest.mark.parametrize(("name", "value", "reason"), [
    ("메인보드", "H610M", "mainboard_unmapped"),
    ("메인보드", "B860 / Z890", "mainboard_unmapped"),
    ("메모리(RAM)용량", "16DDR5", "ram_unmapped"),
    ("메모리(RAM)용량", "GB16", "ram_unmapped"),
    ("SSD용량", "500-500", "ssd_unmapped"),
    ("SSD용량", "GB512", "ssd_unmapped"),
    ("스위칭모드파워서플라이형식", "", "psu_unmapped"),
])
def test_ambiguous_and_malformed_core_specs_fail_closed(name, value, reason) -> None:
    assert desktop_recipe(_attrs(**{name: value}), _evidence()) == reason


def test_only_desktop_adders_selected_and_spec_rows_use_labels() -> None:
    from src.savetax.config import RecipeAdder
    from src.savetax.errors import ConfigError

    ev = _evidence()
    recipe = desktop_recipe(_attrs(), _evidence(recipe_adders=(*ev.recipe_adders, RecipeAdder("other", "공임", 100, "basis"))))
    assert not isinstance(recipe, str)
    assert [a.key for a in recipe.adders] == ["desktop:assembly"]
    assert all(a in ev.recipe_adders for a in recipe.adders)
    assert recipe.spec_rows[0][2] == "CPU"
    with pytest.raises(ConfigError, match="desktop adders missing"):
        desktop_recipe(_attrs(), _evidence(recipe_adders=()))


def test_delivery_terms_require_listings_and_matching_identity() -> None:
    from dataclasses import replace

    ds = _mem_dataset()
    assert not delivery_terms_ok(ds, _Raw({}), "missing", "C1", ("납품장소도",), 12)
    wrong = replace(_listing("납품장소도"), item_id="B")
    assert not delivery_terms_ok(ds, _Raw({"L1": wrong}), "A", "C1", ("납품장소도",), 12)


def _pc_attrs(**over: str) -> dict[str, str]:
    base = {
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
    base.update(over)
    return base


def test_pc_matching_maps_to_finished_pc() -> None:
    from src.savetax.desktoprecipe import desktop_pc_recipe

    recipe = desktop_pc_recipe(_pc_attrs(), _evidence())
    assert not isinstance(recipe, str)
    assert [p.key for p, _ in recipe.parts] == ["pc:ULTRA 5 225", "kbm"]
    assert recipe.adders == ()
    assert recipe.exact_sum is True
    assert recipe.sum_label == "판매가 합계"


def test_pc_hdd_and_reader_add_paged_parts() -> None:
    from src.savetax.desktoprecipe import desktop_pc_recipe

    recipe = desktop_pc_recipe(_pc_attrs(**{"하드디스크(HDD)용량": "2", "멀티메모리카드리더유무": "유"}), _evidence())
    assert not isinstance(recipe, str)
    assert [p.key for p, _ in recipe.parts] == ["pc:ULTRA 5 225", "hdd:2", "kbm", "reader"]


def test_pc_cpu_unmapped_and_board_above() -> None:
    from src.savetax.desktoprecipe import desktop_pc_recipe

    assert desktop_pc_recipe(_pc_attrs(**{"중앙처리장치(CPU)규격": "i5-14400"}), _evidence()) == "cpu_unmapped"
    assert desktop_pc_recipe(_pc_attrs(**{"중앙처리장치(CPU)규격": "Ultra 5 225", "메인보드": "B860"}), _evidence()) == "mainboard_above_reference"


def test_pc_lower_board_accepted_with_difference() -> None:
    from src.savetax.desktoprecipe import desktop_pc_recipe

    recipe = desktop_pc_recipe(_pc_attrs(**{
        "중앙처리장치(CPU)규격": "Ultra 7 265", "메인보드": "H810",
    }), _evidence())
    assert not isinstance(recipe, str)
    assert any("메인보드 등급은 기준 제품이 같거나 높다" in d for d in recipe.differences)


def test_pc_ram_ssd_psu_hdd_gates() -> None:
    from src.savetax.desktoprecipe import desktop_pc_recipe

    assert desktop_pc_recipe(_pc_attrs(**{"메모리(RAM)용량": "32"}), _evidence()) == "ram_unmapped"
    assert desktop_pc_recipe(_pc_attrs(**{"SSD용량": "1TB"}), _evidence()) == "ssd_unmapped"
    assert desktop_pc_recipe(_pc_attrs(**{"스위칭모드파워서플라이형식": "700W"}), _evidence()) == "psu_above_reference"
    assert desktop_pc_recipe(_pc_attrs(**{"스위칭모드파워서플라이형식": "400W"}), _evidence()) != "psu_above_reference"
    attrs = _pc_attrs()
    del attrs["스위칭모드파워서플라이형식"]
    assert not isinstance(desktop_pc_recipe(attrs, _evidence()), str)
    attrs2 = _pc_attrs()
    del attrs2["하드디스크(HDD)용량"]
    assert desktop_pc_recipe(attrs2, _evidence()) == "hdd_unknown"


def test_pc_slim_form_and_rejections() -> None:
    from src.savetax.desktoprecipe import desktop_pc_recipe

    recipe = desktop_pc_recipe(_pc_attrs(**{"컴퓨터형식": "슬림"}), _evidence())
    assert not isinstance(recipe, str)
    assert any("형식이 다르다" in d for d in recipe.differences)
    assert desktop_pc_recipe(_pc_attrs(**{"그래픽": "RTX 5060"}), _evidence()) == "discrete_graphics"
    assert desktop_pc_recipe(_pc_attrs(**{"운영체계": "Windows 11"}), _evidence()) == "os_included"
    assert desktop_pc_recipe(_pc_attrs(**{"광디스크(ODD)규격": "DVD-RW"}), _evidence()) == "odd_included"


def test_pc_legacy_cpu_and_bad_specs_fail_closed() -> None:
    from src.savetax.desktoprecipe import desktop_pc_recipe

    assert desktop_pc_recipe(_pc_attrs(**{"중앙처리장치(CPU)규격": "i5-13500"}), _evidence()) == "cpu_unmapped"
    assert desktop_pc_recipe(_pc_attrs(**{"하드디스크(HDD)용량": "4TB"}), _evidence()) == "hdd_unmapped"
    assert desktop_pc_recipe(_pc_attrs(**{"메인보드": "Q670"}), _evidence()) == "mainboard_unmapped"
    assert desktop_pc_recipe(_pc_attrs(**{"컴퓨터형식": "빅타워"}), _evidence()) == "form_unmapped"
    assert desktop_pc_recipe(_pc_attrs(**{"스위칭모드파워서플라이형식": "고출력"}), _evidence()) == "psu_above_reference"


def test_pc_unknown_rank_chipset_rejected() -> None:
    from src.savetax.desktoprecipe import desktop_pc_recipe

    assert desktop_pc_recipe(_pc_attrs(**{"메인보드": "H610"}), _evidence()) == "mainboard_unmapped"


@pytest.mark.parametrize("power", ["1500.5W", "500W/700W", "-500W", "700W 500W", "9" * 5000 + "W"])
def test_pc_ambiguous_power_is_rejected(power: str) -> None:
    from src.savetax.desktoprecipe import desktop_pc_recipe

    assert desktop_pc_recipe(_pc_attrs(**{"스위칭모드파워서플라이형식": power}), _evidence()) == "psu_above_reference"
