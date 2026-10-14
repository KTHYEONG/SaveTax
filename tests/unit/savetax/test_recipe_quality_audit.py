"""Regression guards for seller evidence integrity and blocked source coverage."""

from dataclasses import replace
from datetime import date

import pytest

from src.savetax.cli import _live_price
from src.savetax.config import EvidenceConfig, RecipeAdder, RecipePart, SaveTaxConfig
from src.savetax.errors import ConfigError, DomainValidationError, NetworkError, NetworkStage
from src.savetax.http import HttpResponse
from src.savetax.publicweb import PublicFetcher
from src.savetax.recipe import Recipe, _second_benchmark, build_recipe_bundle, price_parts
from src.savetax.shops import ShopProduct
from tests.unit.savetax.test_recipe import _Ds, _detail, _fact, _priced, _seller_recipe


def test_default_sources_are_real_paths_or_explicitly_pending() -> None:
    config = EvidenceConfig()
    assert any(p.sources_pending for p in config.recipe_parts)
    for part in config.recipe_parts:
        if part.sources_pending:
            assert not part.sources
            assert not part.pcode
        else:
            assert part.sources
            assert all("/product/cpu-" not in u and "-a" not in u.rsplit("/", 1)[-1] for u in part.sources)
    pending = next(p for p in config.recipe_parts if p.sources_pending)
    class Fetcher:
        def get(self, url: str) -> object:
            raise AssertionError("Pending parts must not trigger network requests")
    assert price_parts(_seller_recipe(pending), Fetcher(), config, date.today(), {}) == f"part_sources_pending:{pending.key}"  # type: ignore[arg-type]


def test_no_sold_out_fallback_in_live_reverification() -> None:
    html = '<title>CPU</title><meta property="product:price:amount" content="88000"><p>품절</p>'
    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        return HttpResponse(200, {}, html.encode(), 0.01)
    fetcher = PublicFetcher(SaveTaxConfig(), EvidenceConfig(), transport=transport)
    assert _live_price(fetcher, "https://compuzone.co.kr/product/x") is None


def test_name_mismatch_does_not_poison_url_cache() -> None:
    config = EvidenceConfig()
    url = "https://compuzone.co.kr/product/x"
    correct = RecipePart("correct", "CPU", (url,), ("225",))
    wrong = replace(correct, key="wrong", name_tokens=("265",))
    cache: dict[str, object] = {url: ShopProduct(url, "CPU 225", 1000)}
    assert price_parts(_seller_recipe(wrong), object(), config, date.today(), cache) == "part_name_mismatch:wrong"  # type: ignore[arg-type]
    result = price_parts(_seller_recipe(correct), object(), config, date.today(), cache)  # type: ignore[arg-type]
    assert not isinstance(result, str)
    assert result[0].unit_price_krw == 1000


def test_network_error_preserves_diagnostic_stage() -> None:
    config = EvidenceConfig()
    part = next(p for p in config.recipe_parts if p.sources)
    class Fetcher:
        def get(self, url: str) -> object:
            raise NetworkError(NetworkStage.DNS_RESOLUTION, "compuzone.co.kr", "unavailable")
    with pytest.raises(NetworkError):
        price_parts(_seller_recipe(part), Fetcher(), config, date.today(), {})  # type: ignore[arg-type]


def test_second_price_requires_corresponding_independent_evidence() -> None:
    base = _priced()
    assert _second_benchmark((replace(base, second_unit_price_krw=104),), (), None) is None
    second = replace(base.evidence, source_url="https://www.prod.danawa.com/info/?pcode=22", observed_price_krw=104)
    assert _second_benchmark((replace(base, second_unit_price_krw=104, second_evidence=second),), (), None) is None
    independent = replace(second, source_url="https://cynexpc.com/shop/product_detail.html?pd_no=1")
    assert _second_benchmark((replace(base, second_unit_price_krw=104, second_evidence=independent),), (), 200) == 200


def test_seller_recipe_requires_cited_adders_and_matching_prices() -> None:
    base = _priced()
    part = RecipePart("cpu:T", "CPU", (base.evidence.source_url,), ("CPU",))
    priced = replace(base, part=part)
    recipe = _seller_recipe(part)
    config = EvidenceConfig(min_item_expected_krw=0)
    detail = _detail((_fact(unit_price="5000"),))
    with pytest.raises(DomainValidationError, match="adders require"):
        build_recipe_bundle("I1", detail, replace(recipe, adders=(RecipeAdder("assembly", "조립", 25, "basis"),)), (priced,), _Ds(), config, today=date.today())  # type: ignore[arg-type]
    with pytest.raises(DomainValidationError, match="prices differ"):
        build_recipe_bundle("I1", detail, recipe, (replace(priced, unit_price_krw=101),), _Ds(), config, today=date.today())  # type: ignore[arg-type]


def test_ninth_seller_part_stays_in_b_without_citation() -> None:
    base = _priced()
    parts = tuple(replace(base, part=RecipePart(f"k{i}", f"부품{i}", (base.evidence.source_url,), ("CPU",))) for i in range(9))
    small = replace(base.evidence, observed_price_krw=10)
    parts = (*parts[:8], replace(parts[8], unit_price_krw=10, observed_price_krw=10, evidence=small))
    recipe = Recipe("desktop", tuple((p.part, 1) for p in parts), (), (), (), False, (), "CPU")
    bundle = build_recipe_bundle("I1", _detail((_fact(unit_price="5000"),)), recipe, parts, _Ds(), EvidenceConfig(min_item_expected_krw=0), today=date.today())  # type: ignore[arg-type]
    assert bundle is not None
    assert len(bundle.lines[0].price_evidence) == 8
    assert bundle.lines[0].benchmark_unit_price_krw == 810
    assert "부품8 10원은 가격 근거 개수 한도(8개)" in bundle.claim


def test_evidence_hosts_are_bare_names() -> None:
    with pytest.raises(ConfigError, match="bare hostname"):
        EvidenceConfig(evidence_hosts=("compuzone.co.kr:443",))


def test_desktop_reader_uses_seller_page_part() -> None:
    from src.savetax.desktoprecipe import desktop_recipe
    from tests.unit.savetax.test_desktoprecipe import _attrs
    recipe = desktop_recipe(_attrs(**{"멀티메모리카드리더유무": "유"}), EvidenceConfig())
    assert not isinstance(recipe, str)
    assert "reader" in [part.key for part, _ in recipe.parts]


@pytest.mark.parametrize(("psu", "form", "expected"), [
    ("450W", "슬림", ("psu:450W-SFX", "case:slim-sfx")),
    ("400W", "슬림", ("psu:400W", "case:slim")),
    ("350W", "타워", ("psu:500W", "case:tower")),
])
def test_desktop_psu_maps_to_same_or_larger_supply(psu: str, form: str, expected: tuple[str, str]) -> None:
    from src.savetax.desktoprecipe import desktop_recipe
    from tests.unit.savetax.test_desktoprecipe import _attrs
    recipe = desktop_recipe(_attrs(**{"스위칭모드파워서플라이형식": psu, "컴퓨터형식": form}), EvidenceConfig())
    assert not isinstance(recipe, str)
    keys = [part.key for part, _ in recipe.parts]
    assert expected[0] in keys
    assert expected[1] in keys


@pytest.mark.parametrize(("psu", "form"), [("600W", "타워"), ("500W", "슬림"), ("정격", "타워")])
def test_desktop_psu_without_reference_is_unmapped(psu: str, form: str) -> None:
    from src.savetax.desktoprecipe import desktop_recipe
    from tests.unit.savetax.test_desktoprecipe import _attrs
    assert desktop_recipe(_attrs(**{"스위칭모드파워서플라이형식": psu, "컴퓨터형식": form}), EvidenceConfig()) == "psu_unmapped"


def test_adder_past_citation_limit_stays_in_b_and_is_disclosed() -> None:
    from src.savetax.evidence import PriceEvidence

    base = _priced()
    parts = tuple(replace(base, part=RecipePart(f"k{i}", "CPU", (base.evidence.source_url,), ("CPU",))) for i in range(8))
    adder = RecipeAdder("desktop:assembly", "조립비", 25, "basis", "https://savetax.now/520")
    recipe = Recipe("desktop", tuple((p.part, 1) for p in parts), (adder,), (), (), False, (), "CPU")
    cited = (PriceEvidence("https://savetax.now/520", "조립 25원", 25, date.today(), True, False),)
    bundle = build_recipe_bundle("I1", _detail((_fact(unit_price="5000"),)), recipe, parts, _Ds(), EvidenceConfig(min_item_expected_krw=0), today=date.today(), adder_evidence=cited)  # type: ignore[arg-type]
    assert bundle is not None
    assert len(bundle.lines[0].price_evidence) == 8
    assert all(e.source_url != "https://savetax.now/520" for e in bundle.lines[0].price_evidence)
    assert bundle.lines[0].benchmark_unit_price_krw == 825
    assert "조립비 25원은 가격 근거 개수 한도(8개)" in bundle.claim
    assert "https://savetax.now/520" in bundle.claim


def test_two_memory_sticks_count_twice_in_sourced_benchmark() -> None:
    from src.savetax.evidence import PriceEvidence

    base = _priced()
    ram = replace(base, part=RecipePart("ram:16", "메모리 16GB", (base.evidence.source_url,), ("CPU",)), quantity=2)
    adder = RecipeAdder("desktop:assembly", "조립비", 25, "basis", "https://savetax.now/520")
    recipe = Recipe("desktop", ((ram.part, 2),), (adder,), (), (), False, (), "CPU")
    cited = (PriceEvidence("https://savetax.now/520", "조립 25원", 25, date.today(), True, False),)
    bundle = build_recipe_bundle("I1", _detail((_fact(unit_price="5000"),)), recipe, (ram,), _Ds(), EvidenceConfig(min_item_expected_krw=0), today=date.today(), adder_evidence=cited)  # type: ignore[arg-type]
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == 225
    note = bundle.lines[0].price_evidence[0].note
    assert note is not None
    assert "메모리 16GB 100원 × 2" in note  # noqa: RUF001
