"""Guard tests for sourced recipe adders."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from src.savetax.competition import Fact, ItemDetail, ItemState
from src.savetax.config import EvidenceConfig, RecipeAdder, RecipePart, SaveTaxConfig
from src.savetax.errors import DomainValidationError, NetworkError, NetworkStage
from src.savetax.evidence import PriceEvidence
from src.savetax.http import HttpResponse
from src.savetax.publicweb import PublicFetcher
from src.savetax.recipe import (
    PricedPart,
    Recipe,
    adder_page_amount,
    build_recipe_bundle,
    price_adders,
    price_parts,
)

_TODAY = date(2026, 10, 13)


def _part(key: str = "cpu:T", label: str = "CPU", pcode: str = "11") -> RecipePart:
    return RecipePart(key, label, pcode, ("울트라5", "225"))


def _sourced_adder() -> RecipeAdder:
    return RecipeAdder("rack:labor", "랙 설치 공임", 167310, "basis", "https://savetax.now/176")


def _recipe(adders: tuple | None = None, **over: Any) -> Recipe:
    kw: dict[str, Any] = {
        "kind": "rack",
        "parts": ((_part(), 1),),
        "adders": ((),) if adders is None else adders,
        "spec_rows": (("폭", "a", "b"),),
        "differences": (),
        "review_required": False,
        "precedent_urls": ("https://savetax.now/176",),
        "title_phrase": "동급 허브랙 판매가와 공식 설치 공임 합계",
    }
    if isinstance(kw["adders"], tuple) and kw["adders"] and isinstance(kw["adders"][0], tuple):
        kw["adders"] = kw["adders"][0]
    kw.update(over)
    return Recipe(**kw)


def _fetcher(routes: dict[str, tuple[int, str]], seen: list[str]) -> PublicFetcher:
    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        seen.append(url)
        for key, (status, body) in routes.items():
            if key in url:
                return HttpResponse(status=status, headers={}, body=body.encode(), elapsed_seconds=0.01)
        return HttpResponse(status=404, headers={}, body=b"", elapsed_seconds=0.01)

    ev = EvidenceConfig(public_request_interval_seconds=0.0)
    return PublicFetcher(SaveTaxConfig(), ev, transport=transport)  # type: ignore[arg-type]


def _evidence(**over: Any) -> EvidenceConfig:
    base: dict[str, Any] = {"min_item_expected_krw": 0, "public_request_interval_seconds": 0.0}
    base.update(over)
    return EvidenceConfig(**base)  # type: ignore[arg-type]


def _fact(fact_id: str = "f1", unit_price: str = "2000", **over: Any) -> Fact:
    kw: dict[str, Any] = {
        "fact_id": fact_id, "listing_id": "L1", "contract_no": "C1", "request_no": "R1",
        "line_no": "1", "request_date": date(2026, 1, 5), "unit": "대", "tax_basis": "tax",
        "basis_id": "B1", "period_kind": "none", "unit_price_krw": Decimal(unit_price),
        "quantity": Decimal("2"), "comparable_amount_krw": Decimal(unit_price) * 2,
        "scorable": True, "exclusion_reasons": (),
    }
    kw.update(over)
    return Fact(**kw)


def _detail(facts: tuple = ()) -> ItemDetail:
    return ItemDetail("I1", ItemState.AVAILABLE, "procured", None, None, None, None, len(facts), False, facts)


def _proof(price: int = 600) -> PriceEvidence:
    return PriceEvidence("https://prod.danawa.com/info/?pcode=11", "인용 " + "x" * 10, price, _TODAY, True, False)


class _Ds:
    pass


def test_adder_amount_found_on_page() -> None:
    text = "앞 내용 랙 설치 공식 공임 167,310원 · 뒷 내용"
    fragment = adder_page_amount(text, 167310)
    assert fragment is not None
    assert "167,310원" in fragment
    assert len(fragment) <= 300


def test_amount_inside_longer_number_ignored() -> None:
    assert adder_page_amount("가격 1,167,310원", 167310) is None


def test_empty_page_returns_none() -> None:
    assert adder_page_amount("   ", 167310) is None


def test_sourced_make_line_failure_gate() -> None:
    priced, recipe, cited = _sourced_bundle()
    bad = _fact("f1", "2000", comparable_amount_krw=Decimal(10))
    bundle = build_recipe_bundle("I1", _detail((bad,)), recipe, priced, _Ds(), _evidence(), today=_TODAY, adder_evidence=cited)  # type: ignore[arg-type]
    assert bundle is None


def test_amount_only_inside_script_ignored() -> None:
    html = "<html><head><script>var x='167,310원';</script></head><body>다른 내용</body></html>"
    assert adder_page_amount(html, 167310) is None


def test_long_window_truncated_to_phrase() -> None:
    text = "서두 " + "가" * 500 + " 랙 설치 공식 공임 167,310원 · " + "나" * 500
    fragment = adder_page_amount(text, 167310)
    assert fragment is not None
    assert "167,310원" in fragment
    assert len(fragment) <= 300
    assert "·" not in fragment


def test_sourced_adder_becomes_evidence() -> None:
    recipe = _recipe(adders=(_sourced_adder(),))
    seen: list[str] = []
    fetcher = _fetcher({"savetax.now/176": (200, "<html><body>랙 설치 공식 공임 167,310원 · 안내</body></html>")}, seen)
    cited = price_adders(recipe, fetcher, _TODAY)
    assert not isinstance(cited, str)
    assert len(cited) == 1
    assert cited[0].observed_price_krw == 167310
    assert cited[0].source_url == "https://savetax.now/176"
    assert cited[0].quote.startswith("랙 설치 공식 공임 167,310원")


def test_unsourced_adder_yields_no_evidence() -> None:
    recipe = _recipe(adders=(RecipeAdder("a:1", "조립비", 30, "basis"),))
    seen: list[str] = []
    cited = price_adders(recipe, _fetcher({}, seen), _TODAY)
    assert cited == ()
    assert seen == []


def test_unreadable_adder_page() -> None:
    recipe = _recipe(adders=(_sourced_adder(),))
    cited = price_adders(recipe, _fetcher({}, []), _TODAY)
    assert cited == "adder_page_unreadable:rack:labor"


def test_missing_amount() -> None:
    recipe = _recipe(adders=(_sourced_adder(),))
    fetcher = _fetcher({"savetax.now/176": (200, "<html><body>다른 공임 안내</body></html>")}, [])
    assert price_adders(recipe, fetcher, _TODAY) == "adder_amount_missing:rack:labor"


def test_empty_body_is_unreadable() -> None:
    recipe = _recipe(adders=(_sourced_adder(),))
    fetcher = _fetcher({"savetax.now/176": (200, "")}, [])
    assert price_adders(recipe, fetcher, _TODAY) == "adder_page_unreadable:rack:labor"


def _sourced_bundle() -> Any:
    priced = (PricedPart(_part(), 1, 600, 600, _proof(600)),)
    adder = RecipeAdder("rack:labor", "랙 설치 공임", 167, "basis", "https://savetax.now/176")
    recipe = _recipe(adders=(adder,), floor_krw=900)
    cited = (
        PriceEvidence("https://savetax.now/176", "[가산액 근거: 랙 설치 공임] 랙 설치 공식 공임 167원 / savetax.now 상세분석 화면, 2026-10-13 확인", 167, _TODAY, True, False, "랙 설치 공임 167원 반영"),
    )
    return priced, recipe, cited


def test_fully_sourced_benchmark_equals_observed_sum() -> None:
    priced, recipe, cited = _sourced_bundle()
    bundle = build_recipe_bundle("I1", _detail((_fact("f1", "2000"),)), recipe, priced, _Ds(), _evidence(), today=_TODAY, adder_evidence=cited)  # type: ignore[arg-type]
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal(767)
    assert len(bundle.lines[0].price_evidence) == 2
    assert sum(e.observed_price_krw for e in bundle.lines[0].price_evidence) == 767


def test_floor_acts_as_gate_only() -> None:
    priced, recipe, cited = _sourced_bundle()
    ev = _evidence(recipe_markup_threshold=Decimal("1.2"))
    bundle = build_recipe_bundle("I1", _detail((_fact("f1", "1000"), _fact("f2", "1100", request_no="R2", line_no="2"))), recipe, priced, _Ds(), ev, today=_TODAY, adder_evidence=cited)  # type: ignore[arg-type]
    assert bundle is not None
    assert [line.fact_id for line in bundle.lines] == ["f2"]
    assert bundle.rejected[0].reason == "below_floor_gate"
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal(767)


def test_sum_written_in_note_not_quote() -> None:
    priced, recipe, cited = _sourced_bundle()
    bundle = build_recipe_bundle("I1", _detail((_fact("f1", "2000"),)), recipe, priced, _Ds(), _evidence(), today=_TODAY, adder_evidence=cited)  # type: ignore[arg-type]
    assert bundle is not None
    part, adder = bundle.lines[0].price_evidence
    assert part.quote == priced[0].evidence.quote
    assert adder.quote == cited[0].quote
    for cited_ev in (part, adder):
        assert cited_ev.note is not None
        assert "767원(= " in cited_ev.note
        assert "CPU" in cited_ev.note
        assert "랙 설치 공임" in cited_ev.note
    assert adder.note is not None
    assert "표준품셈" in adder.note


def test_claim_omits_floor_raise() -> None:
    priced, recipe, cited = _sourced_bundle()
    bundle = build_recipe_bundle("I1", _detail((_fact("f1", "2000"),)), recipe, priced, _Ds(), _evidence(), today=_TODAY, adder_evidence=cited)  # type: ignore[arg-type]
    assert bundle is not None
    assert "= 767원" in bundle.claim
    assert "올림" not in bundle.claim


def test_gate_filtered_claim_mentions_threshold() -> None:
    priced, recipe, cited = _sourced_bundle()
    ev = _evidence(recipe_markup_threshold=Decimal("1.2"))
    bundle = build_recipe_bundle("I1", _detail((_fact("f1", "1000"), _fact("f2", "2000", request_no="R2", line_no="2"))), recipe, priced, _Ds(), ev, today=_TODAY, adder_evidence=cited)  # type: ignore[arg-type]
    assert bundle is not None
    assert "1.2배 미만인 납품요구는 넣지 않았다" in bundle.claim


def test_partially_sourced_recipe_unchanged() -> None:
    priced = (PricedPart(_part(), 1, 600, 600, _proof(600)),)
    adders = (
        RecipeAdder("a:1", "가산1", 167, "basis", "https://savetax.now/176"),
        RecipeAdder("a:2", "가산2", 100, "basis"),
    )
    recipe = _recipe(adders=adders, floor_krw=900)
    cited = (
        PriceEvidence("https://savetax.now/176", "[가산액 근거: 가산1] 공임 167원 / savetax.now 상세분석 화면, 2026-10-13 확인", 167, _TODAY, True, False),
    )
    bundle = build_recipe_bundle("I1", _detail((_fact("f1", "2000"),)), recipe, priced, _Ds(), _evidence(), today=_TODAY, adder_evidence=cited)  # type: ignore[arg-type]
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == Decimal(900)
    assert "올림" in bundle.claim


def test_sourced_mismatch_raises() -> None:
    priced = (PricedPart(_part(), 1, 610, 600, _proof(600)),)
    adder = RecipeAdder("rack:labor", "랙 설치 공임", 167, "basis", "https://savetax.now/176")
    recipe = _recipe(adders=(adder,), floor_krw=900)
    cited = (
        PriceEvidence("https://savetax.now/176", "[가산액 근거: 랙 설치 공임] 공임 167원 / savetax.now 상세분석 화면, 2026-10-13 확인", 167, _TODAY, True, False),
    )
    with pytest.raises(DomainValidationError, match="sourced recipe prices differ"):
        build_recipe_bundle("I1", _detail((_fact("f1", "2000"),)), recipe, priced, _Ds(), _evidence(), today=_TODAY, adder_evidence=cited)  # type: ignore[arg-type]


def test_sourced_eligible_part_unstable() -> None:
    html = (
        "<html><head><script type=\"application/ld+json\">"
        '{"@type": "Product", "name": "울트라5 225 정품", "brand": {"name": "B"}, '
        '"description": "d", "offers": {"lowPrice": 100, "offerCount": 3}}'
        "</script></head><body><ul class=\"list__mall-price\">"
        '<li><span aria-label="몰A"></span><span class="text__num">100</span></li>'
        '<li><span aria-label="몰B"></span><span class="text__num">130</span></li>'
        '<li><span aria-label="몰C"></span><span class="text__num">131</span></li>'
        "</ul></body></html>"
    )
    recipe = _recipe(parts=((_part(), 1),), adders=(_sourced_adder(),))
    out = price_parts(recipe, _fetcher({"pcode=11": (200, html)}, []), _evidence(recipe_min_offers=1), _TODAY, {})
    assert out == "part_price_unstable:cpu:T"


@pytest.mark.parametrize("text", ["167,310원2", "167,310원,", "2167,310원", ",167,310원"])
def test_amount_boundaries_reject_embedded_figures(text: str) -> None:
    assert adder_page_amount(text, 167310) is None


def test_first_valid_amount_fragment_is_visible_and_normalized() -> None:
    text = "1,167,310원 · <b>랙 설치</b>\n 공식 공임 167,310원 · 다음 167,310원"
    assert adder_page_amount(text, 167310) == "랙 설치 공식 공임 167,310원"


@pytest.mark.parametrize(("quantity", "unit", "observed", "proof"), [
    (2, 600, 600, 600), (1, 600, 600, 599), (1, 600, 599, 600),
    (0, 600, 600, 600), (True, 600, 600, 600), (1, 0, 600, 600),
])
def test_sourced_parts_validate_actual_evidence_and_quantities(quantity, unit, observed, proof) -> None:
    priced, recipe, cited = _sourced_bundle()
    invalid = replace(priced[0], quantity=quantity, unit_price_krw=unit,
                      observed_price_krw=observed, evidence=_proof(proof))
    with pytest.raises(DomainValidationError):
        build_recipe_bundle("I1", _detail((_fact(),)), recipe, (invalid,), _Ds(), _evidence(),
                            today=_TODAY, adder_evidence=cited)  # type: ignore[arg-type]


@pytest.mark.parametrize("floor", [0, -1, True])
def test_sourced_recipe_rejects_invalid_floor(floor) -> None:
    priced, recipe, cited = _sourced_bundle()
    with pytest.raises(DomainValidationError):
        build_recipe_bundle("I1", _detail((_fact(),)), replace(recipe, floor_krw=floor),
                            priced, _Ds(), _evidence(), today=_TODAY, adder_evidence=cited)  # type: ignore[arg-type]


def test_unrelated_adder_citation_does_not_remove_floor() -> None:
    priced, recipe, cited = _sourced_bundle()
    unrelated = (replace(cited[0], source_url="https://savetax.now/222"),)
    bundle = build_recipe_bundle("I1", _detail((_fact(),)), recipe, priced, _Ds(), _evidence(),
                                 today=_TODAY, adder_evidence=unrelated)  # type: ignore[arg-type]
    assert bundle is not None
    assert bundle.lines[0].benchmark_unit_price_krw == 900


def test_floor_gate_includes_exact_boundary_and_does_not_mislabel_invalid_lines() -> None:
    priced, recipe, cited = _sourced_bundle()
    facts = (_fact("bad", "2000", comparable_amount_krw=Decimal(10)),
             _fact("boundary", "1080"))
    bundle = build_recipe_bundle("I1", _detail(facts), recipe, priced, _Ds(), _evidence(),
                                 today=_TODAY, adder_evidence=cited)  # type: ignore[arg-type]
    assert bundle is not None
    assert [line.fact_id for line in bundle.lines] == ["boundary"]
    assert bundle.rejected[0].reason == "below_threshold"
    assert "미만인 납품요구는 넣지 않았다" not in bundle.claim


def test_adder_network_error_propagates_unchanged() -> None:
    error = NetworkError(NetworkStage.DNS_RESOLUTION, "savetax.now", "test")

    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        raise error

    fetcher = PublicFetcher(SaveTaxConfig(), EvidenceConfig(), transport=transport, sleep=lambda _: None)  # type: ignore[arg-type]
    with pytest.raises(NetworkError) as caught:
        price_adders(_recipe(adders=(_sourced_adder(),)), fetcher, _TODAY)
    assert caught.value is error


def test_adder_phrase_prefers_the_published_standard_mention() -> None:
    text = (
        "기준 · 파이버마트 LD-H1800-NEW 600,000원 + 랙 설치 공식 공임 167,310원 · LD-H1800-NEW 600,000원(판매 중) + "
        "정보통신 표준품셈 4-3-3 랙 설치 공임 167,310원(VAT 포함) · 2026.10.7 조사"
    )
    assert adder_page_amount(text, 167310) == "정보통신 표준품셈 4-3-3 랙 설치 공임 167,310원(VAT 포함)"


def test_adder_phrase_drops_neighbouring_product_price() -> None:
    text = "파이버마트 LD-H1800-NEW 600,000원 + 랙 설치 공식 공임 167,310원 767,310원"
    assert adder_page_amount(text, 167310) == "랙 설치 공식 공임 167,310원"


def test_adder_phrase_keeps_amount_when_parenthesis_is_unclosed() -> None:
    assert adder_page_amount("공임 167,310원(VAT", 167310) == "공임 167,310원"
