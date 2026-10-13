"""Invariant guards for the parts-sum recipe lane."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from src.savetax.competition import Fact, ItemDetail, ItemState
from src.savetax.config import EvidenceConfig, RecipeAdder, RecipePart, SaveTaxConfig
from src.savetax.danawa import DanawaProduct, MallOffer
from src.savetax.evidence import Lane
from src.savetax.errors import DataIntegrityError, DomainValidationError
from src.savetax.http import HttpResponse
from src.savetax.publicweb import PublicFetcher
from src.savetax.recipe import PricedPart, Recipe, build_recipe_bundle, price_parts, recipe_benchmark
from src.savetax.workflow import load_bundles
from src.savetax.evidence import write_bundles

_TODAY = date(2026, 10, 13)


def _part(key: str = "cpu:T", label: str = "CPU", pcode: str = "11", tokens: tuple[str, ...] = ("울트라5", "225")) -> RecipePart:
    return RecipePart(key, label, pcode, tokens)


def _adder(key: str = "a:1", label: str = "조립비", amount: int = 30) -> RecipeAdder:
    return RecipeAdder(key, label, amount, "basis https://savetax.now/405")


def _recipe(parts: tuple | None = None, adders: tuple | None = None, **over: Any) -> Recipe:
    base_parts = ((_part(), 1),) if parts is None else parts
    base_adders = ((_adder("a:1", "조립비", 30), _adder("a:2", "카드리더", 5)),) if adders is None else adders
    if isinstance(base_adders, tuple) and base_adders and isinstance(base_adders[0], tuple):
        base_adders = base_adders[0]
    kw: dict[str, Any] = {
        "kind": "desktop",
        "parts": base_parts,
        "adders": base_adders,
        "spec_rows": (("CPU", "조달값", "기준값"),),
        "differences": (),
        "review_required": False,
        "precedent_urls": ("https://savetax.now/505",),
        "title_phrase": "동급 사양 부품 합산가",
    }
    kw.update(over)
    return Recipe(**kw)


def _product_html(
    name: str = "울트라5 225 정품",
    low: int | None = 100,
    count: int | None = 3,
    offers: list[tuple[str, int, str]] | None = None,
) -> str:
    rows = offers if offers is not None else [("몰A", 100, "무료배송"), ("몰B", 102, "무료배송"), ("몰C", 105, "무료배송")]
    items = "".join(
        f'<li><span aria-label="{mall}"></span><span class="text__num">{price:,}</span><span>{ship}</span></li>'
        for mall, price, ship in rows
    )
    offers_json = f'"lowPrice": {low}, "offerCount": {count}' if low is not None and count is not None else (
        f'"lowPrice": {low}' if low is not None else ""
    )
    return (
        "<html><head><script type=\"application/ld+json\">"
        f'{{"@type": "Product", "name": "{name}", "brand": {{"name": "B"}}, '
        f'"description": "d", "offers": {{{offers_json}}}}}'
        "</script></head><body>"
        f'<ul class="list__mall-price">{items}</ul></body></html>'
    )


def _fetcher(routes: dict[str, tuple[int, str]], seen: list[str]) -> PublicFetcher:
    def transport(url: str, headers: object, timeout: float) -> HttpResponse:
        seen.append(url)
        for key, (status, body) in routes.items():
            if key in url:
                return HttpResponse(status=status, headers={}, body=body.encode(), elapsed_seconds=0.01)
        return HttpResponse(status=404, headers={}, body=b"", elapsed_seconds=0.01)

    return PublicFetcher(SaveTaxConfig(), EvidenceConfig(), transport=transport)  # type: ignore[arg-type]


def _evidence(**over: Any) -> EvidenceConfig:
    return EvidenceConfig(**over)  # type: ignore[arg-type]


def _fact(fact_id: str = "f1", unit_price: str = "240", unit: str = "대", **over: Any) -> Fact:
    kw: dict[str, Any] = {
        "fact_id": fact_id, "listing_id": "L1", "contract_no": "C1", "request_no": "R1",
        "line_no": "1", "request_date": date(2026, 1, 5), "unit": unit, "tax_basis": "tax",
        "basis_id": "B1", "period_kind": "none", "unit_price_krw": Decimal(unit_price),
        "quantity": Decimal("2"), "comparable_amount_krw": Decimal(unit_price) * 2,
        "scorable": True, "exclusion_reasons": (),
    }
    kw.update(over)
    return Fact(**kw)


def _detail(facts: tuple = ()) -> ItemDetail:
    return ItemDetail("I1", ItemState.AVAILABLE, None, None, None, None, None, len(facts), False, facts)


class _Ds:
    pass


def test_benchmark_sums_parts_times_quantity_plus_adders() -> None:
    ev = _product_html()
    seen: list[str] = []
    fetcher = _fetcher({"pcode=11": (200, ev), "pcode=22": (200, _product_html("T500 500GB", 50, 3, [("몰A", 50, "무료배송"), ("몰B", 51, "무료배송"), ("몰C", 52, "무료배송")]))}, seen)
    r = _recipe(
        parts=((_part("k1", "CPU", "11", ("울트라5",)), 1), (_part("k2", "SSD", "22", ("T500",)), 2)),
        adders=(_adder("a:1", "조립비", 30), _adder("a:2", "카드리더", 5)),
    )
    priced = price_parts(r, fetcher, _evidence(recipe_min_offers=1), _TODAY, {})
    assert not isinstance(priced, str)
    assert recipe_benchmark(priced, r.adders) == 100 + 50 * 2 + 35


def test_floor_raises_lower_sum_and_keeps_higher() -> None:
    p = PricedPart(_part(), 1, 100, 100, _priced_evidence())
    adders = (_adder("a:1", "조립비", 100), _adder("a:2", "카드리더", 35))
    assert recipe_benchmark((p,), adders, 300) == 300
    assert recipe_benchmark((p,), adders, 200) == 235


def _priced_evidence() -> Any:
    from src.savetax.evidence import PriceEvidence

    return PriceEvidence("https://prod.danawa.com/info/?pcode=11", "인용 " + "x" * 10, 100, _TODAY, True, False)


def test_unit_price_never_below_page_lowest() -> None:
    html = _product_html("울트라5 225 정품", low=110, count=3, offers=[("몰A", 100, "무료배송"), ("몰B", 115, "무료배송"), ("몰C", 120, "무료배송")])
    priced = price_parts(_recipe(parts=((_part(), 1),)), _fetcher({"pcode=11": (200, html)}, []), _evidence(recipe_min_offers=1), _TODAY, {})
    assert not isinstance(priced, str)
    assert priced[0].unit_price_krw == 110
    assert priced[0].observed_price_krw == 110


def test_outlier_lowest_uses_second_for_b_but_not_observed() -> None:
    html = _product_html("울트라5 225 정품", low=100, count=3, offers=[("몰A", 100, "무료"), ("몰B", 120, "무료"), ("몰C", 121, "무료")])
    priced = price_parts(_recipe(parts=((_part(), 1),)), _fetcher({"pcode=11": (200, html)}, []), _evidence(recipe_min_offers=1), _TODAY, {})
    assert not isinstance(priced, str)
    assert priced[0].unit_price_krw == 120
    assert priced[0].observed_price_krw == 100
    assert "두 번째 가격 120원" in (priced[0].evidence.note or "")


def test_shipping_is_not_added() -> None:
    html = _product_html("울트라5 225 정품", low=100, count=3, offers=[("몰A", 100, "배송비 3,000원"), ("몰B", 101, "무료배송"), ("몰C", 102, "무료배송")])
    priced = price_parts(_recipe(parts=((_part(), 1),)), _fetcher({"pcode=11": (200, html)}, []), _evidence(recipe_min_offers=1), _TODAY, {})
    assert not isinstance(priced, str)
    assert priced[0].unit_price_krw == 100


def test_too_few_offers_fails_closed() -> None:
    html = _product_html("울트라5 225 정품", low=100, count=2)
    out = price_parts(_recipe(parts=((_part("cpu:T"), 1),)), _fetcher({"pcode=11": (200, html)}, []), _evidence(), _TODAY, {})
    assert out == "part_offers_low:cpu:T"


def test_name_token_mismatch_fails_closed() -> None:
    html = _product_html("다른용량 1TB 모델", low=100, count=3)
    out = price_parts(_recipe(parts=((_part("cpu:T"), 1),)), _fetcher({"pcode=11": (200, html)}, []), _evidence(recipe_min_offers=1), _TODAY, {})
    assert out == "part_identity_mismatch:cpu:T"


def test_not_on_sale_and_unreadable_fail_closed() -> None:
    html = _product_html("울트라5 225 정품", low=None, count=None, offers=[])
    out = price_parts(_recipe(parts=((_part("cpu:T"), 1),)), _fetcher({"pcode=11": (200, html)}, []), _evidence(recipe_min_offers=1), _TODAY, {})
    assert out == "part_not_on_sale:cpu:T"
    out2 = price_parts(_recipe(parts=((_part("cpu:T"), 1),)), _fetcher({"other": (200, html)}, []), _evidence(), _TODAY, {})
    assert out2 == "part_page_unreadable:cpu:T"
    out3 = price_parts(_recipe(parts=((_part("cpu:T"), 1),)), _fetcher({"pcode=11": (200, "<html>no json</html>")}, []), _evidence(), _TODAY, {})
    assert out3 == "part_page_unreadable:cpu:T"


def test_nine_parts_refused_and_cached_page_fetched_once() -> None:
    parts = tuple((_part(f"k{i}", "CPU", str(100 + i), ("울트라5",)), 1) for i in range(9))
    assert price_parts(_recipe(parts=parts), _fetcher({}, []), _evidence(), _TODAY, {}) == "recipe_part_count"
    assert price_parts(_recipe(parts=()), _fetcher({}, []), _evidence(), _TODAY, {}) == "recipe_part_count"
    html = _product_html("울트라5 225 정품", low=100, count=3)
    seen: list[str] = []
    cache: dict = {}
    r1 = _recipe(parts=((_part("k1", "CPU", "11", ("울트라5",)), 1),))
    r2 = _recipe(parts=((_part("k1", "CPU", "11", ("울트라5",)), 1),))
    assert not isinstance(price_parts(r1, _fetcher({"pcode=11": (200, html)}, seen), _evidence(recipe_min_offers=1), _TODAY, cache), str)
    assert not isinstance(price_parts(r2, _fetcher({"pcode=11": (200, html)}, seen), _evidence(recipe_min_offers=1), _TODAY, cache), str)
    assert len(seen) == 1


def _priced(unit: int = 100, qty: int = 1, low: int = 100) -> PricedPart:
    return PricedPart(_part(), qty, unit, low, _priced_evidence())


def test_threshold_filters_lines_and_shares_b_and_evidence() -> None:
    priced = (_priced(100, 1, 100),)
    adders: tuple = ()
    r = _recipe(parts=((_part(), 1),), adders=adders, spec_rows=(("CPU", "a", "b"),))
    ev = _evidence(recipe_markup_threshold=Decimal("1.20"), min_item_expected_krw=0)
    detail = _detail((_fact("f1", "119"), _fact("f2", "120", request_no="R2", line_no="2")))
    bundle = build_recipe_bundle("I1", detail, r, priced, _Ds(), ev, today=_TODAY)  # type: ignore[arg-type]
    assert bundle is not None
    assert len(bundle.lines) == 1
    assert bundle.lines[0].fact_id == "f2"
    assert bundle.rejected[0].reason == "below_threshold"
    assert all(line.benchmark_unit_price_krw == Decimal(100) for line in bundle.lines)
    assert all(len(line.price_evidence) == 1 for line in bundle.lines)
    assert bundle.lane is Lane.RECIPE


def test_no_admissible_line_yields_none_and_unit_rejected() -> None:
    priced = (_priced(100),)
    r = _recipe(parts=((_part(), 1),), adders=())
    ev = _evidence(min_item_expected_krw=0)
    assert build_recipe_bundle("I1", _detail((_fact("f1", "50"),)), r, priced, _Ds(), ev, today=_TODAY) is None  # type: ignore[arg-type]
    bundle = build_recipe_bundle("I1", _detail((_fact("f1", "5000", unit="세트"),)), r, priced, _Ds(), ev, today=_TODAY)  # type: ignore[arg-type]
    assert bundle is None


def test_min_expected_filters_and_review_propagates() -> None:
    priced = (_priced(100),)
    r = _recipe(parts=((_part(), 1),), adders=())
    ev = _evidence(min_item_expected_krw=10_000_000)
    assert build_recipe_bundle("I1", _detail((_fact("f1", "5000"),)), r, priced, _Ds(), ev, today=_TODAY) is None  # type: ignore[arg-type]
    r2 = _recipe(parts=((_part(), 1),), adders=(), review_required=True)
    ev2 = _evidence(min_item_expected_krw=0)
    bundle = build_recipe_bundle("I1", _detail((_fact("f1", "5000"),)), r2, priced, _Ds(), ev2, today=_TODAY)  # type: ignore[arg-type]
    assert bundle is not None
    assert bundle.review_required is True


def test_texts_carry_verbatim_sentences_and_arithmetic() -> None:
    priced = (_priced(100, 1, 100),)
    adders = (_adder("a:1", "조립비", 30), _adder("a:2", "카드리더", 5))
    r = _recipe(parts=((_part(), 1),), adders=adders, differences=("차이점",),
                precedent_urls=("https://savetax.now/505",), spec_rows=(("CPU", "조달CPU", "기준CPU"),))
    ev = _evidence(min_item_expected_krw=0)
    bundle = build_recipe_bundle("I1", _detail((_fact("f1", "5000"),)), r, priced, _Ds(), ev, today=_TODAY)  # type: ignore[arg-type]
    assert bundle is not None
    assert "부품 합계 100원" in bundle.claim
    assert "조립비 30원" in bundle.claim
    assert "135원" in bundle.claim
    for sentence in ("같은 모델을 시중에서 구할 수 없을 때는 동급 제품의 실제 판매가와 비교합니다",
                     "처리 성능·화면·용량 등 제품의 급과 가격을 정하는 핵심 사양은 같거나 더 높아야 합니다",
                     "본체와 부속을 따로 파는 제품은 둘을 더한 값과 비교합니다",
                     "배송비는 더하지 않습니다",
                     "나라장터 납품요구 단가가 기준가의 1.2배 이상인 계약만 판정 목록에 넣습니다"):
        assert sentence in bundle.reproduction_notes
    assert "https://savetax.now/method" in bundle.reproduction_notes
    assert "https://savetax.now/505" in bundle.reproduction_notes
    assert len(bundle.claim) <= 2000
    assert len(bundle.comparison_scope) <= 1000
    assert len(bundle.reproduction_notes) <= 4000
    assert "동급 사양 부품 합산가 대비" in bundle.title
    assert "(동일 모델 아님)" in bundle.title


def test_floor_raise_sentence_and_bundle_round_trip(tmp_path) -> None:
    priced = (_priced(100, 1, 100),)
    adders = (_adder("a:1", "조립비", 30),)
    r = _recipe(parts=((_part(), 1),), adders=adders, floor_krw=500)
    ev = _evidence(min_item_expected_krw=0)
    bundle = build_recipe_bundle("I1", _detail((_fact("f1", "5000"),)), r, priced, _Ds(), ev, today=_TODAY)  # type: ignore[arg-type]
    assert bundle is not None
    assert "그 값으로 올림 = 500원" in bundle.claim
    assert "부품 합계 100원 + 조립비 30원 = 130원 →" in bundle.claim
    path = tmp_path / "recipe-bundles.jsonl"
    write_bundles([bundle], path)
    from datetime import datetime, UTC

    loaded = load_bundles([path], now=datetime.now(UTC))
    assert loaded["I1"].lane is Lane.RECIPE
    assert loaded["I1"].lines == bundle.lines


def test_product_dataclass_round_trip() -> None:
    p = DanawaProduct("1", "n", None, "d", 10, 2, (MallOffer("m", 10, 0),), None)
    assert p.low_price_krw == 10


def test_offer_count_none_uses_offers_and_empty_offers_use_low() -> None:
    html = _product_html("울트라5 225 정품", low=100, count=None)
    priced = price_parts(_recipe(parts=((_part(), 1),)), _fetcher({"pcode=11": (200, html)}, []), _evidence(recipe_min_offers=1), _TODAY, {})
    assert not isinstance(priced, str)
    assert priced[0].unit_price_krw == 100
    html2 = _product_html("울트라5 225 정품", low=100, count=3, offers=[])
    priced2 = price_parts(_recipe(parts=((_part(), 1),)), _fetcher({"pcode=11": (200, html2)}, []), _evidence(recipe_min_offers=1), _TODAY, {})
    assert not isinstance(priced2, str)
    assert priced2[0].unit_price_krw == 100


def test_cached_reason_returned_without_fetch() -> None:
    seen: list[str] = []
    out = price_parts(_recipe(parts=((_part("cpu:T"), 1),)), _fetcher({}, seen), _evidence(), _TODAY, {"11": "part_page_unreadable:cpu:T"})
    assert out == "part_page_unreadable:cpu:T"
    assert seen == []


def test_evidence_failure_fails_closed() -> None:
    long_name = "울트라5 225 정품 " + "가" * 1950
    html = _product_html(long_name, low=100, count=3)
    out = price_parts(_recipe(parts=((_part("cpu:T"), 1),)), _fetcher({"pcode=11": (200, html)}, []), _evidence(recipe_min_offers=1), _TODAY, {})
    assert out == "part_page_unreadable:cpu:T"


def test_blank_unit_and_other_rejections() -> None:
    priced = (_priced(100),)
    r = _recipe(parts=((_part(), 1),), adders=())
    ev = _evidence(min_item_expected_krw=0)
    assert build_recipe_bundle("I1", _detail((_fact("f1", "5000", unit="미상"),)), r, priced, _Ds(), ev, today=_TODAY) is None  # type: ignore[arg-type]
    assert build_recipe_bundle("I1", _detail((_fact("f1", "5000", scorable=False),)), r, priced, _Ds(), ev, today=_TODAY) is None  # type: ignore[arg-type]
    assert build_recipe_bundle("I1", _detail((_fact("f1", "5000", period_kind="month"),)), r, priced, _Ds(), ev, today=_TODAY) is None  # type: ignore[arg-type]
    assert build_recipe_bundle("I1", _detail((_fact("f1", "5000", request_date=None),)), r, priced, _Ds(), ev, today=_TODAY) is None  # type: ignore[arg-type]


def test_scope_truncation_and_notes_truncation() -> None:
    priced = (_priced(100),)
    rows = tuple((f"스펙{i}", "조달값" * 40, "기준값" * 40) for i in range(12))
    r = _recipe(parts=((_part(), 1),), adders=(), spec_rows=rows)
    ev = _evidence(min_item_expected_krw=0)
    bundle = build_recipe_bundle("I1", _detail((_fact("f1", "5000"),)), r, priced, _Ds(), ev, today=_TODAY)  # type: ignore[arg-type]
    assert bundle is not None
    assert len(bundle.comparison_scope) <= 1000
    r2 = _recipe(parts=((_part(), 1),), adders=(), spec_rows=())
    bundle2 = build_recipe_bundle("I1", _detail((_fact("f1", "5000"),)), r2, priced, _Ds(), ev, today=_TODAY)  # type: ignore[arg-type]
    assert bundle2 is not None
    big_adders = tuple(_adder(f"a:{i}", "가산항목" * 30, 10 + i) for i in range(12))
    big_diffs = tuple("차이설명" * 100 + str(i) for i in range(10))
    r3 = _recipe(parts=((_part(), 1),), adders=big_adders, differences=big_diffs)
    bundle3 = build_recipe_bundle("I1", _detail((_fact("f1", "5000"),)), r3, priced, _Ds(), ev, today=_TODAY)  # type: ignore[arg-type]
    assert bundle3 is not None
    assert len(bundle3.reproduction_notes) <= 4000
    r4 = _recipe(parts=((_part(), 1),), adders=big_adders, differences=())
    bundle4 = build_recipe_bundle("I1", _detail((_fact("f1", "5000"),)), r4, priced, _Ds(), ev, today=_TODAY)  # type: ignore[arg-type]
    assert bundle4 is not None
    assert len(bundle4.reproduction_notes) <= 4000


def test_notes_truncated_without_differences() -> None:
    priced = (_priced(100),)
    big_adders = tuple(_adder(f"a:{i}", "L", 10 + i) for i in range(3))
    big_adders = tuple(RecipeAdder(a.key, a.label, a.amount_krw, "근거" * 600) for a in big_adders)
    r = _recipe(parts=((_part(), 1),), adders=big_adders, differences=())
    ev = _evidence(min_item_expected_krw=0)
    with pytest.raises(DomainValidationError, match="notes exceed"):
        build_recipe_bundle("I1", _detail((_fact("f1", "5000"),)), r, priced, _Ds(), ev, today=_TODAY)  # type: ignore[arg-type]


def test_zero_expected_line_rejected() -> None:
    priced = (_priced(100),)
    r = _recipe(parts=((_part(), 1),), adders=())
    ev = _evidence(min_item_expected_krw=0)
    fact = _fact("f1", "5000", comparable_amount_krw=Decimal(10))
    bundle = build_recipe_bundle("I1", _detail((fact,)), r, priced, _Ds(), ev, today=_TODAY)  # type: ignore[arg-type]
    assert bundle is None


@pytest.mark.parametrize(("quantity", "unit", "observed"), [(0, 100, 100), (-1, 100, 100), (True, 100, 100), (1, 0, 100), (1, 99, 100)])
def test_benchmark_rejects_invalid_parts(quantity, unit, observed) -> None:
    part = PricedPart(_part(), quantity, unit, observed, _priced_evidence())
    with pytest.raises(DomainValidationError):
        recipe_benchmark((part,), ())


def test_benchmark_rejects_empty_parts_and_invalid_adders() -> None:
    with pytest.raises(DomainValidationError):
        recipe_benchmark((), ())
    with pytest.raises(DomainValidationError):
        recipe_benchmark((_priced(),), (_adder(amount=-1),))
    with pytest.raises(DomainValidationError):
        recipe_benchmark((_priced(),), (), -1)


def test_bundle_identity_and_availability_gates() -> None:
    from dataclasses import replace

    r = _recipe(adders=())
    detail = _detail((_fact(),))
    ev = _evidence(min_item_expected_krw=0)
    with pytest.raises(DataIntegrityError):
        build_recipe_bundle("other", detail, r, (_priced(),), _Ds(), ev, today=_TODAY)
    assert build_recipe_bundle("I1", replace(detail, state=ItemState.CLAIMED), r, (_priced(),), _Ds(), ev, today=_TODAY) is None


def test_long_claim_is_refused_without_losing_required_clauses() -> None:
    priced = (_priced(),)
    r = _recipe(adders=(), parts=((_part(), 1),))
    from dataclasses import replace

    priced = (replace(priced[0], part=_part(label="부품" * 1200)),)
    with pytest.raises(DomainValidationError, match="claim exceeds"):
        build_recipe_bundle("I1", _detail((_fact("f1", "5000"),)), r, priced, _Ds(), _evidence(min_item_expected_krw=0), today=_TODAY)


def test_all_lines_share_seven_part_evidence_and_quantity_arithmetic() -> None:
    from dataclasses import replace

    priced = tuple(replace(_priced(), part=_part(key=f"part:{i}", pcode=str(i + 1)), quantity=2 if i == 1 else 1) for i in range(7))
    recipe = _recipe(parts=tuple((p.part, p.quantity) for p in priced), adders=(_adder(amount=30),))
    detail = _detail((_fact("f1", "1000"), _fact("f2", "1200"), _fact("f3", "995")))
    bundle = build_recipe_bundle("I1", detail, recipe, priced, _Ds(), _evidence(min_item_expected_krw=0), today=_TODAY)
    assert bundle is not None
    assert [line.fact_id for line in bundle.lines] == ["f1", "f2"]
    assert [line.benchmark_unit_price_krw for line in bundle.lines] == [Decimal(830), Decimal(830)]
    assert [line.expected_krw for line in bundle.lines] == [340, 740]
    assert bundle.lines[0].price_evidence == bundle.lines[1].price_evidence
    assert len(bundle.lines[0].price_evidence) == 7
    assert bundle.rejected[0].reason == "below_threshold"
