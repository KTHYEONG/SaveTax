"""Parts-sum recipe lane over seller product pages and fixed adders."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal, localcontext
from urllib.parse import urlparse

from src.savetax.audit import _arithmetic_precision
from src.savetax.competition import ItemDetail, ItemState
from src.savetax.config import EvidenceConfig, RecipeAdder, RecipePart
from src.savetax.danawa import DanawaProduct, MallOffer, parse_product
from src.savetax.dataset import Dataset
from src.savetax.errors import ConfigError, DataIntegrityError, DomainValidationError, NetworkError, SaveTaxError
from src.savetax.evidence import EvidenceBundle, Lane, LineProposal, PriceEvidence, RejectedFact, make_line
from src.savetax.publicweb import PublicFetcher
from src.savetax.redaction import redact
from src.savetax.units import normalize_unit

logger = logging.getLogger(__name__)

__all__ = ["PricedPart", "Recipe", "adder_page_amount", "build_recipe_bundle", "mount_form_gate", "price_adders", "price_parts", "recipe_benchmark"]

_SENTENCES = (
    "조달 전용 모델이나 상표만 바꾼 수입품, 단종 제품처럼 같은 모델을 시중에서 구할 수 없을 때는 동급 제품의 실제 판매가와 비교합니다",
    "처리 성능·화면·용량 등 제품의 급과 가격을 정하는 핵심 사양은 같거나 더 높아야 합니다",
    "본체와 부속을 따로 파는 제품은 둘을 더한 값과 비교합니다",
    "배송비는 더하지 않습니다",
    "나라장터 납품요구 단가가 기준가의 1.2배 이상인 계약만 판정 목록에 넣습니다",
)
_METHOD_URL = "https://savetax.now/method"
_LIMIT_NOTE = "보증 기간, 부가 기능·소프트웨어, 인증, 현장 여건에 따른 설치 난이도는 가격에 반영하지 않았다. 가격은 확인일 현재 판매가이며 납품요구 당시 가격과 다를 수 있다."


@dataclass(frozen=True)
class Recipe:
    """Equivalent-product assembly and its published basis."""

    kind: str
    parts: tuple[tuple["RecipePart", int], ...]  # noqa: UP037
    adders: tuple["RecipeAdder", ...]  # noqa: UP037
    spec_rows: tuple[tuple[str, str, str], ...]
    differences: tuple[str, ...]
    review_required: bool
    precedent_urls: tuple[str, ...]
    title_phrase: str
    floor_krw: int | None = None
    sum_label: str = "부품 합계"
    exact_sum: bool = False


@dataclass(frozen=True)
class PricedPart:
    """One reference part with its seller-page unit price and citation."""

    part: RecipePart
    quantity: int
    unit_price_krw: int
    observed_price_krw: int
    evidence: PriceEvidence
    second_unit_price_krw: int | None = None
    second_evidence: PriceEvidence | None = None


_QUOTE_SPEC_CHARS = 400


def _ordered(product: DanawaProduct) -> list[MallOffer]:
    return sorted(product.offers, key=lambda o: (o.price_krw, o.mall))


def _compact_name(value: str) -> str:
    return "".join(value.split()).casefold()


def _offer_count(product: DanawaProduct) -> int:
    if product.offer_count is not None:
        return product.offer_count
    return len(product.offers)


def _second_used(product: DanawaProduct, evidence: EvidenceConfig) -> bool:
    offers = _ordered(product)
    return len(offers) >= 2 and Decimal(offers[1].price_krw) > Decimal(offers[0].price_krw) * evidence.outlier_ratio


def _chosen_price(product: DanawaProduct, evidence: EvidenceConfig) -> int | None:
    offers = _ordered(product)
    if not offers:
        return None
    if len(offers) >= 2 and Decimal(offers[1].price_krw) > Decimal(offers[0].price_krw) * evidence.outlier_ratio:
        return offers[1].price_krw
    return offers[0].price_krw


def _sum_expression(priced: Sequence[PricedPart]) -> str:
    terms = " + ".join(f"{p.part.label} {p.unit_price_krw * p.quantity:,}원" for p in priced)
    return f"(= {terms})"


def _quote(part: RecipePart, product: DanawaProduct, today: date) -> str:
    count = _offer_count(product)
    low = product.low_price_krw or 0
    # Rows under the page's stated lowest price carry payment conditions the page excludes from that price.
    sellers = ", ".join(f"{o.mall} {o.price_krw:,}원" for o in [o for o in _ordered(product) if o.price_krw >= low][:3])
    head = product.name
    spec = " ".join(product.description.split())[:_QUOTE_SPEC_CHARS]
    if spec:
        head += f" / 상세 스펙 {spec}"
    head += f" / 최저가 {low:,}원 / 판매자 {count}곳"
    if sellers:
        head += f" / {sellers}"
    return head + f" / 다나와 상품 페이지, {today} 확인"


def _note(part: RecipePart, quantity: int, adders: Sequence["RecipeAdder"], product: DanawaProduct, evidence: EvidenceConfig) -> str:  # noqa: UP037
    labels = " + ".join(a.label for a in adders)
    text = f"{part.label} {quantity}개 반영. 비교 단가 = 부품 합계 + {labels}"
    if _second_used(product, evidence):
        chosen = _chosen_price(product, evidence) or 0
        text += f" / 최저가가 다른 판매처와 15% 넘게 벌어져 두 번째 가격 {chosen:,}원을 반영"
    return text


def adder_page_amount(page_text: str, amount_krw: int) -> str | None:
    """Return the visible phrase that states `amount_krw` on a public page, or None when absent."""
    from src.savetax.publicweb import visible_text

    figure = f"{amount_krw:,}원"
    norm = " ".join(visible_text(page_text).split())
    fallback: str | None = None
    pos = 0
    while (idx := norm.find(figure, pos)) >= 0:
        pos = idx + 1
        end = idx + len(figure)
        if (idx > 0 and (norm[idx - 1].isdigit() or norm[idx - 1] == ",")) or (
            end < len(norm) and (norm[end].isdigit() or norm[end] == ",")
        ):
            continue
        phrase = _amount_phrase(norm, idx, end)
        # The phrase naming the published standard is the citation; other mentions sit next to unrelated prices.
        if _ADDER_STANDARD_MARK in phrase:
            return phrase
        fallback = fallback or phrase
    return fallback


_ADDER_STANDARD_MARK = "표준품셈"
_ADDER_PHRASE_LIMIT = 300
_NOTE_LIMIT = 500
_MAX_PRICE_EVIDENCE = 8


def _amount_phrase(norm: str, start: int, end: int) -> str:
    floor = max(0, start - _ADDER_PHRASE_LIMIT)
    cuts = [cut + len(sep) for sep in (" + ", " · ", ". ") if (cut := norm.rfind(sep, floor, start)) >= 0]
    left = max(cuts, default=floor)
    close = norm.find(")", end, end + 60) if norm.startswith("(", end) else -1
    right = close + 1 if close >= 0 else end
    return norm[left:right][-_ADDER_PHRASE_LIMIT:].strip()


def price_adders(
    recipe: Recipe, fetcher: PublicFetcher, today: date
) -> tuple[PriceEvidence, ...] | str:
    """Cite every sourced adder of the recipe; returns a reason code when a page does not state its amount."""
    cited: list[PriceEvidence] = []
    for adder in recipe.adders:
        if adder.source_url is None:
            continue
        page = fetcher.get(adder.source_url)
        if page.status != 200 or not page.text:
            return f"adder_page_unreadable:{adder.key}"
        fragment = adder_page_amount(page.text, adder.amount_krw)
        if fragment is None:
            return f"adder_amount_missing:{adder.key}"
        quote = (
            f"{fragment} / savetax.now 상세분석 화면, {today} 확인"
        )
        note = f"{adder.label} {adder.amount_krw:,}원 반영"
        cited.append(
            PriceEvidence(
                source_url=adder.source_url,
                quote=quote,
                observed_price_krw=adder.amount_krw,
                observed_at=today,
                vat_included=True,
                shipping_included=False,
                note=note,
            )
        )
    return tuple(cited)


def _sourced_eligible(recipe: Recipe) -> bool:
    return (
        not recipe.exact_sum
        and len(recipe.adders) > 0
        and all(a.source_url is not None for a in recipe.adders)
    )


def _seller_quote(title: str, price: int, host: str, today: date) -> str:
    return f"{title} / 판매가 {price:,}원 / {host} 상품 페이지, {today} 확인"


def _seller_note(part: RecipePart, quantity: int, recipe: Recipe) -> str:
    if recipe.exact_sum:
        return f"{part.label} {quantity}개 반영. 비교 단가 = {recipe.sum_label}"[:500]
    labels = " + ".join(a.label for a in recipe.adders)
    base = f"{part.label} {quantity}개 반영. 비교 단가 = 부품 합계 + {labels}" if labels else f"{part.label} {quantity}개 반영"
    return base[:500]


def _price_seller_part(
    part: RecipePart,
    quantity: int,
    recipe: Recipe,
    fetcher: PublicFetcher,
    today: date,
    cache: dict[str, object],
) -> PricedPart | str:
    from src.savetax.shops import ShopProduct, parse_seller_page

    readable: list[tuple[int, str, ShopProduct]] = []
    saw_sold_out = False
    saw_mismatch = False
    for url in part.sources:
        cached = cache.get(url)
        if isinstance(cached, str):
            if cached == "sold_out":
                saw_sold_out = True
            elif cached == "mismatch":
                saw_mismatch = True
            continue
        product: ShopProduct | None = cached if isinstance(cached, ShopProduct) else None
        if product is None:
            try:
                page = fetcher.get(url)
            except (ConfigError, NetworkError):
                raise
            except SaveTaxError:
                continue
            except (OSError, ValueError):
                continue
            if page.status != 200 or not page.text:
                cache[url] = "unreadable"
                continue
            raw = page.raw if getattr(page, "raw", b"") else page.text.encode("utf-8")
            content_type = getattr(page, "content_type", None)
            try:
                parsed = parse_seller_page(raw, url, content_type)
            except (ValueError, OSError):
                cache[url] = "unreadable"
                continue
            if parsed is None:
                from src.savetax.shops import _decode_seller_page

                lowered = _decode_seller_page(raw, content_type)
                if any(marker in lowered for marker in ("품절", "판매종료", "일시품절")):
                    saw_sold_out = True
                    cache[url] = "sold_out"
                else:
                    cache[url] = "unreadable"
                continue
            flat = _compact_name(parsed.title or "")
            if any(_compact_name(token) not in flat for token in part.name_tokens):
                saw_mismatch = True
                continue
            cache[url] = parsed
            product = parsed
        else:
            flat = _compact_name(product.title or "")
            if any(_compact_name(token) not in flat for token in part.name_tokens):
                saw_mismatch = True
                continue
        assert product is not None
        readable.append((product.price_krw, url, product))
    if not readable:
        if saw_sold_out:
            return f"part_sold_out:{part.key}"
        if saw_mismatch:
            return f"part_name_mismatch:{part.key}"
        return f"part_page_unreadable:{part.key}"
    readable.sort(key=lambda item: (item[0], item[1]))
    chosen_price, chosen_url, chosen_product = readable[0]
    chosen_host = (urlparse(chosen_url).hostname or "").lower()
    second: tuple[int, str, ShopProduct] | None = None
    for price, url, product in readable[1:]:
        if (urlparse(url).hostname or "").lower().removeprefix("www.") != chosen_host.removeprefix("www."):
            second = (price, url, product)
            break
    try:
        proof = PriceEvidence(
            source_url=chosen_url,
            quote=_seller_quote(chosen_product.title, chosen_price, chosen_host, today),
            observed_price_krw=chosen_price,
            observed_at=today,
            vat_included=True,
            shipping_included=False,
            note=_seller_note(part, quantity, recipe),
        )
        second_evidence: PriceEvidence | None = None
        second_price: int | None = None
        if second is not None:
            second_price, second_url, second_product = second
            second_host = (urlparse(second_url).hostname or "").lower()
            second_evidence = PriceEvidence(
                source_url=second_url,
                quote=_seller_quote(second_product.title, second_price, second_host, today),
                observed_price_krw=second_price,
                observed_at=today,
                vat_included=True,
                shipping_included=False,
                note=_seller_note(part, quantity, recipe),
            )
    except DomainValidationError:
        return f"part_page_unreadable:{part.key}"
    return PricedPart(part, quantity, chosen_price, chosen_price, proof, second_price, second_evidence)


def _price_legacy_part(
    part: RecipePart,
    quantity: int,
    recipe: Recipe,
    fetcher: PublicFetcher,
    evidence: EvidenceConfig,
    today: date,
    cache: dict[str, object],
) -> PricedPart | str:
    url = evidence.danawa_product_url_template.format(pcode=part.pcode)
    cached = cache.get(part.pcode)
    if isinstance(cached, str):
        return cached
    product: DanawaProduct | None = cached if isinstance(cached, DanawaProduct) else None
    if product is None:
        page = fetcher.get(url)
        if page.status != 200:
            reason = f"part_page_unreadable:{part.key}"
            cache[part.pcode] = reason
            return reason
        try:
            product = parse_product(page.text, part.pcode)
        except (DataIntegrityError, DomainValidationError, ValueError):
            reason = f"part_page_unreadable:{part.key}"
            cache[part.pcode] = reason
            return reason
        cache[part.pcode] = product
    assert product is not None
    if product.low_price_krw is None:
        return f"part_not_on_sale:{part.key}"
    if _offer_count(product) < evidence.recipe_min_offers:
        return f"part_offers_low:{part.key}"
    flat = _compact_name(product.name or "")
    if any(_compact_name(token) not in flat for token in part.name_tokens):
        return f"part_identity_mismatch:{part.key}"
    chosen = _chosen_price(product, evidence)
    base = product.low_price_krw if chosen is None else max(product.low_price_krw, chosen)
    if (recipe.exact_sum or _sourced_eligible(recipe)) and base != int(product.low_price_krw):
        return f"part_price_unstable:{part.key}"
    try:
        proof = PriceEvidence(
            source_url=url,
            quote=_quote(part, product, today),
            observed_price_krw=int(product.low_price_krw),
            observed_at=today,
            vat_included=True,
            shipping_included=False,
            note=(f"{part.label} {quantity}개 반영. 비교 단가 = {recipe.sum_label}" if recipe.exact_sum else _note(part, quantity, recipe.adders, product, evidence)),
        )
    except DomainValidationError:
        return f"part_page_unreadable:{part.key}"
    return PricedPart(part, quantity, base, int(product.low_price_krw), proof)


def mount_form_gate(recipe: Recipe, procured_text: str) -> str | None:
    """Return mount_form_mismatch when the recipe panel form does not fit the procured mount, else None."""
    forms = {part.mount_form for part, _ in recipe.parts if part.mount_form is not None}
    if not forms:
        return None
    enclosure = ("단자함" in procured_text or "함체" in procured_text)
    if enclosure:
        return None if "enclosure" in forms and "rack" not in forms else "mount_form_mismatch"
    return None if "rack" in forms and "enclosure" not in forms else "mount_form_mismatch"


def price_parts(
    recipe: Recipe,
    fetcher: PublicFetcher,
    evidence: EvidenceConfig,
    today: date,
    cache: dict[str, object],
) -> tuple[PricedPart, ...] | str:
    """Price every part from its seller page; returns a reason code when any part cannot be priced."""
    # One seller part past the citation cap may ride uncited in B (see build_recipe_bundle); more would leave too much
    # of B outside the evidence objects.
    if not recipe.parts or len(recipe.parts) > _MAX_PRICE_EVIDENCE + 1:
        return "recipe_part_count"
    priced: list[PricedPart] = []
    for part, quantity in recipe.parts:
        if part.sources_pending:
            return f"part_sources_pending:{part.key}"
        if part.sources:
            out = _price_seller_part(part, quantity, recipe, fetcher, today, cache)
        elif part.pcode:
            out = _price_legacy_part(part, quantity, recipe, fetcher, evidence, today, cache)
        else:
            return f"part_page_unreadable:{part.key}"
        if isinstance(out, str):
            return out
        priced.append(out)
    return tuple(priced)


def recipe_benchmark(priced: Sequence[PricedPart], adders: Sequence["RecipeAdder"], floor_krw: int | None = None) -> int:  # noqa: UP037
    """Return B in KRW: parts times quantity plus every adder, raised to the floor when one is given."""
    if not priced or any(type(value) is not int or value <= 0 for p in priced for value in (p.quantity, p.unit_price_krw, p.observed_price_krw)):
        raise DomainValidationError("[DATA] invalid recipe parts: positive integer amounts and quantities required")
    if any(p.unit_price_krw < p.observed_price_krw for p in priced):
        raise DomainValidationError("[DATA] recipe benchmark is below observed price")
    if any(type(a.amount_krw) is not int or a.amount_krw <= 0 for a in adders) or (floor_krw is not None and (type(floor_krw) is not int or floor_krw <= 0)):
        raise DomainValidationError("[DATA] invalid recipe adders or floor: positive integers required")
    total = sum(p.unit_price_krw * p.quantity for p in priced) + sum(a.amount_krw for a in adders)
    if floor_krw is not None and total < floor_krw:
        return floor_krw
    return total


def _is_single_piece(unit: str, evidence: EvidenceConfig) -> bool:
    norm = normalize_unit(unit)
    if norm is None:
        return False
    return any(norm == normalize_unit(allowed) for allowed in evidence.single_piece_units)


def _threshold_text(evidence: EvidenceConfig) -> str:
    text = format(evidence.recipe_markup_threshold, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _claim(
    item_id: str,
    priced: Sequence[PricedPart],
    adders: Sequence["RecipeAdder"],  # noqa: UP037
    benchmark: int,
    floor: int | None,
    raised: bool,
    kept: int,
    sum_label: str = "부품 합계",
    exact_sum: bool = False,
) -> str:
    parts_sum = sum(p.unit_price_krw * p.quantity for p in priced)
    labels = ", ".join(p.part.label for p in priced)
    if not exact_sum:
        arithmetic = f"{sum_label} {parts_sum:,}원"
        for adder in adders:
            arithmetic += f" + {adder.label} {adder.amount_krw:,}원"
        subtotal = parts_sum + sum(a.amount_krw for a in adders)
        arithmetic += f" = {subtotal:,}원"
        if raised and floor is not None:
            arithmetic += f" → 주최 측이 같은 급 판정에 쓴 가장 높은 기준가 {floor:,}원보다 낮아 그 값으로 올림 = {benchmark:,}원"
    else:
        breakdown = " + ".join(f"{p.part.label} {p.observed_price_krw * p.quantity:,}원" for p in priced)
        arithmetic = f"{sum_label} {benchmark:,}원 ({breakdown})"
    comparison = (
        f"같은 급의 시판 제품({labels}) 판매가 합계와 비교한 것으로 동일 모델 비교가 아니다. "
        if exact_sum else
        f"같은 급의 시판 제품({labels}) 판매가에 주최 측이 공개한 가산액을 더한 값과 비교한 것으로 동일 모델 비교가 아니다. "
    )
    return (
        f"물품식별번호 {item_id}는 조달 전용 모델로 같은 모델의 판매 페이지가 없어 동일 모델 비교가 불가하다. "
        f"{comparison}"
        f"{arithmetic}. 납품요구 단가가 비교 단가의 기준 배율 이상인 납품요구 {kept}건을 비교하였다. "
        "납품요구 기록 기준이며 검수·지급 완료나 특정 업체·기관의 위법·의도를 뜻하지 않는다."
    )


def _scope(recipe: Recipe, evidence: EvidenceConfig) -> str:
    head = "동급 사양 비교(주최 측 「자료와 방법」 판정 2, 동일 모델 아님) · "
    tail = f" · 신품 · 부가세 포함 · 배송비 미가산 · 납품요구 단가가 비교 단가의 {_threshold_text(evidence)}배 이상"
    rows = [f"{name}: 조달 {proc} / 기준 {ref}" for name, proc, ref in recipe.spec_rows]
    while rows:
        text = head + " · ".join(rows) + tail
        if len(text) <= 1000:
            return text
        rows.pop()
    return head + tail.removeprefix(" · ")


def _notes(recipe: Recipe, evidence: EvidenceConfig, today: date) -> str:
    _ = evidence
    lines = [f"{s}({_METHOD_URL}, {today} 확인)" for s in _SENTENCES]
    base = " ".join(lines)
    precedents = "[선례] " + ", ".join(recipe.precedent_urls)
    adders = "[가산] " + "; ".join(f"{a.label} {a.amount_krw:,}원, {a.basis}" for a in recipe.adders)
    diffs = list(recipe.differences)
    while True:
        tail = " [차이·한계] " + (" / ".join(diffs) + " " if diffs else "") + _LIMIT_NOTE
        text = f"{base} {precedents} {adders}{tail}"
        if len(text) <= 4000:
            return text
        if not diffs:
            raise DomainValidationError("[DATA] recipe notes exceed text limit")
        diffs.pop()


def _second_seller_note(priced: Sequence[PricedPart], adders: Sequence["RecipeAdder"], second_benchmark: int) -> str:  # noqa: UP037
    """Disclose each part's second-seller page so reviewers can recheck the 1.2-1.3x band gate."""
    parts = [
        f"{p.part.label} {int(p.second_unit_price_krw or 0):,}원{f' × {p.quantity}' if p.quantity != 1 else ''}"  # noqa: RUF001
        f"({p.second_evidence.source_url if p.second_evidence else ''})"
        for p in priced
    ]
    parts += [f"{a.label} {a.amount_krw:,}원" for a in adders]
    summed = sum(int(p.second_unit_price_krw or 0) * p.quantity for p in priced) + sum(a.amount_krw for a in adders)
    floor = "" if summed == second_benchmark else f", 하한 적용 {second_benchmark:,}원"
    return (
        f" [두 번째 판매처] {' + '.join(parts)} = {summed:,}원{floor}."
        " 납품요구 단가가 가장 싼 판매처 기준 1.3배 미만인 납품요구는 이 값의 1.2배 이상인 것만 넣었다."
    )


def _fully_sourced(recipe: Recipe, adder_evidence: Sequence[PriceEvidence]) -> bool:
    if recipe.exact_sum or not recipe.adders:
        return False
    if any(a.source_url is None for a in recipe.adders):
        return False
    if len(adder_evidence) != len(recipe.adders):
        return False
    return sorted((a.source_url, a.amount_krw) for a in recipe.adders) == sorted(
        (e.source_url, e.observed_price_krw) for e in adder_evidence
    )


def _sourced_sum_expression(priced: Sequence[PricedPart], adders: Sequence["RecipeAdder"]) -> str:  # noqa: UP037
    terms = [
        f"{p.part.label} {p.unit_price_krw:,}원 × {p.quantity}"  # noqa: RUF001
        if p.quantity > 1
        else f"{p.part.label} {p.unit_price_krw:,}원"
        for p in priced
    ]
    terms += [f"{a.label} {a.amount_krw:,}원" for a in adders]
    return f"(= {' + '.join(terms)})"


def _sourced_claim(
    item_id: str,
    priced: Sequence[PricedPart],
    adders: Sequence["RecipeAdder"],  # noqa: UP037
    benchmark: int,
    kept: int,
    gate_filtered: bool,
    evidence: EvidenceConfig,
) -> str:
    parts_sum = sum(p.unit_price_krw * p.quantity for p in priced)
    labels = ", ".join(p.part.label for p in priced)
    arithmetic = f"판매가 {parts_sum:,}원"
    for adder in adders:
        arithmetic += f" + {adder.label} {adder.amount_krw:,}원"
    arithmetic += f" = {benchmark:,}원"
    comparison = (
        f"같은 급의 시판 제품({labels}) 판매가에 주최 측이 공개한 가산액을 더한 값과 비교한 것으로 동일 모델 비교가 아니다. "
    )
    text = (
        f"물품식별번호 {item_id}는 조달 전용 모델로 같은 모델의 판매 페이지가 없어 동일 모델 비교가 불가하다. "
        f"{comparison}"
        f"{arithmetic}. 납품요구 단가가 비교 단가의 기준 배율 이상인 납품요구 {kept}건을 비교하였다. "
    )
    if gate_filtered:
        text += f"주최 측이 같은 급 판정에 쓴 가장 높은 기준가의 {_threshold_text(evidence)}배 미만인 납품요구는 넣지 않았다. "
    text += "납품요구 기록 기준이며 검수·지급 완료나 특정 업체·기관의 위법·의도를 뜻하지 않는다."
    return text


def _second_benchmark(
    priced: Sequence[PricedPart],
    adders: Sequence["RecipeAdder"],  # noqa: UP037
    floor_krw: int | None,
) -> int | None:
    if not priced or any(p.second_unit_price_krw is None for p in priced):
        return None
    if any(type(p.second_unit_price_krw) is not int or int(p.second_unit_price_krw or 0) <= 0 for p in priced):
        return None
    if any(
        p.second_evidence is None
        or p.second_evidence.observed_price_krw != p.second_unit_price_krw
        or (bool(p.part.sources) and p.second_evidence.source_url not in p.part.sources)
        or int(p.second_unit_price_krw or 0) < p.unit_price_krw
        or (urlparse(p.second_evidence.source_url).hostname or "").removeprefix("www.")
        == (urlparse(p.evidence.source_url).hostname or "").removeprefix("www.")
        for p in priced
    ):
        return None
    total = sum(int(p.second_unit_price_krw or 0) * p.quantity for p in priced) + sum(a.amount_krw for a in adders)
    if floor_krw is not None and total < floor_krw:
        return floor_krw
    return total


def build_recipe_bundle(
    item_id: str,
    detail: ItemDetail,
    recipe: Recipe,
    priced: Sequence[PricedPart],
    dataset: Dataset,
    evidence: EvidenceConfig,
    *,
    today: date,
    allowed_states: frozenset[ItemState] = frozenset({ItemState.AVAILABLE}),
    adder_evidence: Sequence[PriceEvidence] = (),
    attributes: Mapping[str, str] | None = None,
) -> EvidenceBundle | None:
    """Return the recipe-lane bundle for one item, or None when no line reaches the markup threshold."""
    _ = dataset
    if detail.item_id != item_id:
        raise DataIntegrityError("[DATA] item detail identity mismatch")
    if detail.state not in allowed_states:
        return None
    procured_text = detail.name or ""
    if attributes:
        procured_text += " " + " ".join(str(v) for v in attributes.values())
    if mount_form_gate(recipe, procured_text) == "mount_form_mismatch":
        return None
    sourced = _fully_sourced(recipe, adder_evidence)
    seller_lane = bool(priced) and all(bool(p.part.sources) for p in priced)
    if seller_lane and recipe.adders and not sourced:
        raise DomainValidationError("[DATA] seller recipe adders require matching page evidence")
    if seller_lane and any(
        p.evidence.source_url not in p.part.sources
        or p.unit_price_krw != p.observed_price_krw
        or p.observed_price_krw != p.evidence.observed_price_krw
        for p in priced
    ):
        raise DomainValidationError("[DATA] seller recipe prices differ from observed evidence")
    if recipe.exact_sum:
        if recipe.adders or recipe.floor_krw is not None:
            raise DomainValidationError("[DATA] exact-sum recipe carries an adder or floor")
        benchmark = recipe_benchmark(priced, ())
        if any(p.unit_price_krw != p.observed_price_krw or p.observed_price_krw != p.evidence.observed_price_krw for p in priced):
            raise DomainValidationError("[DATA] exact-sum recipe prices differ from observed evidence")
    elif sourced:
        recipe_benchmark(priced, recipe.adders, recipe.floor_krw)
        benchmark = recipe_benchmark(priced, recipe.adders)
        observed_sum = sum(p.evidence.observed_price_krw * p.quantity for p in priced) + sum(
            e.observed_price_krw for e in adder_evidence
        )
        declared = {part.key: quantity for part, quantity in recipe.parts}
        if benchmark != observed_sum or any(
            p.quantity != declared.get(p.part.key)
            or p.unit_price_krw != p.observed_price_krw or p.observed_price_krw != p.evidence.observed_price_krw
            for p in priced
        ):
            raise DomainValidationError("[DATA] sourced recipe prices differ from observed evidence")
    else:
        benchmark = recipe_benchmark(priced, recipe.adders, recipe.floor_krw)
    procured = detail.name or item_id
    if sourced:
        expression = _sourced_sum_expression(priced, recipe.adders)
    else:
        expression = _sum_expression(priced) if recipe.exact_sum else ""
    # Organizer rules: a quote carries only what the page shows. The equivalence and the sum are the submitter's
    # reasoning, so they go to the evidence note (read by reviewers, not by the automatic check), never the quote.
    rationale = (
        f"조달 물품 「{procured}」(물품식별번호 {item_id})와 같은 모델이 아닌 동급 대체품 근거. "
        f"비교 단가 {benchmark:,}원{expression}의 구성 요소"
    )

    def noted(e: PriceEvidence, role: str) -> PriceEvidence:
        return replace(e, note=f"{role}. {rationale}"[:_NOTE_LIMIT])

    part_evidences = tuple(noted(p.evidence, f"동급 사양 비교의 {p.part.label} 판매가") for p in priced)
    sourced_adders = [a for a in recipe.adders if a.source_url is not None]
    adder_evidences = tuple(
        noted(e, f"{a.label}: 판매가가 아니라 주최 측이 공개한 정보통신 표준품셈 기준 공임")
        for a, e in zip(sourced_adders, adder_evidence, strict=False)
    )
    # The schema caps price_evidence at eight. Past the cap the smallest components stay in B without an evidence
    # object: that only raises B, so the declared saving stays below what the cited prices alone would give, and the
    # claim names each uncited amount with its public source.
    candidates = [(e, p.part.label) for p, e in zip(priced, part_evidences, strict=True)] + [
        (e, a.label) for a, e in zip(sourced_adders, adder_evidences, strict=False)
    ]
    by_size = sorted(range(len(candidates)), key=lambda i: candidates[i][0].observed_price_krw, reverse=True)
    cited_idx = sorted(by_size[:_MAX_PRICE_EVIDENCE])
    uncited = [candidates[i] for i in sorted(by_size[_MAX_PRICE_EVIDENCE:])]
    evidences = tuple(candidates[i][0] for i in cited_idx)
    lines: list[LineProposal] = []
    rejected: list[RejectedFact] = []
    gate_base = benchmark if seller_lane or recipe.floor_krw is None else max(benchmark, recipe.floor_krw)
    second_benchmark = _second_benchmark(priced, recipe.adders, None if seller_lane else recipe.floor_krw)
    band_dropped = 0
    for fact in detail.facts:
        if not fact.scorable:
            rejected.append(RejectedFact(fact.fact_id, "not_scorable"))
            continue
        if fact.period_kind != "none":
            rejected.append(RejectedFact(fact.fact_id, "period_not_none"))
            continue
        if fact.request_date is None:
            rejected.append(RejectedFact(fact.fact_id, "no_request_date"))
            continue
        if not _is_single_piece(fact.unit, evidence):
            rejected.append(RejectedFact(fact.fact_id, "unit_not_single_piece"))
            continue
        if sourced:
            with localcontext() as context:
                context.prec = _arithmetic_precision(evidence.recipe_markup_threshold, Decimal(gate_base))
                below_gate = fact.unit_price_krw < evidence.recipe_markup_threshold * Decimal(gate_base)
            if below_gate:
                rejected.append(RejectedFact(fact.fact_id, "below_floor_gate"))
                continue
        else:
            with localcontext() as context:
                context.prec = _arithmetic_precision(evidence.recipe_markup_threshold, Decimal(benchmark))
                below_threshold = fact.unit_price_krw < evidence.recipe_markup_threshold * Decimal(benchmark)
            if below_threshold:
                rejected.append(RejectedFact(fact.fact_id, "below_threshold"))
                continue
        with localcontext() as context:
            context.prec = _arithmetic_precision(evidence.recipe_band_upper, Decimal(benchmark))
            in_band = seller_lane and fact.unit_price_krw < evidence.recipe_band_upper * Decimal(benchmark)
        if in_band:
            if second_benchmark is None:
                rejected.append(RejectedFact(fact.fact_id, "band_single_seller"))
                band_dropped += 1
                continue
            with localcontext() as context:
                context.prec = _arithmetic_precision(evidence.recipe_markup_threshold, Decimal(second_benchmark))
                below_second = fact.unit_price_krw < evidence.recipe_markup_threshold * Decimal(second_benchmark)
            if below_second:
                rejected.append(RejectedFact(fact.fact_id, "band_second_seller"))
                band_dropped += 1
                continue
        try:
            lines.append(make_line(fact, Decimal(benchmark), evidences))
        except DomainValidationError:
            rejected.append(RejectedFact(fact.fact_id, "below_threshold"))
            continue
    if not lines:
        return None
    total = sum(line.expected_krw for line in lines)
    if total < evidence.min_item_expected_krw:
        return None
    best = max(line.fact_unit_price_krw / line.benchmark_unit_price_krw for line in lines)
    if sourced:
        # Seller recipes gate on B itself (no organizer floor), so the floor sentence would misstate the rule.
        gate_filtered = not seller_lane and any(r.reason == "below_floor_gate" for r in rejected)
        claim = _sourced_claim(item_id, priced, recipe.adders, benchmark, len(lines), gate_filtered, evidence)
    else:
        raised = recipe.floor_krw is not None and benchmark == recipe.floor_krw and (
            sum(p.unit_price_krw * p.quantity for p in priced) + sum(a.amount_krw for a in recipe.adders) < recipe.floor_krw
        )
        claim = _claim(item_id, priced, recipe.adders, benchmark, recipe.floor_krw, raised, len(lines), recipe.sum_label, recipe.exact_sum)
    if second_benchmark is not None:
        claim += f" 두 번째 판매처 기준 비교 단가 {second_benchmark:,}원."
    if band_dropped:
        claim += f" 1.2배 이상 1.3배 미만 납품요구 {band_dropped}건은 두 번째 판매처 기준을 넘지 못해 넣지 않았다."
    for cited_ev, label in uncited:
        claim += (
            f" {label} {cited_ev.observed_price_krw:,}원은 가격 근거 개수 한도(8개) 때문에 근거 객체 없이 비교 단가에 더했다"
            f"(출처 {cited_ev.source_url}). 이 금액만큼 신고 절감액이 근거 가격만으로 계산한 값보다 작다."
        )
    if len(claim) > 2000:
        raise DomainValidationError("[DATA] recipe claim exceeds text limit")
    notes = _notes(recipe, evidence, today)
    if second_benchmark is not None:
        notes += _second_seller_note(priced, recipe.adders, second_benchmark)
        if len(notes) > 4000:
            raise DomainValidationError("[DATA] recipe notes exceed text limit")
    bundle = EvidenceBundle(
        item_id=item_id,
        lane=Lane.RECIPE,
        title=f"{recipe.title_phrase} 대비 {best:.2f}배(동일 모델 아님)",
        claim=claim,
        comparison_scope=_scope(recipe, evidence),
        reproduction_notes=notes,
        lines=tuple(lines),
        rejected=tuple(rejected),
        caveats=(),
        review_required=recipe.review_required,
        generated_at=datetime.now(UTC),
    )
    logger.info(
        "[EVIDENCE] item_id=%s lane=recipe lines=%d expected_krw=%d benchmark_krw=%d",
        redact(item_id),
        len(lines),
        total,
        benchmark,
    )
    return bundle
