"""Retail same-model evidence lane over Danawa catalog and specialty shops."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import ROUND_CEILING, Decimal
from typing import Any
from urllib.parse import quote

from src.savetax.competition import ItemDetail, ItemState
from src.savetax.config import AuditConfig, EvidenceConfig
from src.savetax.danawa import (
    DanawaHit,
    DanawaProduct,
    MallOffer,
    model_pattern,
    parse_price_history,
    parse_product,
    parse_search,
)
from src.savetax.dataset import Dataset
from src.savetax.errors import DataIntegrityError, DomainValidationError
from src.savetax.evidence import EvidenceBundle, Lane, LineProposal, PriceEvidence, RejectedFact, make_line
from src.savetax.publicweb import PublicFetcher
from src.savetax.rawapi import RawApiClient
from src.savetax.redaction import redact
from src.savetax.shops import ShopProduct, parse_shop_product, parse_shop_search, shop_search_url

logger = logging.getLogger(__name__)

__all__ = [
    "brand_matches",
    "build_retail_bundle",
    "current_benchmark",
    "is_accessory_or_pack",
]

_CLAIM = "납품요구 기록의 단가가 동일 모델의 온라인 소매 최저가보다 높게 나타난다."
_SCOPE = "다나와 카탈로그 또는 전문점의 동일 모델 소매가와 요청일 기준으로 비교하였다."

_COMPANY_TOKENS = frozenset(
    {
        "주식회사",
        "유한회사",
        "주식",
        "회사",
        "주",
        "유",
        "사",
        "CORPORATION",
        "LIMITED",
        "COMPANY",
        "CORP",
        "LTD",
        "INC",
        "CO",
        "LLC",
        "GMBH",
        "PTE",
    }
)


def _normalize(value: str, evidence: EvidenceConfig) -> str:
    text = value.upper()
    text = text.replace("(주문자상표부착)", " ").replace("주문자상표부착", " ")
    for token in ("주식회사", "유한회사", "(주)", "(유)", "㈜"):
        text = text.replace(token, " ")
    for korean, latin in evidence.brand_aliases:
        if korean:
            text = text.replace(korean, latin.upper())
    for token in ("전자", "코리아", "KOREA", "ELECTRONICS"):
        text = text.replace(token, "")
    text = re.sub(r"[^A-Z0-9가-힣]+", " ", text)
    return "".join(part for part in text.split() if part not in _COMPANY_TOKENS)


def brand_matches(manufacturer: str, product_name: str, brand: str | None, evidence: EvidenceConfig) -> bool:
    """Return True when the procurement manufacturer and the catalog brand name the same maker."""
    if not manufacturer or not manufacturer.strip():
        return False
    if brand is None or not brand.strip():
        return False
    maker = _normalize(manufacturer, evidence)
    catalog = _normalize(brand, evidence)
    if not maker or not catalog:
        return False
    page = _normalize(product_name, evidence)
    head_maker = maker[:4]
    if head_maker and (head_maker in catalog or head_maker in page):
        return True
    head_brand = catalog[:4]
    return bool(head_brand and head_brand in maker)


def is_accessory_or_pack(name: str, description: str, category: str | None, evidence: EvidenceConfig) -> bool:
    """Return True when the product is an accessory, consumable, used item or multi-pack."""
    for marker in evidence.accessory_markers:
        if marker and (marker in name or marker in description[:40] or (category is not None and marker in category)):
            return True
    return any(pattern and re.search(pattern, name) for pattern in evidence.pack_markers)


def _ordered_offers(product: DanawaProduct) -> list[MallOffer]:
    return sorted(product.offers, key=lambda offer: (offer.price_krw, offer.mall))


def current_benchmark(
    product: DanawaProduct,
    history: Mapping[str, int],
    window: tuple[str, str],
    evidence: EvidenceConfig,
) -> int | None:
    """Return the supported current unit price including shipping, or None when support is insufficient."""
    if product.low_price_krw is None:
        return None
    offers = _ordered_offers(product)
    if not offers:
        return None
    if len(offers) >= 2:
        lowest, second = offers[0], offers[1]
        if Decimal(second.price_krw) <= Decimal(lowest.price_krw) * evidence.outlier_ratio:
            return lowest.price_krw + (lowest.shipping_krw or 0)
        return second.price_krw + (second.shipping_krw or 0)
    lowest_total = offers[0].price_krw + (offers[0].shipping_krw or 0)
    values = [value for key, value in history.items() if window[0] <= key <= window[1]]
    if not values:
        return None
    if Decimal(max(values)) <= Decimal(min(values)) * evidence.single_offer_stability_ratio:
        return lowest_total
    return None


def _core_region(value: str | None) -> str | None:
    if value is None:
        return None
    core = value.split("(", 1)[0].strip()
    return core or None


def _ceil_amount(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def _installation(
    fact_price: Decimal,
    order_terms: str,
    b_time: int,
    order_details: list[Any],
    evidence: EvidenceConfig,
) -> tuple[int, bool]:
    if order_terms not in evidence.install_terms:
        return 0, False
    install_prices = [
        detail.price_krw
        for detail in order_details
        if detail is not None and (detail.delivery_terms or "").strip() in evidence.install_terms
    ]
    sibling_prices = [
        detail.price_krw
        for detail in order_details
        if detail is not None
        and (detail.delivery_terms or "").strip()
        and (detail.delivery_terms or "").strip() not in evidence.install_terms
    ]
    if install_prices and sibling_prices:
        install_price = fact_price if fact_price in install_prices else max(install_prices)
        delta = _ceil_amount(install_price - min(sibling_prices))
        if delta >= 0:
            return delta, False
    return _ceil_amount(Decimal(b_time) * evidence.install_allowance_rate), True


def _order_rows(dataset: Dataset, item_id: str, contract_no: str) -> list[tuple[str, Any]]:
    return dataset.connection.execute(
        "SELECT listing_id, price FROM listings WHERE item_id = ? AND contract_no = ?",
        [item_id, contract_no],
    ).fetchall()


def _resolve_order(
    dataset: Dataset,
    raw: RawApiClient,
    audit: AuditConfig,
    item_id: str,
    fact: Any,
) -> tuple[str, str, list[Any]] | tuple[RejectedFact, None, None]:
    rows = _order_rows(dataset, item_id, fact.contract_no)
    if not rows:
        return RejectedFact(fact.fact_id, "order_listing_not_served"), None, None
    if len(rows) > audit.max_condition_listings:
        return RejectedFact(fact.fact_id, "too_many_order_listings"), None, None
    details: list[Any] = [raw.get_listing(lid) for lid, _ in rows]
    for (lid, _), detail in zip(rows, details, strict=True):
        if (
            detail is None
            or detail.listing_id != lid
            or detail.item_id != item_id
            or detail.contract_no != fact.contract_no
        ):
            return RejectedFact(fact.fact_id, "order_listing_not_served"), None, None
    matched = [
        lid
        for (lid, _), detail in zip(rows, details, strict=True)
        if detail is not None and detail.price_krw == fact.unit_price_krw
    ]
    wanted = set(matched) if matched else {lid for lid, _ in rows}
    selected = [detail for (lid, _), detail in zip(rows, details, strict=True) if lid in wanted]
    if any(
        detail is None or detail.price_kind == "catalog_reference" or not (detail.delivery_terms or "").strip() or not _core_region(detail.supply_region)
        for detail in selected
    ):
        return RejectedFact(fact.fact_id, "order_listing_not_served"), None, None
    pairs = {
        (str(detail.delivery_terms).strip(), _core_region(detail.supply_region) or "")
        for detail in selected
        if detail is not None
    }
    if len(pairs) != 1:
        return RejectedFact(fact.fact_id, "order_conditions_ambiguous"), None, None
    terms, _core = next(iter(pairs))
    compatible = [detail for detail in details if detail is not None and detail.unit == fact.unit
                  and detail.price_kind != "catalog_reference" and _core_region(detail.supply_region) == _core]
    return terms, _core, compatible


def _seller_text(offer: MallOffer) -> str:
    if offer.shipping_krw == 0:
        ship = "무료배송"
    elif offer.shipping_krw is None:
        ship = "배송비 미표기"
    else:
        ship = f"배송비 {offer.shipping_krw:,}원"
    return f"{offer.mall} {offer.price_krw:,}원({ship})"


def _catalog_quote(product: DanawaProduct, category: str | None, today: date) -> str:
    parts = [product.name or "상품"]
    if product.registered_month:
        parts.append(f"등록월 {product.registered_month}")
    if product.brand:
        parts.append(f"제조사 {product.brand}")
    if product.low_price_krw is not None:
        parts.append(f"최저가 {product.low_price_krw:,}원")
    count = product.offer_count if product.offer_count is not None else len(product.offers)
    parts.append(f"판매자 {count}곳")
    parts.extend(_seller_text(offer) for offer in _ordered_offers(product)[:3])
    parts.append(f"다나와 상품 페이지, {today.isoformat()} 확인")
    return " / ".join(parts)


def _chosen_offer(product: DanawaProduct, evidence: EvidenceConfig) -> MallOffer | None:
    offers = _ordered_offers(product)
    if not offers:
        return None
    if len(offers) >= 2:
        lowest, second = offers[0], offers[1]
        if Decimal(second.price_krw) <= Decimal(lowest.price_krw) * evidence.outlier_ratio:
            return lowest
        return second
    return offers[0]


def _second_used(product: DanawaProduct, evidence: EvidenceConfig) -> bool:
    offers = _ordered_offers(product)
    if len(offers) < 2:
        return False
    return Decimal(offers[1].price_krw) > Decimal(offers[0].price_krw) * evidence.outlier_ratio


def build_retail_bundle(
    detail: ItemDetail,
    dataset: Dataset,
    raw: RawApiClient,
    fetcher: PublicFetcher,
    evidence: EvidenceConfig,
    audit: AuditConfig,
    *,
    today: date,
) -> EvidenceBundle | None:
    """Return the retail bundle for one item, or None when no line survives the gates."""
    if detail.state is not ItemState.AVAILABLE:
        return None
    model = (detail.model or "").strip()
    if not model or redact(model) != model:
        return None
    try:
        pattern = model_pattern(model)
    except DomainValidationError:
        return None
    manufacturer = detail.manufacturer or ""
    search_model = re.sub(r"^\([A-Z]{2}\)", "", model)
    search_url = evidence.danawa_search_url_template.format(query=quote(search_model, safe=""))
    hits: list[DanawaHit] = []
    page = fetcher.get(search_url)
    if page.status == 200:
        hits = [
            hit
            for hit in parse_search(page.text)
            if pattern.search(hit.name) and not is_accessory_or_pack(hit.name, "", hit.category, evidence)
        ]
    hits.sort(key=lambda hit: (hit.price_krw is None, hit.price_krw or 0, hit.pcode))
    if hits:
        hit = hits[0]
        return _catalog_bundle(
            detail, dataset, raw, fetcher, evidence, audit, today, pattern, manufacturer, hit.pcode, hit.category
        )
    return _shop_bundle(detail, dataset, raw, fetcher, evidence, audit, today, model, pattern)


def _catalog_bundle(
    detail: ItemDetail,
    dataset: Dataset,
    raw: RawApiClient,
    fetcher: PublicFetcher,
    evidence: EvidenceConfig,
    audit: AuditConfig,
    today: date,
    pattern: re.Pattern[str],
    manufacturer: str,
    pcode: str,
    category: str | None,
) -> EvidenceBundle | None:
    catalog_url = evidence.danawa_product_url_template.format(pcode=pcode)
    page = fetcher.get(catalog_url)
    if page.status != 200:
        return None
    try:
        product = parse_product(page.text, pcode)
    except DataIntegrityError:
        return None
    if not pattern.search(product.name or ""):
        return None
    if is_accessory_or_pack(product.name or "", product.description or "", category, evidence):
        return None
    history_url = evidence.danawa_history_url_template.format(pcode=pcode)
    history: dict[str, int] = {}
    hist_page = fetcher.get(history_url, referer=catalog_url)
    if hist_page.status == 200:
        try:
            history = parse_price_history(hist_page.text)
        except DataIntegrityError:
            history = {}
    months = sorted({fact.request_date.strftime("%Y-%m") for fact in detail.facts if fact.scorable and fact.request_date is not None})
    window: tuple[str, str] = (months[0], months[-1]) if months else ("9999-99", "0000-00")
    bench = current_benchmark(product, history, window, evidence)
    brand_ok = brand_matches(manufacturer, product.name or "", product.brand, evidence)
    review = not brand_ok or len(product.offers) < evidence.min_offer_count
    if product.low_price_krw is None or product.low_price_krw <= 0:
        return None
    chosen = _chosen_offer(product, evidence)
    second_fallback = _second_used(product, evidence)
    unknown_shipping = chosen is not None and chosen.shipping_krw is None
    lowest = next(iter(_ordered_offers(product)), None)
    shipping_included = lowest is not None and lowest.shipping_krw == 0
    quote = _catalog_quote(product, category, today)
    lines: list[LineProposal] = []
    rejected: list[RejectedFact] = []
    caveats: list[str] = []
    if second_fallback:
        caveats.append("차순위 저가 사용: 최저가 행이 이상치로 판단되어 2순위 행을 기준으로 하였다")
    if unknown_shipping:
        caveats.append("배송비 미표기: 최저 행의 배송비가 확인되지 않아 0원으로 계산하였다")
    allowance_used = False
    for fact in detail.facts:
        if not fact.scorable:
            rejected.append(RejectedFact(fact.fact_id, "not_scorable"))
            continue
        if fact.request_date is None:
            rejected.append(RejectedFact(fact.fact_id, "no_request_date"))
            continue
        if fact.unit not in evidence.single_piece_units:
            rejected.append(RejectedFact(fact.fact_id, "unit_not_single_piece"))
            continue
        if bench is None:
            rejected.append(RejectedFact(fact.fact_id, "unsupported_price"))
            continue
        month_key = fact.request_date.strftime("%Y-%m")
        if month_key not in history:
            rejected.append(RejectedFact(fact.fact_id, "no_history_for_month"))
            continue
        b_time = bench if bench >= history[month_key] else history[month_key]
        resolved = _resolve_order(dataset, raw, audit, detail.item_id, fact)
        if isinstance(resolved[0], RejectedFact):
            rejected.append(resolved[0])
            continue
        order_terms = resolved[0]
        order_details = resolved[2]
        assert isinstance(order_terms, str)
        assert order_details is not None
        installation, used_allowance = _installation(fact.unit_price_krw, order_terms, b_time, order_details, evidence)
        allowance_used = allowance_used or used_allowance
        total = b_time + installation
        if fact.unit_price_krw < Decimal(total) * evidence.retail_markup_threshold:
            rejected.append(RejectedFact(fact.fact_id, "no_margin"))
            continue
        if not brand_ok and fact.unit_price_krw > Decimal(total) * evidence.oem_gap_cap:
            rejected.append(RejectedFact(fact.fact_id, "implausible_gap"))
            continue
        try:
            proof = PriceEvidence(
                source_url=catalog_url,
                quote=quote,
                observed_price_krw=int(product.low_price_krw),
                observed_at=today,
                vat_included=True,
                shipping_included=shipping_included,
            )
            lines.append(make_line(fact, Decimal(total), [proof]))
        except DomainValidationError:
            rejected.append(RejectedFact(fact.fact_id, "no_margin"))
            continue
    if allowance_used:
        caveats.append(f"설치 허용률 적용: 형제 등재가 없어 허용률 {evidence.install_allowance_rate}을 적용하였다")
    if not lines:
        return None
    ratios = [line.fact_unit_price_krw / line.benchmark_unit_price_krw for line in lines]
    best = max(ratios)
    notes = (
        f"다나와 상품 페이지({catalog_url})와 가격 히스토리({history_url})의 "
        f"{', '.join(months) if months else '월 정보 없음'} 최저가를 확인하였다"
    )
    total_expected = sum(line.expected_krw for line in lines)
    logger.info(
        "[EVIDENCE] item_id=%s lane=retail lines=%d review=%s expected_krw=%d",
        redact(detail.item_id),
        len(lines),
        review,
        total_expected,
    )
    return EvidenceBundle(
        item_id=detail.item_id,
        lane=Lane.RETAIL,
        title=f"온라인 최저가 대비 {best:.2f}배",
        claim=_CLAIM,
        comparison_scope=_SCOPE,
        reproduction_notes=notes,
        lines=tuple(lines),
        rejected=tuple(rejected),
        caveats=tuple(caveats),
        review_required=review,
        generated_at=datetime.now(UTC),
    )


def _shop_bundle(
    detail: ItemDetail,
    dataset: Dataset,
    raw: RawApiClient,
    fetcher: PublicFetcher,
    evidence: EvidenceConfig,
    audit: AuditConfig,
    today: date,
    model: str,
    pattern: re.Pattern[str],
) -> EvidenceBundle | None:
    for shop in evidence.shops:
        search_page = fetcher.get(shop_search_url(shop, model))
        if search_page.status != 200:
            continue
        for link in parse_shop_search(search_page.text, shop):
            product_page = fetcher.get(link)
            if product_page.status != 200:
                continue
            shop_product = parse_shop_product(product_page.text, link)
            if shop_product is None:
                continue
            if not pattern.search(shop_product.title):
                continue
            if is_accessory_or_pack(shop_product.title, "", None, evidence):
                continue
            bundle = _shop_product_bundle(detail, dataset, raw, evidence, audit, today, shop_product)
            if bundle is not None and bundle.lines:
                return bundle
    logger.info(
        "[EVIDENCE] item_id=%s lane=retail lines=0 review=False expected_krw=0",
        redact(detail.item_id),
    )
    return None


def _shop_product_bundle(
    detail: ItemDetail,
    dataset: Dataset,
    raw: RawApiClient,
    evidence: EvidenceConfig,
    audit: AuditConfig,
    today: date,
    shop_product: ShopProduct,
) -> EvidenceBundle | None:
    lines: list[LineProposal] = []
    rejected: list[RejectedFact] = []
    caveats: list[str] = []
    allowance_used = False
    for fact in detail.facts:
        if not fact.scorable:
            rejected.append(RejectedFact(fact.fact_id, "not_scorable"))
            continue
        if fact.request_date is None:
            rejected.append(RejectedFact(fact.fact_id, "no_request_date"))
            continue
        if fact.unit not in evidence.single_piece_units:
            rejected.append(RejectedFact(fact.fact_id, "unit_not_single_piece"))
            continue
        b_time = int(shop_product.price_krw)
        resolved = _resolve_order(dataset, raw, audit, detail.item_id, fact)
        if isinstance(resolved[0], RejectedFact):
            rejected.append(resolved[0])
            continue
        order_terms = resolved[0]
        order_details = resolved[2]
        assert isinstance(order_terms, str)
        assert order_details is not None
        installation, used_allowance = _installation(fact.unit_price_krw, order_terms, b_time, order_details, evidence)
        allowance_used = allowance_used or used_allowance
        total = b_time + installation
        if fact.unit_price_krw < Decimal(total) * evidence.retail_markup_threshold:
            rejected.append(RejectedFact(fact.fact_id, "no_margin"))
            continue
        quote = (
            f"{shop_product.title} / 판매가 {shop_product.price_krw:,}원 / 전문점 상품 페이지, {today.isoformat()} 확인"
        )
        try:
            proof = PriceEvidence(
                source_url=shop_product.url,
                quote=quote,
                observed_price_krw=int(shop_product.price_krw),
                observed_at=today,
                vat_included=True,
                shipping_included=False,
            )
            lines.append(make_line(fact, Decimal(total), [proof]))
        except DomainValidationError:
            rejected.append(RejectedFact(fact.fact_id, "no_margin"))
            continue
    if allowance_used:
        caveats.append(f"설치 허용률 적용: 형제 등재가 없어 허용률 {evidence.install_allowance_rate}을 적용하였다")
    if not lines:
        return None
    ratios = [line.fact_unit_price_krw / line.benchmark_unit_price_krw for line in lines]
    best = max(ratios)
    total_expected = sum(line.expected_krw for line in lines)
    logger.info(
        "[EVIDENCE] item_id=%s lane=retail lines=%d review=%s expected_krw=%d",
        redact(detail.item_id),
        len(lines),
        True,
        total_expected,
    )
    return EvidenceBundle(
        item_id=detail.item_id,
        lane=Lane.RETAIL,
        title=f"온라인 최저가 대비 {best:.2f}배",
        claim=_CLAIM,
        comparison_scope=_SCOPE,
        reproduction_notes=f"전문점 상품 페이지({shop_product.url})의 판매가를 확인하였다",
        lines=tuple(lines),
        rejected=tuple(rejected),
        caveats=tuple(caveats),
        review_required=True,
        generated_at=datetime.now(UTC),
    )
