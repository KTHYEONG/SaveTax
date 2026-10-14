"""Endpoints, thresholds, and credential loading."""

from __future__ import annotations

import enum
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import NoReturn, SupportsIndex
from urllib.parse import quote, unquote, urlparse

from src.savetax.errors import ConfigError, CredentialError

logger = logging.getLogger(__name__)

_DATASET_DIR_PREFIX = "savetax-products-"
_TOKEN_PREFIX = "savetax_"  # noqa: S105
_FORBIDDEN_VALUE_CHARS = frozenset("\"'<>")
_ZIP_URL = "https://savetax-dataset.reform-dev-infra.workers.dev/latest/savetax-products.zip"

__all__ = ["DEFAULT_DESKTOP_PC_REFS", "DEFAULT_EVIDENCE_HOSTS", "DEFAULT_RACK_CLASSES", "DEFAULT_RECIPE_ADDERS", "DEFAULT_RECIPE_PARTS", "DEFAULT_SERIES_BLOCKS", "ApprovalConfig", "AuditConfig", "DesktopPcRef", "EvidenceConfig", "ListingSeries", "RackClass", "RecipeAdder", "RecipePart", "SaveTaxConfig", "Secret", "SeriesBlock", "ShopSpec", "UnitConflictPolicy", "VariantPolicy", "load_token"]


class Secret:
    """Opaque holder for the participant token; never renders its value."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """Return the underlying token value."""
        return self._value

    def __repr__(self) -> str:
        return "Secret(***)"

    def __str__(self) -> str:
        return "Secret(***)"

    def __format__(self, _spec: str) -> str:
        return "Secret(***)"

    def __reduce__(self) -> NoReturn:
        raise TypeError("Secret cannot be pickled")

    def __reduce_ex__(self, _protocol: SupportsIndex) -> NoReturn:
        raise TypeError("Secret cannot be pickled")


def _check_url(value: str, field_name: str) -> None:
    if any(c.isspace() or ord(c) < 32 for c in value) or "\\" in value:
        raise ConfigError(f"[SYS] invalid {field_name}: unsafe URL characters")
    try:
        _check_parsed_url(value, field_name)
    except ValueError:
        raise ConfigError(f"[SYS] invalid {field_name}: malformed URL") from None


def _check_parsed_url(value: str, field_name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ConfigError(f"[SYS] invalid {field_name}: must be an https URL")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError(f"[SYS] invalid {field_name}: userinfo is forbidden")
    if "#" in value:
        raise ConfigError(f"[SYS] invalid {field_name}: fragment is forbidden")
    if parsed.port not in (None, 443):
        raise ConfigError(f"[SYS] invalid {field_name}: unsupported HTTPS port")
    opaque_prefix = "/v1/events/current/submission-requests/"
    if parsed.hostname == "hackathon.savetax.now" and parsed.path.startswith(opaque_prefix):
        encoded = parsed.path[len(opaque_prefix) :]
        key = unquote(encoded)
        if (
            not 1 <= len(key) <= 128
            or any(not 33 <= ord(char) <= 126 for char in key)
            or encoded != quote(key, safe="").replace(".", "%2E")
            or parsed.query
        ):
            raise ConfigError(f"[SYS] invalid {field_name}: malformed request key path")
        return
    path = unquote(parsed.path)
    if ".." in path.split("/") or "%" in path or "\\" in path:
        raise ConfigError(f"[SYS] invalid {field_name}: dot-dot path segment is forbidden")


@dataclass(frozen=True)
class SaveTaxConfig:
    """Endpoints, timeouts, and local paths."""

    competition_base_url: str = "https://hackathon.savetax.now/v1"
    raw_api_base_url: str = "https://api.savetax.now/v1"
    dataset_zip_url: str = "https://savetax-dataset.reform-dev-infra.workers.dev/latest/savetax-products.zip"
    guidance_manifest_url: str = "https://hackathon.savetax.now/docs/agent-guidance-version.json"
    user_agent: str = "Mozilla/5.0"
    timeout_seconds: float = 30.0
    retry_cooldown_seconds: float = 15.0
    rate_limit_max_wait_seconds: float = 60.0
    dataset_dir: Path = field(default_factory=lambda: Path("data/savetax-products-20261010-r5"))
    candidates_dir: Path = field(default_factory=lambda: Path("scratch/candidates"))
    participant_handle: str = ""
    token_env_var: str = "SAVETAX_TOKEN"  # noqa: S105
    dotenv_path: Path = field(default_factory=lambda: Path(".env"))
    dotenv_keys: tuple[str, ...] = ("SAVETAX_TOKEN", "tok")  # noqa: S105
    delivery_page_size: int = 200
    delivery_rows_max: int = 4000

    def __post_init__(self) -> None:
        _check_url(self.competition_base_url, "competition_base_url")
        _check_url(self.raw_api_base_url, "raw_api_base_url")
        _check_url(self.dataset_zip_url, "dataset_zip_url")
        _check_url(self.guidance_manifest_url, "guidance_manifest_url")
        if not self.user_agent:
            raise ConfigError("[SYS] invalid user_agent: must be non-empty")
        if not math.isfinite(self.timeout_seconds) or not self.timeout_seconds > 0:
            raise ConfigError("[SYS] invalid timeout_seconds: must be > 0")
        if not math.isfinite(self.retry_cooldown_seconds) or not self.retry_cooldown_seconds > 0:
            raise ConfigError("[SYS] invalid retry_cooldown_seconds: must be > 0")
        if not math.isfinite(self.rate_limit_max_wait_seconds) or not self.rate_limit_max_wait_seconds > 0:
            raise ConfigError("[SYS] invalid rate_limit_max_wait_seconds: must be > 0")
        if not self.dotenv_keys:
            raise ConfigError("[SYS] invalid dotenv_keys: must be non-empty")
        if isinstance(self.delivery_page_size, bool) or not isinstance(self.delivery_page_size, int):
            raise ConfigError("[SYS] invalid delivery_page_size: expected integer")
        if isinstance(self.delivery_rows_max, bool) or not isinstance(self.delivery_rows_max, int):
            raise ConfigError("[SYS] invalid delivery_rows_max: expected integer")
        if not 1 <= self.delivery_page_size <= 200:
            raise ConfigError("[SYS] invalid delivery_page_size: must be 1..200")
        if self.delivery_rows_max < self.delivery_page_size:
            raise ConfigError("[SYS] invalid delivery_rows_max: must be >= delivery_page_size")

    @property
    def dataset_version(self) -> str:
        """Return the dataset release suffix encoded in the directory name."""
        name = self.dataset_dir.name
        if not name.startswith(_DATASET_DIR_PREFIX):
            raise ConfigError("[SYS] invalid dataset_dir: missing savetax-products- prefix")
        suffix = name[len(_DATASET_DIR_PREFIX) :]
        if not suffix:
            raise ConfigError("[SYS] invalid dataset_dir: empty version suffix")
        return suffix

    @property
    def dataset_parquet_path(self) -> Path:
        """Return the listings parquet path inside the dataset directory."""
        return self.dataset_dir / "listings.parquet"

    def allows_credential(self, url: str) -> bool:
        """Return True only for the three protected credential destinations."""
        try:
            _check_url(url, "credential destination")
        except ConfigError:
            return False
        if url == self.dataset_zip_url:
            return url == _ZIP_URL
        official = url.startswith("https://hackathon.savetax.now/v1/") or url.startswith("https://api.savetax.now/v1/")
        return official and (
            url.startswith(self.competition_base_url + "/") or url.startswith(self.raw_api_base_url + "/")
        )


@dataclass(frozen=True)
class AuditConfig:
    """Screening thresholds."""

    markup_threshold: Decimal = Decimal("1.2")
    min_item_order_amount_krw: int = 10_000_000
    min_unit_price_krw: Decimal = Decimal("100")
    comparable_large_categories: tuple[str, ...] = ("전자/정보/통신/영상", "소프트웨어")
    generic_manufacturers: tuple[str, ...] = ("조합공통품목", "일반형")
    min_model_alnum_length: int = 6
    require_reseller: bool = True
    batch_size: int = 500
    max_condition_listings: int = 8
    progress_every: int = 50

    def __post_init__(self) -> None:
        if not self.markup_threshold.is_finite() or self.markup_threshold < 1:
            raise ConfigError("[SYS] invalid markup_threshold: must be >= 1")
        if self.min_item_order_amount_krw < 0:
            raise ConfigError("[SYS] invalid min_item_order_amount_krw: must be >= 0")
        if not self.min_unit_price_krw.is_finite() or self.min_unit_price_krw <= 0:
            raise ConfigError("[SYS] invalid min_unit_price_krw: must be > 0")
        if self.min_model_alnum_length < 1:
            raise ConfigError("[SYS] invalid min_model_alnum_length: must be >= 1")
        if self.batch_size < 1:
            raise ConfigError("[SYS] invalid batch_size: must be >= 1")
        if self.max_condition_listings < 1:
            raise ConfigError("[SYS] invalid max_condition_listings: must be >= 1")
        if self.progress_every < 1:
            raise ConfigError("[SYS] invalid progress_every: must be >= 1")
        if not self.comparable_large_categories:
            raise ConfigError("[SYS] invalid comparable_large_categories: must be non-empty")


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return values
    except (OSError, UnicodeError):
        raise CredentialError("[SYS] unreadable participant token source (source: dotenv)") from None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1].strip()
        if key and key not in values:
            values[key] = val
    return values


def _valid_token(value: str) -> bool:
    if not value.startswith(_TOKEN_PREFIX) or len(value) <= len(_TOKEN_PREFIX):
        return False
    return not any(c.isspace() or c in _FORBIDDEN_VALUE_CHARS for c in value)


def load_token(config: SaveTaxConfig, environ: Mapping[str, str]) -> Secret:
    """Return the participant token from environ or the dotenv file."""
    env_value = environ.get(config.token_env_var, "")
    if env_value:
        if not _valid_token(env_value):
            raise CredentialError("[SYS] invalid participant token (source: env)")
        return Secret(env_value)
    dotenv_values = _parse_dotenv(config.dotenv_path)
    for key in config.dotenv_keys:
        if key not in dotenv_values:
            continue
        candidate = dotenv_values[key]
        if not _valid_token(candidate):
            raise CredentialError("[SYS] invalid participant token (source: dotenv)")
        return Secret(candidate)
    raise CredentialError("[SYS] missing participant token (source: env/dotenv)")


@dataclass(frozen=True)
class ShopSpec:
    """One retail shop searchable by model: host and storefront platform ("cafe24" or "godomall")."""

    host: str
    platform: str


_TEMPLATE_PLACEHOLDERS: tuple[tuple[str, str], ...] = (
    ("listing_page_url_template", "{listing_id}"),
    ("danawa_search_url_template", "{query}"),
    ("danawa_product_url_template", "{pcode}"),
    ("danawa_history_url_template", "{pcode}"),
)

_CREDENTIAL_HOSTS = frozenset({"hackathon.savetax.now", "api.savetax.now"})


class UnitConflictPolicy(enum.StrEnum):
    """How to handle a display-unit conflict on a listing page."""

    ANNOTATE = "annotate"
    REJECT = "reject"


class VariantPolicy(enum.StrEnum):
    """How to gate benchmarks when an item carries hidden variants."""

    SERIES = "series"
    STRICT = "strict"
    ALLOW_LOW_FACT = "allow_low_fact"
    OFF = "off"


class ListingSeries(enum.StrEnum):
    """Product line a listing belongs to inside its contract."""

    GENERAL = "general"
    SEISMIC = "seismic"


@dataclass(frozen=True)
class SeriesBlock:
    """Inclusive listing-sequence range of one contract that belongs to one series."""

    contract_no: str
    first_sequence: int
    last_sequence: int
    series: ListingSeries

    def __post_init__(self) -> None:
        if not self.contract_no.strip():
            raise ConfigError("[SYS] invalid series_blocks: contract_no must be non-empty")
        if isinstance(self.first_sequence, bool) or isinstance(self.last_sequence, bool):
            raise ConfigError("[SYS] invalid series_blocks: sequences must be integers")
        if not isinstance(self.first_sequence, int) or not isinstance(self.last_sequence, int):
            raise ConfigError("[SYS] invalid series_blocks: sequences must be integers")
        if not 1 <= self.first_sequence <= self.last_sequence:
            raise ConfigError("[SYS] invalid series_blocks: need 1 <= first_sequence <= last_sequence")
        try:
            object.__setattr__(self, "series", ListingSeries(self.series))
        except ValueError:
            raise ConfigError("[SYS] invalid series_blocks: unknown series") from None


# Contract spec sheets and listing layout, 2026-10-13.
DEFAULT_SERIES_BLOCKS: tuple[SeriesBlock, ...] = (
    SeriesBlock("122010318_1", 1070000272, 1070000660, ListingSeries.GENERAL),
    SeriesBlock("122010318_1", 1070000673, 1070001037, ListingSeries.SEISMIC),
    SeriesBlock("331930619_1", 1190000234, 1190000449, ListingSeries.GENERAL),
    SeriesBlock("331930619_1", 1190000455, 1190000654, ListingSeries.SEISMIC),
    SeriesBlock("331930619_1", 1190000655, 1190000694, ListingSeries.GENERAL),
    SeriesBlock("331930619_1", 1190000695, 1190000744, ListingSeries.SEISMIC),
    SeriesBlock("231732062_1", 1230000327, 1230000675, ListingSeries.GENERAL),
    SeriesBlock("231732062_1", 1230000788, 1230000877, ListingSeries.GENERAL),
    SeriesBlock("231732062_1", 1230000878, 1230000942, ListingSeries.SEISMIC),
    SeriesBlock("381730406_1", 1140000082, 1140000129, ListingSeries.GENERAL),
    SeriesBlock("381730407_1", 1140000128, 1140000285, ListingSeries.GENERAL),
    SeriesBlock("382030583_1", 1070000306, 1070000406, ListingSeries.GENERAL),
    SeriesBlock("382030583_1", 1070000660, 1070000675, ListingSeries.SEISMIC),
    SeriesBlock("382030583_1", 1070000682, 1070000702, ListingSeries.GENERAL),
    SeriesBlock("382330004_1", 1030000044, 1030000195, ListingSeries.GENERAL),
    SeriesBlock("221730822_1", 1200000091, 1200000254, ListingSeries.GENERAL),
    SeriesBlock("221730822_1", 1200000277, 1200000481, ListingSeries.SEISMIC),
)


@dataclass(frozen=True)
class RackClass:
    """Reference rack for procurement racks up to a height; floor is the organizer's highest published B of the class."""

    max_height_mm: int
    part_key: str
    floor_krw: int


DEFAULT_RACK_CLASSES: tuple["RackClass", ...] = (  # noqa: UP037
    RackClass(1200, "rack:1200", 563_310),
    RackClass(1400, "rack:1400", 696_310),
    RackClass(1600, "rack:1600", 937_310),
    RackClass(1800, "rack:1800", 866_310),
    RackClass(2000, "rack:2000", 965_310),
)


_PCODE_RE = re.compile(r"^\d{1,12}$")


@dataclass(frozen=True)
class RecipePart:
    """One reference product of a recipe: seller page URLs and the tokens its page title must contain."""

    key: str
    label: str
    sources: tuple[str, ...] = ()
    name_tokens: tuple[str, ...] = ()
    mount_form: str | None = None
    pcode: str = ""
    sources_pending: bool = False

    def __post_init__(self) -> None:
        sources = self.sources
        pcode = self.pcode
        if isinstance(sources, str):
            if _PCODE_RE.fullmatch(sources):
                pcode = pcode or sources
                sources = ()
            else:
                sources = (sources,)
            object.__setattr__(self, "sources", sources)
            object.__setattr__(self, "pcode", pcode)
        elif isinstance(sources, list):
            object.__setattr__(self, "sources", tuple(sources))
        if isinstance(self.name_tokens, list):
            object.__setattr__(self, "name_tokens", tuple(self.name_tokens))


@dataclass(frozen=True)
class RecipeAdder:
    """A fixed KRW amount added to the benchmark, with its published basis and optional public source page."""

    key: str
    label: str
    amount_krw: int
    basis: str
    source_url: str | None = None


@dataclass(frozen=True)
class DesktopPcRef:
    """Finished reference PC for one CPU class; chipset is the reference mainboard class."""

    cpu_key: str
    part_key: str
    chipset: str



_REJECTED_PART_HOSTS = frozenset({"prod.danawa.com", "search.danawa.com", "shop.danawa.com", "price.enuri.com"})

DEFAULT_EVIDENCE_HOSTS: tuple[str, ...] = (
    "oversystem.co.kr",
    "cs-office.net",
    "pctopia.co.kr",
    "cynexpc.com",
    "kingdommol.com",
    "duracom.co.kr",
    "picknmall.co.kr",
    "compuzone.co.kr",
    "yopc.co.kr",
    "fibermart.co.kr",
    "rftown.net",
    "netzone.co.kr",
    "www.compuzone.co.kr",
    "www.yopc.co.kr",
    "www.cynexpc.com",
    "www.kingdommol.com",
    "www.duracom.co.kr",
    "www.picknmall.co.kr",
    "usr.icoda.co.kr",
    "www.suzakpc.co.kr",
    "comoasis.co.kr",
    "www.pc4all.co.kr",
    "www.lanstar.co.kr",
    "www.devicemart.co.kr",
    "comvia.co.kr",
    "www.rftown.net",
    "www.netzone.co.kr",
    "pc-factory.co.kr",
    "www.comoasis.co.kr",
)


DEFAULT_RECIPE_PARTS: tuple["RecipePart", ...] = (  # noqa: UP037
    RecipePart("cpu:ULTRA 5 225", "CPU", ("https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=1203135", "https://www.yopc.co.kr/shop/item.php?it_id=5706172508"), ("225", "정품")),
    RecipePart("cpu:ULTRA 5 235", "CPU", ("https://comoasis.co.kr/shop/item.php?it_id=5706172509&service_id=pcdn", "https://www.pc4all.co.kr/shop/product_detail.html?pd_no=172430"), ("235", "정품")),
    RecipePart("cpu:ULTRA 7 265", "CPU", ("https://www.cynexpc.com/shop/product_detail.html?pd_no=181741",), ("265", "정품")),
    RecipePart("cpu:I5 14400", "CPU", ("https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=1102707", "https://usr.icoda.co.kr/item/view/1475392"), ("i5", "14400", "정품")),
    RecipePart("cpu:I5 14500", "CPU", ("https://comoasis.co.kr/shop/item.php?it_id=5698142153&service_id=pcdn", "https://www.pc4all.co.kr/shop/product_detail.html?pd_no=166256"), ("i5", "14500", "정품")),
    RecipePart("cpu:I7 14700", "CPU", (), ("i7", "14700", "정품"), sources_pending=True),
    RecipePart("ram:16", "메모리 16GB", ("https://www.kingdommol.com/shop/product_detail.html?pd_no=93410",), ("DDR5", "5600", "16GB")),
    RecipePart("ssd:500", "SSD 500GB", ("https://www.cynexpc.com/shop/product_detail.html?pd_no=192001", "https://oversystem.co.kr/stuff/stuff_view/0/11475"), ("T500", "500GB")),
    RecipePart("hdd:2", "HDD 2TB", ("https://www.duracom.co.kr/shop/product_detail.html?pd_no=118703", "https://www.picknmall.co.kr/shop/product_detail.html?pd_no=115643"), ("WD", "2TB")),
    RecipePart("mb:H610", "메인보드", ("https://www.pc4all.co.kr/shop/product_detail.html?pd_no=164248", "https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=1160176"), ("H610M",)),
    RecipePart("mb:H810", "메인보드", ("https://usr.icoda.co.kr/item/view/1771060", "https://www.suzakpc.co.kr/shop/item.php?it_id=5749621340"), ("H810M",)),
    RecipePart("mb:B760", "메인보드", ("https://www.duracom.co.kr/shop/product_detail.html?pd_no=110195", "https://www.comoasis.co.kr/shop/item.php?it_id=5676620235&service_id=pcdn"), ("B760M",)),
    RecipePart("mb:B860", "메인보드", ("https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=1228071",), ("B860M",)),
    RecipePart("psu:500W", "전원", ("https://oversystem.co.kr/stuff/stuff_view/0/10126",), ("500W",)),
    RecipePart("psu:400W", "전원", ("https://pc-factory.co.kr/shop/product_detail.html?pd_no=161087",), ("400", "TFX")),
    RecipePart("psu:450W-SFX", "전원", ("https://kingdommol.com/shop/product_detail.html?pd_no=29633", "https://www.cynexpc.com/shop/product_detail.html?pd_no=110163"), ("SFX", "450W")),
    RecipePart("case:slim-sfx", "케이스", ("https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=1345100", "https://usr.icoda.co.kr/item/view/1752128"), ("엘포즈", "슬림LP")),
    RecipePart("case:tower", "케이스", ("https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=1115493",), ("TA20",)),
    RecipePart("case:slim", "케이스", ("https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=1126931",), ("미니멀",)),
    RecipePart("kbm", "키보드·마우스", ("https://www.suzakpc.co.kr/shop/item.php?it_id=5676620315", "https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=1035497"), ("KM", "C1")),
    RecipePart("reader", "카드리더", ("https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=621907",), ("IN-UACARD",)),
    RecipePart("pc:ULTRA 5 225", "완제품 PC", (), ("AG251", "16GB", "512GB"), sources_pending=True),
    RecipePart("pc:ULTRA 5 235", "완제품 PC", (), ("AG351", "16GB", "512GB"), sources_pending=True),
    RecipePart("pc:ULTRA 7 265", "완제품 PC", (), ("AG651", "16GB", "512GB"), sources_pending=True),
    RecipePart("panel:24", "24포트 패치판넬(CAT.6A)", ("https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=1121630", "https://www.netzone.co.kr/shop/shopdetail.html?branduid=12239799"), ("IN-50U6A24P", "24포트"), "rack"),
    RecipePart("panel:24-rack", "24포트 패치판넬(CAT.6A, 랙형)", ("https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=1121630", "https://www.netzone.co.kr/shop/shopdetail.html?branduid=12239799"), ("IN-50U6A24P", "24포트"), "rack"),
    RecipePart("panel:24-enclosure", "24포트 패치판넬(CAT.6, 함체형)", ("https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=1174437", "https://www.devicemart.co.kr/goods/view?no=15542398"), ("SAFE-6patch-24P-BOX",), "enclosure"),
    RecipePart("rack:1200", "허브랙 22U", ("https://fibermart.co.kr/product/detail.html?product_no=11644",), ("LD-SGDR750", "H1200")),
    RecipePart("rack:1400", "서버랙 27U", ("https://fibermart.co.kr/product/detail.html?product_no=15587",), ("SAFE-1400S", "27U")),
    RecipePart("rack:1600", "통신랙 31U", ("https://fibermart.co.kr/product/detail.html?product_no=10362",), ("SGDR-750", "H1600")),
    RecipePart("rack:1800", "허브랙", ("https://fibermart.co.kr/product/detail.html?product_no=19212",), ("LD-H1800-NEW",)),
    RecipePart("rack:2000", "통신랙 40U", ("https://fibermart.co.kr/product/detail.html?product_no=10364",), ("SGDR-750", "H2000")),
)

DEFAULT_RECIPE_ADDERS: tuple["RecipeAdder", ...] = (  # noqa: UP037
    RecipeAdder("desktop:assembly", "조립비", 25000, "주최 측 데스크톱 판정의 조립비(https://savetax.now/520)", "https://savetax.now/520"),
    RecipeAdder("desktop:reader", "카드리더", 5600, "주최 측 데스크톱 판정의 카드리더 가격(https://savetax.now/405)"),
    RecipeAdder("panel:labor-terminated", "성단·설치 공임", 401649, "정보통신 표준품셈 4-3-3 패치판넬 24포트 성단·설치·결선 공임(부가세 포함), 주최 측 판정 https://savetax.now/222", "https://savetax.now/222"),
    RecipeAdder("panel:labor-mounted", "설치 공임", 79164, "정보통신 표준품셈 4-3-3 조립된 패치판넬 24포트 설치 공임(부가세 포함), 주최 측 판정 https://savetax.now/621", "https://savetax.now/621"),
    RecipeAdder("rack:labor", "랙 설치 공임", 167310, "정보통신 표준품셈 4-3-3 고정형 19인치 랙 설치 공임(부가세 포함), 주최 측 판정 https://savetax.now/176", "https://savetax.now/176"),
)

DEFAULT_DESKTOP_PC_REFS: tuple["DesktopPcRef", ...] = (  # noqa: UP037
    DesktopPcRef("ULTRA 5 225", "pc:ULTRA 5 225", "H810"),
    DesktopPcRef("ULTRA 5 235", "pc:ULTRA 5 235", "H810"),
    DesktopPcRef("ULTRA 7 265", "pc:ULTRA 7 265", "B860"),
)


def _check_recipe_parts(parts: object) -> tuple["RecipePart", ...]:  # noqa: UP037
    if not isinstance(parts, tuple):
        raise ConfigError("[SYS] invalid recipe_parts: expected a tuple")
    seen: set[str] = set()
    for part in parts:
        if not isinstance(part, RecipePart):
            raise ConfigError("[SYS] invalid recipe_parts: expected RecipePart")
        if not isinstance(part.key, str) or not part.key.strip() or part.key in seen:
            raise ConfigError("[SYS] invalid recipe_parts: duplicate or empty key")
        seen.add(part.key)
        sources = part.sources
        if isinstance(sources, str):
            raise ConfigError("[SYS] invalid recipe_parts: sources must be a tuple")
        if not isinstance(sources, tuple):
            raise ConfigError("[SYS] invalid recipe_parts: sources must be a tuple")
        pcode = part.pcode
        if sources:
            if any(not isinstance(url, str) or not url.strip() for url in sources):
                raise ConfigError("[SYS] invalid recipe_parts: sources must be non-empty")
            for url in sources:
                try:
                    _check_url(url, "recipe_parts: sources")
                except ConfigError:
                    raise ConfigError("[SYS] invalid recipe_parts: sources") from None
                host = (urlparse(url).hostname or "").lower().rstrip(".").removeprefix("www.")
                if host in _REJECTED_PART_HOSTS:
                    raise ConfigError("[SYS] invalid recipe_parts: danawa price-comparison page is not a seller page")
        elif part.sources_pending is True and not pcode:
            pass
        else:
            raise ConfigError("[SYS] invalid recipe_parts: sources must be non-empty")
        if not isinstance(part.name_tokens, tuple) or not part.name_tokens or any(not isinstance(token, str) or not token.strip() for token in part.name_tokens):
            raise ConfigError("[SYS] invalid recipe_parts: name_tokens must be non-empty")
        if part.mount_form is not None and part.mount_form not in ("rack", "enclosure"):
            raise ConfigError("[SYS] invalid recipe_parts: mount_form must be rack or enclosure")
    return parts


def _check_recipe_adders(adders: object) -> tuple["RecipeAdder", ...]:  # noqa: UP037
    if not isinstance(adders, tuple):
        raise ConfigError("[SYS] invalid recipe_adders: expected a tuple")
    seen: set[str] = set()
    for adder in adders:
        if not isinstance(adder, RecipeAdder):
            raise ConfigError("[SYS] invalid recipe_adders: expected RecipeAdder")
        if not isinstance(adder.key, str) or not adder.key.strip() or adder.key in seen:
            raise ConfigError("[SYS] invalid recipe_adders: duplicate or empty key")
        seen.add(adder.key)
        if isinstance(adder.amount_krw, bool) or not isinstance(adder.amount_krw, int) or adder.amount_krw <= 0:
            raise ConfigError("[SYS] invalid recipe_adders: amount_krw must be > 0")
        if adder.source_url is not None:
            if not isinstance(adder.source_url, str):
                raise ConfigError("[SYS] invalid recipe_adders: source_url")
            try:
                _check_url(adder.source_url, "recipe_adders: source_url")
            except ConfigError:
                raise ConfigError("[SYS] invalid recipe_adders: source_url") from None
    return adders


@dataclass(frozen=True)
class EvidenceConfig:
    """Thresholds, public hosts and output paths for evidence collection."""

    evidence_dir: Path = Path("scratch/evidence")
    listing_page_url_template: str = "https://savetax.now/listing/{listing_id}"
    danawa_search_url_template: str = "https://search.danawa.com/?query={query}"
    danawa_product_url_template: str = "https://prod.danawa.com/info/?pcode={pcode}"
    danawa_history_url_template: str = (
        "https://prod.danawa.com/info/ajax/getProductPriceList.ajax.php?productCode={pcode}"
    )
    public_user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
    public_request_interval_seconds: float = 1.0
    shops: tuple[ShopSpec, ...] = (
        ShopSpec("itncctv.kr", "cafe24"),
        ShopSpec("rftown.com", "cafe24"),
        ShopSpec("fibermart.co.kr", "cafe24"),
        ShopSpec("fiberland.co.kr", "cafe24"),
        ShopSpec("exatek.co.kr", "cafe24"),
        ShopSpec("btobcctv.com", "cafe24"),
        ShopSpec("tipsmall.co.kr", "cafe24"),
        ShopSpec("jiantatmall.co.kr", "cafe24"),
        ShopSpec("e-punglim.co.kr", "cafe24"),
        ShopSpec("boosungcamera.co.kr", "cafe24"),
        ShopSpec("hungrydica.kr", "cafe24"),
        ShopSpec("www.laondigital.co.kr", "cafe24"),
        ShopSpec("mediaviewsu.com", "cafe24"),
        ShopSpec("icepia.co.kr", "cafe24"),
        ShopSpec("www.cctv-mstore.com", "godomall"),
        ShopSpec("www.safecctv.net", "godomall"),
        ShopSpec("www.comeontop.co.kr", "godomall"),
        ShopSpec("www.bipum.net", "godomall"),
    )
    cross_markup_threshold: Decimal = Decimal("1.20")
    retail_markup_threshold: Decimal = Decimal("1.10")
    require_price_proof: bool = True
    min_item_expected_krw: int = 300_000
    max_benchmarks_per_item: int = 5
    region_all_token: str = "전지역"  # noqa: S105
    install_terms: tuple[str, ...] = ("현장설치도",)
    install_allowance_rate: Decimal = Decimal("0.15")
    min_offer_count: int = 2
    outlier_ratio: Decimal = Decimal("1.15")
    single_offer_stability_ratio: Decimal = Decimal("1.25")
    oem_gap_cap: Decimal = Decimal("3")
    single_piece_units: tuple[str, ...] = ("대", "개", "EA", "ea", "Ea")
    accessory_markers: tuple[str, ...] = (
        "액세서리",
        "소모품",
        "전용",
        "호환",
        "리모컨",
        "리모콘",
        "램프",
        "케이스",
        "브라켓",
        "브래킷",
        "거치대",
        "토너",
        "잉크",
        "필터",
        "렌탈",
        "중고",
        "리퍼",
        "파우치",
        "커버",
        "필름",
        "가방",
        "회전유리판",
        "부품",
        "충전기",
        "배터리",
        "재생",
        "해외구매",
    )
    pack_markers: tuple[str, ...] = (
        r"\(\s*\d+\s*[xX×]",  # noqa: RUF001
        r"[xX×]\s*\d+\s*\)",  # noqa: RUF001
        r"\d+\s*(개입|팩|박스)",
        r"세트\)",
        r"패키지",
    )
    brand_aliases: tuple[tuple[str, str], ...] = (
        ("엘지", "LG"),
        ("삼성", "SAMSUNG"),
        ("캐논", "CANON"),
        ("엡손", "EPSON"),
        ("브라더", "BROTHER"),
        ("소니", "SONY"),
        ("파나소닉", "PANASONIC"),
        ("니콘", "NIKON"),
        ("뷰소닉", "VIEWSONIC"),
        ("벤큐", "BENQ"),
        ("시스코", "CISCO"),
        ("넷기어", "NETGEAR"),
        ("씨게이트", "SEAGATE"),
        ("웨스턴디지털", "WESTERNDIGITAL"),
        ("인터엠", "인터M"),
        ("솔텍", "SOLTECH"),
        ("하니웰", "HONEYWELL"),
        ("아이디스", "IDIS"),
        ("에이텐", "ATEN"),
        ("쿠쿠", "CUCKOO"),
        ("델", "DELL"),
        ("레노버", "LENOVO"),
        ("옵토마", "OPTOMA"),
        ("샤프", "SHARP"),
        ("리코", "RICOH"),
        ("교세라", "KYOCERA"),
        ("신도", "SINDOH"),
        ("티피링크", "TPLINK"),
    )
    max_lines_per_draft: int = 500
    max_draft_bytes: int = 1_000_000
    unit_conflict_policy: UnitConflictPolicy = UnitConflictPolicy.ANNOTATE
    variant_policy: VariantPolicy = VariantPolicy.SERIES
    series_blocks: tuple[SeriesBlock, ...] = DEFAULT_SERIES_BLOCKS
    agency_markup_threshold: Decimal = Decimal("1.20")
    agency_window_days: int = 180
    agency_near_days: int = 3
    agency_price_floor_ratio: Decimal = Decimal("0.30")
    agency_mass_price_agencies: int = 5
    agency_min_supporters: int = 2
    agency_min_lines: int = 6
    agency_min_order_amount_krw: int = 30_000_000
    agency_island_tokens: tuple[str, ...] = ("제주", "서귀포", "울릉", "신안", "옹진", "완도", "진도")
    agency_excluded_categories: tuple[str, ...] = ("폐쇄형배전반", "분전반", "전동기제어반")
    agency_review_name_pattern: str = r"\d~\d"
    agency_request_scope: bool = True
    agency_request_listings_max: int = 200
    recipe_markup_threshold: Decimal = Decimal("1.20")
    recipe_band_upper: Decimal = Decimal("1.30")
    recipe_min_offers: int = 3
    recipe_listings_max: int = 12
    recipe_parts: tuple["RecipePart", ...] = DEFAULT_RECIPE_PARTS  # noqa: UP037
    recipe_adders: tuple["RecipeAdder", ...] = DEFAULT_RECIPE_ADDERS  # noqa: UP037
    evidence_hosts: tuple[str, ...] = DEFAULT_EVIDENCE_HOSTS
    desktop_category: str = "데스크톱컴퓨터"
    desktop_delivery_terms: tuple[str, ...] = ("납품장소도",)
    desktop_os_optional_values: tuple[str, ...] = ("옵션제공", "옵션 제공", "선택", "옵션품목(복구솔루션)")
    panel_category: str = "랙캐비닛용패널"
    panel_part_key: str = "panel:24"
    panel_floor_krw: int = 467_649
    panel_floor_unterminated_krw: int = 145_164
    rack_category: str = "네트워크시스템장비용랙"
    rack_classes: tuple["RackClass", ...] = DEFAULT_RACK_CLASSES  # noqa: UP037
    rack_width_mm: tuple[int, int] = (550, 650)
    rack_max_depth_mm: int = 750
    rack_exclusion_pattern: str = r"내진|서버|방음|항온|냉방|옥외|벽부|UPS|배터리|CCTV|방진|이동형"
    desktop_pc_refs: tuple["DesktopPcRef", ...] = DEFAULT_DESKTOP_PC_REFS  # noqa: UP037
    desktop_chipset_rank: tuple[str, ...] = ("H810", "B860")
    desktop_pc_max_power_w: int = 500

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "unit_conflict_policy", UnitConflictPolicy(self.unit_conflict_policy))
        except ValueError:
            raise ConfigError("[SYS] invalid unit_conflict_policy: must be annotate or reject") from None
        try:
            object.__setattr__(self, "variant_policy", VariantPolicy(self.variant_policy))
        except ValueError:
            raise ConfigError(
                "[SYS] invalid variant_policy: must be series, strict, allow_low_fact or off"
            ) from None
        object.__setattr__(self, "series_blocks", tuple(self.series_blocks))
        seen: dict[str, list[tuple[int, int]]] = {}
        for block in self.series_blocks:
            if not isinstance(block, SeriesBlock):
                raise ConfigError("[SYS] invalid series_blocks: expected SeriesBlock")
            ranges = seen.setdefault(block.contract_no, [])
            for first, last in ranges:
                if block.first_sequence <= last and first <= block.last_sequence:
                    raise ConfigError("[SYS] invalid series_blocks: overlapping blocks of one contract")
            ranges.append((block.first_sequence, block.last_sequence))
        for field_name, placeholder in _TEMPLATE_PLACEHOLDERS:
            value = getattr(self, field_name)
            _check_url(value, field_name)
            if not value.startswith("https://") or placeholder not in value:
                raise ConfigError(f"[SYS] invalid {field_name}: must be an https URL with {placeholder}")
        for threshold_name in ("cross_markup_threshold", "retail_markup_threshold"):
            threshold = getattr(self, threshold_name)
            if not threshold.is_finite() or threshold < 1:
                raise ConfigError(f"[SYS] invalid {threshold_name}: must be finite and >= 1")
        if not isinstance(self.require_price_proof, bool):
            raise ConfigError("[SYS] invalid require_price_proof: must be a bool")
        for rate_name in ("install_allowance_rate",):
            rate = getattr(self, rate_name)
            if not rate.is_finite() or rate < 0:
                raise ConfigError(f"[SYS] invalid {rate_name}: must be finite and >= 0")
        for ratio_name in ("outlier_ratio", "single_offer_stability_ratio", "oem_gap_cap"):
            ratio = getattr(self, ratio_name)
            if not ratio.is_finite() or ratio < 1:
                raise ConfigError(f"[SYS] invalid {ratio_name}: must be finite and >= 1")
        if self.min_item_expected_krw < 0:
            raise ConfigError("[SYS] invalid min_item_expected_krw: must be >= 0")
        if self.min_offer_count < 1:
            raise ConfigError("[SYS] invalid min_offer_count: must be >= 1")
        if self.max_benchmarks_per_item < 1:
            raise ConfigError("[SYS] invalid max_benchmarks_per_item: must be >= 1")
        if self.max_lines_per_draft < 1:
            raise ConfigError("[SYS] invalid max_lines_per_draft: must be >= 1")
        if self.max_draft_bytes < 1:
            raise ConfigError("[SYS] invalid max_draft_bytes: must be >= 1")
        if not math.isfinite(self.public_request_interval_seconds) or self.public_request_interval_seconds < 0:
            raise ConfigError("[SYS] invalid public_request_interval_seconds: must be finite and >= 0")
        if not self.agency_markup_threshold.is_finite() or self.agency_markup_threshold <= 1:
            raise ConfigError("[SYS] invalid agency_markup_threshold: must be finite and > 1")
        for name in ("agency_window_days", "agency_near_days", "agency_mass_price_agencies",
                     "agency_min_supporters", "agency_min_lines", "agency_min_order_amount_krw"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"[SYS] invalid {name}: expected integer")
        if self.agency_window_days < 1:
            raise ConfigError("[SYS] invalid agency_window_days: must be >= 1")
        if not 0 <= self.agency_near_days <= self.agency_window_days:
            raise ConfigError("[SYS] invalid agency_near_days: must be 0..agency_window_days")
        if (not self.agency_price_floor_ratio.is_finite() or not 0 < self.agency_price_floor_ratio < 1):
            raise ConfigError("[SYS] invalid agency_price_floor_ratio: must be finite and in (0, 1)")
        if self.agency_mass_price_agencies < 2:
            raise ConfigError("[SYS] invalid agency_mass_price_agencies: must be >= 2")
        if self.agency_min_supporters < 1:
            raise ConfigError("[SYS] invalid agency_min_supporters: must be >= 1")
        if self.agency_min_lines < 2:
            raise ConfigError("[SYS] invalid agency_min_lines: must be >= 2")
        if self.agency_min_order_amount_krw < 0:
            raise ConfigError("[SYS] invalid agency_min_order_amount_krw: must be >= 0")
        if not isinstance(self.agency_request_scope, bool):
            raise ConfigError("[SYS] invalid agency_request_scope: expected bool")
        if isinstance(self.agency_request_listings_max, bool) or not isinstance(
            self.agency_request_listings_max, int
        ):
            raise ConfigError("[SYS] invalid agency_request_listings_max: must be >= 1")
        if self.agency_request_listings_max < 1:
            raise ConfigError("[SYS] invalid agency_request_listings_max: must be >= 1")
        if not isinstance(self.recipe_markup_threshold, Decimal) or not self.recipe_markup_threshold.is_finite() or self.recipe_markup_threshold <= 1:
            raise ConfigError("[SYS] invalid recipe_markup_threshold: must be finite and > 1")
        if not isinstance(self.recipe_band_upper, Decimal) or not self.recipe_band_upper.is_finite() or self.recipe_band_upper <= self.recipe_markup_threshold:
            raise ConfigError("[SYS] invalid recipe_band_upper: must be finite and > recipe_markup_threshold")
        for name in ("recipe_min_offers", "recipe_listings_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"[SYS] invalid {name}: must be an integer >= 1")
        object.__setattr__(self, "recipe_parts", _check_recipe_parts(self.recipe_parts))
        object.__setattr__(self, "recipe_adders", _check_recipe_adders(self.recipe_adders))
        object.__setattr__(self, "evidence_hosts", tuple(self.evidence_hosts))
        if not self.evidence_hosts or any(not isinstance(h, str) or not h.strip() or "/" in h for h in self.evidence_hosts):
            raise ConfigError("[SYS] invalid evidence_hosts: must be non-empty bare hostnames")
        for host in self.evidence_hosts:
            _check_url("https://" + host, "evidence_hosts")
            if urlparse("https://" + host).hostname != host.lower():
                raise ConfigError("[SYS] invalid evidence_hosts: expected a bare hostname")
            if (host.lower() in _CREDENTIAL_HOSTS or host.lower().endswith("workers.dev")
                    or host.lower().rstrip(".").removeprefix("www.") in _REJECTED_PART_HOSTS):
                raise ConfigError(f"[SYS] invalid evidence_hosts: {host!r} is not a seller page host")
        for part in self.recipe_parts:
            if part.sources:
                for url in part.sources:
                    host = (urlparse(url).hostname or "").lower()
                    if host not in {h.lower() for h in self.evidence_hosts}:
                        raise ConfigError("[SYS] invalid recipe_parts: source host is not in evidence_hosts")
        object.__setattr__(self, "desktop_delivery_terms", tuple(self.desktop_delivery_terms))
        object.__setattr__(self, "desktop_os_optional_values", tuple(self.desktop_os_optional_values))
        object.__setattr__(self, "agency_island_tokens", tuple(self.agency_island_tokens))
        object.__setattr__(self, "agency_excluded_categories", tuple(self.agency_excluded_categories))
        part_keys = {part.key for part in self.recipe_parts}
        if not isinstance(self.panel_part_key, str) or self.panel_part_key not in part_keys:
            raise ConfigError("[SYS] invalid rack_classes: unknown recipe_parts key for panel")
        for floor_name in ("panel_floor_krw", "panel_floor_unterminated_krw"):
            floor_value = getattr(self, floor_name)
            if isinstance(floor_value, bool) or not isinstance(floor_value, int) or floor_value <= 0:
                raise ConfigError(f"[SYS] invalid {floor_name}: must be > 0")
        if (
            not isinstance(self.rack_width_mm, tuple)
            or len(self.rack_width_mm) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in self.rack_width_mm)
            or not 0 < self.rack_width_mm[0] < self.rack_width_mm[1]
        ):
            raise ConfigError("[SYS] invalid rack_width_mm: must be an increasing pair")
        if isinstance(self.rack_max_depth_mm, bool) or not isinstance(self.rack_max_depth_mm, int) or self.rack_max_depth_mm <= 0:
            raise ConfigError("[SYS] invalid rack_max_depth_mm: must be > 0")
        if not isinstance(self.rack_exclusion_pattern, str):
            raise ConfigError("[SYS] invalid rack_exclusion_pattern: expected a string")
        try:
            re.compile(self.rack_exclusion_pattern)
        except re.error:
            raise ConfigError("[SYS] invalid rack_exclusion_pattern: must compile") from None
        if not isinstance(self.rack_classes, tuple) or not self.rack_classes:
            raise ConfigError("[SYS] invalid rack_classes: expected a non-empty tuple")
        previous = 0
        for entry in self.rack_classes:
            if not isinstance(entry, RackClass):
                raise ConfigError("[SYS] invalid rack_classes: expected RackClass")
            if isinstance(entry.max_height_mm, bool) or not isinstance(entry.max_height_mm, int) or entry.max_height_mm <= 0:
                raise ConfigError("[SYS] invalid rack_classes: max_height_mm must be > 0")
            if entry.max_height_mm <= previous:
                raise ConfigError("[SYS] invalid rack_classes: max_height_mm must strictly increase")
            previous = entry.max_height_mm
            if not isinstance(entry.part_key, str) or entry.part_key not in part_keys:
                raise ConfigError("[SYS] invalid rack_classes: unknown recipe_parts key")
            if isinstance(entry.floor_krw, bool) or not isinstance(entry.floor_krw, int) or entry.floor_krw <= 0:
                raise ConfigError("[SYS] invalid rack_classes: floor must be > 0")
        if not isinstance(self.desktop_pc_refs, tuple) or not self.desktop_pc_refs:
            raise ConfigError("[SYS] invalid desktop_pc_refs: expected a non-empty tuple")
        if (
            not isinstance(self.desktop_chipset_rank, tuple)
            or not self.desktop_chipset_rank
            or any(not isinstance(c, str) or not c.strip() for c in self.desktop_chipset_rank)
            or len(set(self.desktop_chipset_rank)) != len(self.desktop_chipset_rank)
        ):
            raise ConfigError("[SYS] invalid desktop_pc_refs: bad chipset rank")
        if (
            isinstance(self.desktop_pc_max_power_w, bool)
            or not isinstance(self.desktop_pc_max_power_w, int)
            or self.desktop_pc_max_power_w < 1
        ):
            raise ConfigError("[SYS] invalid desktop_pc_refs: max power must be >= 1")
        seen_cpu: set[str] = set()
        for ref in self.desktop_pc_refs:
            if not isinstance(ref, DesktopPcRef):
                raise ConfigError("[SYS] invalid desktop_pc_refs: expected DesktopPcRef")
            if not isinstance(ref.cpu_key, str) or not ref.cpu_key.strip() or ref.cpu_key in seen_cpu:
                raise ConfigError("[SYS] invalid desktop_pc_refs: duplicate or empty cpu_key")
            seen_cpu.add(ref.cpu_key)
            if not isinstance(ref.part_key, str) or ref.part_key not in part_keys:
                raise ConfigError(f"[SYS] invalid desktop_pc_refs: unknown part_key {ref.part_key!r}")
            if ref.chipset not in self.desktop_chipset_rank:
                raise ConfigError(f"[SYS] invalid desktop_pc_refs: unknown chipset {ref.chipset!r}")
        try:
            re.compile(self.agency_review_name_pattern)
        except re.error:
            raise ConfigError("[SYS] invalid agency_review_name_pattern: must compile") from None
        for shop in self.shops:
            if shop.platform not in ("cafe24", "godomall"):
                raise ConfigError(f"[SYS] invalid shop platform: {shop.platform!r} must be cafe24 or godomall")
            if not shop.host or "/" in shop.host:
                raise ConfigError(f"[SYS] invalid shop host: {shop.host!r} must be a bare hostname")
            _check_url("https://" + shop.host, "shop host")
            if urlparse("https://" + shop.host).hostname != shop.host.lower():
                raise ConfigError("[SYS] invalid shop host: must be a bare hostname")
        for host in self.public_hosts():
            normalized = host.lower().rstrip(".")
            if normalized in _CREDENTIAL_HOSTS or normalized.endswith("workers.dev"):
                raise ConfigError(f"[SYS] invalid public host: {host!r} is a credential destination")

    def public_hosts(self) -> frozenset[str]:
        """Return every host a credential-less page fetch may target."""
        hosts = {urlparse(getattr(self, field_name)).hostname or "" for field_name, _ in _TEMPLATE_PLACEHOLDERS}
        hosts.update(shop.host.lower() for shop in self.shops)
        hosts.update(h.lower() for h in self.evidence_hosts)
        for adder in self.recipe_adders:
            if adder.source_url is not None:
                hosts.add(urlparse(adder.source_url).hostname or "")
        for part in self.recipe_parts:
            for url in part.sources:
                hosts.add(urlparse(url).hostname or "")
        hosts.discard("")
        return frozenset(hosts)


@dataclass(frozen=True)
class ApprovalConfig:
    """Approval gate and submission pacing."""

    ttl_hours: int = 6
    reserve_claims: int = 5
    lint_poll_seconds: float = 7.0
    lint_timeout_seconds: float = 600.0
    pause_between_submits_seconds: float = 1.0
    stop_after_consecutive_suspect: int = 2
    ledger_path: Path = field(default_factory=lambda: Path("scratch/evidence/ledger.jsonl"))

    def __post_init__(self) -> None:
        for name in ("ttl_hours", "reserve_claims", "stop_after_consecutive_suspect"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"[SYS] invalid {name}: expected integer")
        if self.ttl_hours < 1:
            raise ConfigError("[SYS] invalid ttl_hours: must be >= 1")
        if self.reserve_claims < 0:
            raise ConfigError("[SYS] invalid reserve_claims: must be >= 0")
        if not math.isfinite(self.lint_poll_seconds) or self.lint_poll_seconds <= 0:
            raise ConfigError("[SYS] invalid lint_poll_seconds: must be > 0")
        if not math.isfinite(self.lint_timeout_seconds) or self.lint_timeout_seconds < self.lint_poll_seconds:
            raise ConfigError("[SYS] invalid lint_timeout_seconds: must be >= lint_poll_seconds")
        if not math.isfinite(self.pause_between_submits_seconds) or self.pause_between_submits_seconds < 0:
            raise ConfigError("[SYS] invalid pause_between_submits_seconds: must be >= 0")
        if self.stop_after_consecutive_suspect < 1:
            raise ConfigError("[SYS] invalid stop_after_consecutive_suspect: must be >= 1")
