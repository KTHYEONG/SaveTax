"""Endpoints, thresholds, and credential loading."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import NoReturn, SupportsIndex
from urllib.parse import unquote, urlparse

from src.savetax.errors import ConfigError, CredentialError

logger = logging.getLogger(__name__)

_DATASET_DIR_PREFIX = "savetax-products-"
_TOKEN_PREFIX = "savetax_"  # noqa: S105
_FORBIDDEN_VALUE_CHARS = frozenset("\"'<>")
_ZIP_URL = "https://savetax-dataset.reform-dev-infra.workers.dev/latest/savetax-products.zip"

__all__ = ["AuditConfig", "SaveTaxConfig", "Secret", "load_token"]


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
