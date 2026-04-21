from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping


class WorkloadType(StrEnum):
    """Databricks compute workload type.

    Determines the per-DBU price.  The DBU *count* per VM instance is
    the same across workload types; only the unit price differs.
    """

    ALL_PURPOSE = "all_purpose"
    JOBS_COMPUTE = "jobs_compute"


# Default per-DBU rates (USD list price, Premium tier).
# These serve as **fallback** presets when the Azure Retail Prices API
# has not yet been queried.
# Correct rates are fetched at runtime via
# ``dbu_rates.fetch_dbu_unit_prices()`` and stored in the cache.
WORKLOAD_DBU_PRESETS: dict[WorkloadType, float] = {
    WorkloadType.ALL_PURPOSE: 0.55,
    WorkloadType.JOBS_COMPUTE: 0.30,
}

WORKLOAD_LABELS: dict[WorkloadType, str] = {
    WorkloadType.ALL_PURPOSE: "All-Purpose Compute",
    WorkloadType.JOBS_COMPUTE: "Jobs Compute",
}

_WORKLOAD_CYCLE = [WorkloadType.ALL_PURPOSE, WorkloadType.JOBS_COMPUTE]


@dataclass(slots=True)
class PricingAdjustments:
    """Organization-specific pricing adjustments applied on top of list prices."""

    discount_rate: float = 0.0
    vat_rate: float = 0.0
    currency_code: str = "USD"
    azure_region: str = ""
    price_cache_ttl_seconds: int = 14400  # 4 hours
    auto_fetch_pricing: bool = True


@dataclass(slots=True)
class DatabricksPricing:
    """Configurable Databricks DBU inputs used by the cost engine."""

    dbu_rate_per_unit: float = 0.55  # USD; auto-updated from Azure API
    workload_type: str = "all_purpose"  # "all_purpose" or "jobs_compute"


@dataclass(slots=True)
class WorkspacePreferences:
    """Operator preferences and restrictions for recommendations."""

    preferred_regions: list[str] = field(default_factory=list)
    approved_runtimes: list[str] = field(default_factory=list)
    blocked_node_types: list[str] = field(default_factory=list)
    blocked_gpu_families: list[str] = field(default_factory=list)
    blocked_skus: list[str] = field(default_factory=list)
    prefer_serverless_serving: bool = False


@dataclass(slots=True)
class CatalogDefaults:
    """Default Unity Catalog targets used for deployment plans."""

    catalog: str | None = None
    schema: str | None = None
    volume: str | None = None


@dataclass(slots=True)
class ProfileNames:
    """Named config profiles used by runtime and storage helpers."""

    config: str = "default"
    inventory: str = "default"
    model: str = "default"
    runtime: str = "default"


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""

    pricing: PricingAdjustments = field(default_factory=PricingAdjustments)
    databricks: DatabricksPricing = field(default_factory=DatabricksPricing)
    workspace: WorkspacePreferences = field(default_factory=WorkspacePreferences)
    catalog: CatalogDefaults = field(default_factory=CatalogDefaults)
    profiles: ProfileNames = field(default_factory=ProfileNames)


DEFAULT_CONFIG_TEMPLATE = dedent(
    """
    # dbx-model-planner.toml
    # Keep secrets out of this file. Put credentials in environment variables
    # such as DATABRICKS_HOST and DATABRICKS_TOKEN instead of embedding them here.

    [pricing]
    discount_rate = 0.0
    vat_rate = 0.0
    currency_code = "USD"
    azure_region = ""
    price_cache_ttl_seconds = 14400
    auto_fetch_pricing = true

    [databricks]
    dbu_rate_per_unit = 0.55
    workload_type = "all_purpose"

    [workspace]
    preferred_regions = []
    approved_runtimes = []
    blocked_node_types = []
    blocked_gpu_families = []
    blocked_skus = []
    prefer_serverless_serving = false

    [catalog]
    # catalog = "main"
    # schema = "models"
    # volume = "artifacts"

    [profiles]
    config = "default"
    inventory = "default"
    model = "default"
    runtime = "default"
    """
).strip()


def render_default_config_template() -> str:
    """Return the local TOML config template."""

    return f"{DEFAULT_CONFIG_TEMPLATE}\n"


def write_default_config_template(path: Path | str) -> Path:
    """Write the local TOML config template to disk."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_default_config_template(), encoding="utf-8")
    return destination


def save_pricing_config(
    *,
    azure_region: str = "",
    discount_rate: float = 0.0,
    vat_rate: float = 0.0,
    config_path: Path | str | None = None,
) -> Path:
    """Persist pricing settings (region, discount, VAT) to config.toml.

    If the file exists, individual values are updated in-place.
    Otherwise a fresh config is created from the default template.
    """
    path = _resolve_config_path(config_path, dict(os.environ))

    _REPLACEMENTS: list[tuple[str, str]] = [
        (r'azure_region\s*=\s*"[^"]*"', f'azure_region = "{azure_region}"'),
        (r"discount_rate\s*=\s*[\d.]+", f"discount_rate = {discount_rate}"),
        (r"vat_rate\s*=\s*[\d.]+", f"vat_rate = {vat_rate}"),
    ]

    if path.exists():
        content = path.read_text(encoding="utf-8")
        for pattern, replacement in _REPLACEMENTS:
            content, n = re.subn(pattern, replacement, content, count=1)
            if n == 0:
                # Key not found — append to end of [pricing] section
                content = content.rstrip() + f"\n{replacement}\n"
        path.write_text(content, encoding="utf-8")
    else:
        # Generate from template with substituted values
        content = DEFAULT_CONFIG_TEMPLATE
        content = content.replace('azure_region = ""', f'azure_region = "{azure_region}"')
        content = content.replace("discount_rate = 0.0", f"discount_rate = {discount_rate}")
        content = content.replace("vat_rate = 0.0", f"vat_rate = {vat_rate}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n", encoding="utf-8")

    return path


def _default_config_path() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser()
    return config_home / "dbx-model-planner" / "config.toml"


def _resolve_config_path(config_path: Path | str | None, env: Mapping[str, str]) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser()

    for env_name in ("DBX_MODEL_PLANNER_CONFIG", "DBX_MODEL_PLANNER_CONFIG_PATH"):
        value = env.get(env_name)
        if value:
            return Path(value).expanduser()

    return _default_config_path()


def load_app_config(
    config_path: Path | str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load config from TOML plus environment overrides."""

    env_map = dict(os.environ if env is None else env)
    resolved_path = _resolve_config_path(config_path, env_map)
    config = AppConfig()

    if resolved_path.exists():
        with resolved_path.open("rb") as handle:
            loaded = tomllib.load(handle)
        _apply_mapping(config, loaded)

    _apply_env_overrides(config, env_map)
    return config


def _apply_mapping(config: AppConfig, loaded: Mapping[str, Any]) -> None:
    pricing = _mapping_section(loaded, "pricing")
    if pricing:
        config.pricing.discount_rate = float(pricing.get("discount_rate", config.pricing.discount_rate))
        config.pricing.vat_rate = float(pricing.get("vat_rate", config.pricing.vat_rate))
        config.pricing.currency_code = str(pricing.get("currency_code", config.pricing.currency_code))
        config.pricing.azure_region = str(pricing.get("azure_region", config.pricing.azure_region))
        config.pricing.price_cache_ttl_seconds = int(
            pricing.get("price_cache_ttl_seconds", config.pricing.price_cache_ttl_seconds)
        )
        config.pricing.auto_fetch_pricing = bool(
            pricing.get("auto_fetch_pricing", config.pricing.auto_fetch_pricing)
        )

    databricks = _mapping_section(loaded, "databricks")
    if databricks:
        config.databricks.dbu_rate_per_unit = float(
            databricks.get("dbu_rate_per_unit", config.databricks.dbu_rate_per_unit)
        )
        config.databricks.workload_type = str(
            databricks.get("workload_type", config.databricks.workload_type)
        )

    workspace = _mapping_section(loaded, "workspace")
    if workspace:
        config.workspace.preferred_regions = _as_str_list(
            workspace.get("preferred_regions", config.workspace.preferred_regions)
        )
        config.workspace.approved_runtimes = _as_str_list(
            workspace.get("approved_runtimes", config.workspace.approved_runtimes)
        )
        config.workspace.blocked_node_types = _as_str_list(
            workspace.get("blocked_node_types", config.workspace.blocked_node_types)
        )
        config.workspace.blocked_gpu_families = _as_str_list(
            workspace.get("blocked_gpu_families", config.workspace.blocked_gpu_families)
        )
        config.workspace.blocked_skus = _as_str_list(workspace.get("blocked_skus", config.workspace.blocked_skus))
        config.workspace.prefer_serverless_serving = bool(
            workspace.get("prefer_serverless_serving", config.workspace.prefer_serverless_serving)
        )

    catalog = _mapping_section(loaded, "catalog")
    if catalog:
        config.catalog.catalog = _as_optional_str(catalog.get("catalog", config.catalog.catalog))
        config.catalog.schema = _as_optional_str(catalog.get("schema", config.catalog.schema))
        config.catalog.volume = _as_optional_str(catalog.get("volume", config.catalog.volume))

    profiles = _mapping_section(loaded, "profiles")
    if profiles:
        config.profiles.config = str(profiles.get("config", config.profiles.config))
        config.profiles.inventory = str(profiles.get("inventory", config.profiles.inventory))
        config.profiles.model = str(profiles.get("model", config.profiles.model))
        config.profiles.runtime = str(profiles.get("runtime", config.profiles.runtime))


def _apply_env_overrides(config: AppConfig, env: Mapping[str, str]) -> None:
    if value := _first_env(env, "DBX_MODEL_PLANNER_PRICING_DISCOUNT_RATE"):
        config.pricing.discount_rate = float(value)
    if value := _first_env(env, "DBX_MODEL_PLANNER_PRICING_VAT_RATE"):
        config.pricing.vat_rate = float(value)
    if value := _first_env(env, "DBX_MODEL_PLANNER_PRICING_CURRENCY_CODE"):
        config.pricing.currency_code = value
    if value := _first_env(env, "DBX_MODEL_PLANNER_PRICING_AZURE_REGION"):
        config.pricing.azure_region = value
    if value := _first_env(env, "DBX_MODEL_PLANNER_PRICING_CACHE_TTL"):
        config.pricing.price_cache_ttl_seconds = int(value)
    if value := _first_env(env, "DBX_MODEL_PLANNER_PRICING_AUTO_FETCH"):
        config.pricing.auto_fetch_pricing = _parse_bool(value)

    if value := _first_env(env, "DBX_MODEL_PLANNER_DATABRICKS_DBU_RATE_PER_UNIT"):
        config.databricks.dbu_rate_per_unit = float(value)
    if value := _first_env(env, "DBX_MODEL_PLANNER_DATABRICKS_WORKLOAD_TYPE"):
        config.databricks.workload_type = value

    if value := _first_env(env, "DBX_MODEL_PLANNER_WORKSPACE_PREFERRED_REGIONS"):
        config.workspace.preferred_regions = _parse_str_list(value)
    if value := _first_env(env, "DBX_MODEL_PLANNER_WORKSPACE_APPROVED_RUNTIMES"):
        config.workspace.approved_runtimes = _parse_str_list(value)
    if value := _first_env(env, "DBX_MODEL_PLANNER_WORKSPACE_BLOCKED_NODE_TYPES"):
        config.workspace.blocked_node_types = _parse_str_list(value)
    if value := _first_env(env, "DBX_MODEL_PLANNER_WORKSPACE_BLOCKED_GPU_FAMILIES"):
        config.workspace.blocked_gpu_families = _parse_str_list(value)
    if value := _first_env(env, "DBX_MODEL_PLANNER_WORKSPACE_BLOCKED_SKUS"):
        config.workspace.blocked_skus = _parse_str_list(value)
    if value := _first_env(env, "DBX_MODEL_PLANNER_WORKSPACE_PREFER_SERVERLESS_SERVING"):
        config.workspace.prefer_serverless_serving = _parse_bool(value)

    if value := _first_env(env, "DBX_MODEL_PLANNER_CATALOG_NAME"):
        config.catalog.catalog = value
    if value := _first_env(env, "DBX_MODEL_PLANNER_CATALOG_SCHEMA"):
        config.catalog.schema = value
    if value := _first_env(env, "DBX_MODEL_PLANNER_CATALOG_VOLUME"):
        config.catalog.volume = value

    if value := _first_env(env, "DBX_MODEL_PLANNER_PROFILE_CONFIG"):
        config.profiles.config = value
    if value := _first_env(env, "DBX_MODEL_PLANNER_PROFILE_INVENTORY"):
        config.profiles.inventory = value
    if value := _first_env(env, "DBX_MODEL_PLANNER_PROFILE_MODEL"):
        config.profiles.model = value
    if value := _first_env(env, "DBX_MODEL_PLANNER_PROFILE_RUNTIME"):
        config.profiles.runtime = value


def _mapping_section(loaded: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = loaded.get(name, {})
    return section if isinstance(section, Mapping) else {}


def _first_env(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value is not None and value != "":
            return value
    return None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    if isinstance(value, str):
        return _parse_str_list(value)
    return [str(item) for item in list(value)]


def _parse_str_list(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        else:
            if isinstance(parsed, list):
                return [str(item) for item in parsed]

    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
