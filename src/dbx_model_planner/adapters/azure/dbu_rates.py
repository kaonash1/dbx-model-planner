"""Fetch DBU-per-hour rates per VM instance from the Azure Databricks pricing page.

The Azure Databricks pricing page at
https://azure.microsoft.com/en-us/pricing/details/databricks/
contains HTML tables listing the DBU consumption rate (DBU/hr) for every
VM instance type, broken down by workload type (All-Purpose, Jobs, etc.).

This module parses the first set of tables on the page (All-Purpose
Compute, Premium tier).  The DBU *count* per VM is identical across
workload types (All-Purpose, Jobs Compute, etc.); only the per-DBU
unit price differs.

Per-DBU unit prices are fetched from the Azure Retail Prices API
(``serviceName eq 'Azure Databricks'``) in USD for the configured
region.

The Databricks REST API and system tables do NOT expose DBU rates per
node type; the HTML page is the only publicly available source for
*counts*, while the Retail Prices API is used for per-DBU *prices*.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

AZURE_PRICING_URL = (
    "https://azure.microsoft.com/en-us/pricing/details/databricks/"
)

# Regex to match a pricing table row.
# Each <tr> has: <td>Instance</td> <td>vCPU</td> <td>RAM</td> <td>DBU Count</td>
#   <td>DBU Price</td> ...
# The DBU Count value may be on a separate line with leading whitespace.
# Group 5 (optional, not captured) is the DBU Price column which we ignore.
# Column 5 may contain "$-" or be absent for some instances.
_ROW_PATTERN = re.compile(
    r"<tr>\s*\n\s*<td>([^<]+)</td>\s*\n\s*<td>(\d+)</td>"
    r"\s*\n\s*<td>([^<]+)</td>\s*\n\s*<td>\s*\n\s*([\d.]+)"
    r"(?:\s*</td>\s*\n?\s*<td>[^0-9]*?[\d.]+)?",
)

# Default cache TTL: 24 hours (DBU rates change very rarely)
DEFAULT_DBU_CACHE_TTL = 24 * 60 * 60

# User-Agent for the HTTP request
_USER_AGENT = "dbx-model-planner/1.0"

# GPU model tokens that appear in Azure pricing page instance names
# but may be omitted in Databricks node_type_ids.
# Example: page has "ND96asr A100 v4" → Standard_ND96asr_A100_v4,
# but Databricks uses Standard_ND96asr_v4 (no "A100").
_GPU_MODEL_TOKENS = frozenset({
    "A100", "A10", "H100", "T4", "V100", "K80", "MI200", "MI300X", "L4", "L40S",
})


@dataclass(slots=True)
class DbuRateEntry:
    """DBU rate for a single VM instance type."""

    instance_name: str  # e.g. "NC24ads A100 v4"
    node_type_id: str  # e.g. "Standard_NC24ads_A100_v4"
    dbu_per_hour: float
    vcpu_count: int
    ram_gib: str


@dataclass(slots=True)
class DbuRateCache:
    """Cached DBU rates with TTL."""

    entries: dict[str, DbuRateEntry] = field(default_factory=dict)  # key=node_type_id
    fetched_at: float = 0.0
    ttl_seconds: float = DEFAULT_DBU_CACHE_TTL

    # Per-DBU unit prices by workload type key, fetched from
    # the Azure Retail Prices API in USD.
    # Example: {"all_purpose": 0.55, "jobs_compute": 0.30}
    dbu_unit_prices: dict[str, float] = field(default_factory=dict)
    unit_price_currency: str | None = None

    @property
    def is_expired(self) -> bool:
        if self.fetched_at == 0.0:
            return True
        return (time.time() - self.fetched_at) > self.ttl_seconds

    @property
    def is_populated(self) -> bool:
        return len(self.entries) > 0 and not self.is_expired

    def get_rate(self, node_type_id: str) -> float | None:
        """Get DBU/hr for a node type, or None if unknown.

        Tries an exact match first, then attempts to find a cache entry
        whose key differs only by a GPU model token (e.g. "A100").
        """
        entry = self.entries.get(node_type_id)
        if entry is not None and not self.is_expired:
            return entry.dbu_per_hour
        if self.is_expired:
            return None
        # Fuzzy: try adding GPU model tokens to find a match
        # e.g. Standard_ND96asr_v4 → Standard_ND96asr_A100_v4
        parts = node_type_id.split("_")
        for nid, e in self.entries.items():
            eparts = nid.split("_")
            diff = set(eparts) - set(parts)
            if len(diff) == 1 and diff.issubset(_GPU_MODEL_TOKENS):
                # Verify structural match: removing the token gives same id
                short = "_".join(p for p in eparts if p not in diff)
                if short == node_type_id:
                    return e.dbu_per_hour
        return None

    def as_dict(self) -> dict[str, float]:
        """Return {node_type_id: dbu_per_hour} for all cached entries.

        Includes alias entries for node_type_ids where the GPU model token
        (e.g. "A100") was stripped by Databricks.  For example, the pricing
        page lists "ND96asr A100 v4" → Standard_ND96asr_A100_v4, but
        Databricks may report Standard_ND96asr_v4.  Both keys are included.
        """
        if self.is_expired:
            return {}
        result = {nid: e.dbu_per_hour for nid, e in self.entries.items()}
        # Build short-form aliases by stripping GPU model tokens
        for nid, entry in self.entries.items():
            parts = nid.split("_")
            for token in _GPU_MODEL_TOKENS:
                if token in parts:
                    short = "_".join(p for p in parts if p != token)
                    if short not in result:
                        result[short] = entry.dbu_per_hour
        return result

    def get_unit_price(self, workload_type: str, currency_code: str) -> float | None:
        """Return the per-DBU unit price for a workload type and currency.

        Returns ``None`` if no API-derived price is available, or if the
        cached currency doesn't match the requested one.
        """
        if not self.dbu_unit_prices or self.is_expired:
            return None
        if self.unit_price_currency and self.unit_price_currency != currency_code:
            return None
        return self.dbu_unit_prices.get(workload_type)


def _instance_name_to_node_type_id(instance_name: str) -> str:
    """Convert Azure pricing page instance name to Databricks node_type_id.

    Example: "NC24ads A100 v4" -> "Standard_NC24ads_A100_v4"
    """
    return "Standard_" + re.sub(r"\s+", "_", instance_name.strip())


def _extract_section(html: str, section_heading: str) -> str:
    """Extract the content between two <h3> tags starting from the given heading.

    Returns the HTML between <h3>{section_heading}</h3> and the next <h3>.
    """
    pattern = f"<h3>{re.escape(section_heading)}</h3>"
    match = re.search(pattern, html)
    if not match:
        return ""
    start = match.start()
    next_h3 = html.find("<h3>", start + len(match.group()))
    if next_h3 < 0:
        return html[start:]
    return html[start:next_h3]


def parse_dbu_rates_from_html(html: str) -> list[DbuRateEntry]:
    """Parse DBU rates from the Azure Databricks pricing page HTML.

    Extracts from the first occurrence of each VM category section
    (General purpose, Memory optimized, Storage optimized, GPU,
    Confidential Compute), which corresponds to All-Purpose Compute.

    Returns a list of DbuRateEntry for every VM instance found.
    """
    categories = [
        "General purpose",
        "Memory optimized",
        "Storage optimized",
        "GPU",
        "Confidential Compute",
    ]

    entries: list[DbuRateEntry] = []
    seen_names: set[str] = set()

    for category in categories:
        section = _extract_section(html, category)
        if not section:
            logger.debug("Section '%s' not found in pricing page", category)
            continue

        for match in _ROW_PATTERN.finditer(section):
            instance_name = match.group(1).strip()
            vcpu = int(match.group(2))
            ram = match.group(3).strip()
            dbu = float(match.group(4))

            if instance_name in seen_names:
                continue
            seen_names.add(instance_name)

            node_type_id = _instance_name_to_node_type_id(instance_name)
            entries.append(DbuRateEntry(
                instance_name=instance_name,
                node_type_id=node_type_id,
                dbu_per_hour=dbu,
                vcpu_count=vcpu,
                ram_gib=ram,
            ))

    logger.info("Parsed %d VM instance DBU rates from pricing page", len(entries))
    return entries


def fetch_dbu_rates(timeout: float = 60.0) -> list[DbuRateEntry]:
    """Fetch and parse DBU rates from the Azure Databricks pricing page.

    Returns a list of DbuRateEntry or an empty list on failure.
    """
    headers = {"User-Agent": _USER_AGENT}
    req = urllib.request.Request(AZURE_PRICING_URL, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("Failed to fetch Azure pricing page: %s", exc)
        return []

    return parse_dbu_rates_from_html(html)


def fetch_dbu_unit_prices(
    region: str = "eastus",
    currency_code: str = "USD",
    timeout: float = 30.0,
) -> dict[str, float]:
    """Fetch per-DBU unit prices from the Azure Retail Prices API.

    Queries ``serviceName eq 'Azure Databricks'`` for the given region.
    Returns a dict mapping workload type keys to per-DBU unit prices
    in USD.

    Example return: ``{"all_purpose": 0.4774, "jobs_compute": 0.2604}``.
    Returns an empty dict on failure.
    """
    from .pricing import (
        AzureRetailPriceQuery,
        fetch_azure_retail_prices,
        normalize_azure_region,
    )

    normalized = normalize_azure_region(region)
    if normalized is None:
        logger.warning("Cannot normalize region '%s' for DBU price fetch", region)
        return {}

    query = AzureRetailPriceQuery(
        arm_region_name=normalized,
        currency_code=currency_code,
        service_name="Azure Databricks",
        service_family="Analytics",
        price_type="Consumption",
    )

    try:
        records = fetch_azure_retail_prices(query, timeout=timeout)
    except Exception as exc:
        logger.warning("Failed to fetch DBU unit prices: %s", exc)
        return {}

    if not records:
        return {}

    # Match records to workload types by meter name keywords.
    # We care about the Premium tier (the default for Azure Databricks).
    # Meter names look like:
    #   "Premium All-purpose Compute DBU"
    #   "Premium Jobs Compute DBU"
    #   "Premium Jobs Light Compute DBU"
    result: dict[str, float] = {}
    for record in records:
        meter = (record.meter_name or "").lower()
        if "dbu" not in meter:
            continue
        if "premium" not in meter:
            continue

        price = record.unit_price
        if price <= 0:
            continue

        if "all-purpose" in meter or "all purpose" in meter:
            result.setdefault("all_purpose", price)
        elif "jobs light" in meter or "jobs-light" in meter:
            result.setdefault("jobs_light", price)
        elif "jobs compute" in meter or "jobs-compute" in meter or "jobs" in meter:
            # Check "jobs light" first above to avoid matching "jobs" too broadly
            if "light" not in meter:
                result.setdefault("jobs_compute", price)

    logger.info(
        "Fetched DBU unit prices from Azure API (%s, %s): %s",
        normalized, currency_code, result,
    )
    return result


def build_dbu_rate_cache(
    entries: list[DbuRateEntry] | None = None,
    ttl_seconds: float = DEFAULT_DBU_CACHE_TTL,
) -> DbuRateCache:
    """Build a DbuRateCache from parsed entries, or fetch fresh if None."""
    if entries is None:
        entries = fetch_dbu_rates()

    cache = DbuRateCache(ttl_seconds=ttl_seconds)
    for entry in entries:
        cache.entries[entry.node_type_id] = entry
    cache.fetched_at = time.time()
    return cache


# -- File-based persistence ------------------------------------------------


def _default_cache_path() -> Path:
    """Return the default file path for the DBU rate cache."""
    import os

    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "dbx-model-planner" / "dbu_rate_cache.json"


def save_dbu_cache(cache: DbuRateCache, path: Path | None = None) -> None:
    """Persist the DBU rate cache to a JSON file."""
    path = path or _default_cache_path()
    data = {
        "fetched_at": cache.fetched_at,
        "ttl_seconds": cache.ttl_seconds,
        "dbu_unit_prices": cache.dbu_unit_prices,
        "unit_price_currency": cache.unit_price_currency,
        "entries": {
            nid: {
                "instance_name": e.instance_name,
                "node_type_id": e.node_type_id,
                "dbu_per_hour": e.dbu_per_hour,
                "vcpu_count": e.vcpu_count,
                "ram_gib": e.ram_gib,
            }
            for nid, e in cache.entries.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.debug("Saved DBU rate cache (%d entries) to %s", len(cache.entries), path)


def load_dbu_cache(path: Path | None = None) -> DbuRateCache | None:
    """Load a DBU rate cache from a JSON file, or None if missing/corrupt."""
    path = path or _default_cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cache = DbuRateCache(
            fetched_at=data.get("fetched_at", 0.0),
            ttl_seconds=data.get("ttl_seconds", DEFAULT_DBU_CACHE_TTL),
            dbu_unit_prices=data.get("dbu_unit_prices") or {},
            unit_price_currency=data.get("unit_price_currency"),
        )
        for nid, entry_data in data.get("entries", {}).items():
            cache.entries[nid] = DbuRateEntry(
                instance_name=entry_data["instance_name"],
                node_type_id=entry_data["node_type_id"],
                dbu_per_hour=entry_data["dbu_per_hour"],
                vcpu_count=entry_data["vcpu_count"],
                ram_gib=entry_data["ram_gib"],
            )
        return cache
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to load DBU rate cache from %s: %s", path, exc)
        return None
