"""TTL-based price cache for Azure VM pricing.

Fetches retail prices for all workspace node types in a single bulk query,
caches results in memory with a configurable TTL, and optionally persists
to a JSON file for cross-session reuse.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pricing import (
    AzureRetailPriceQuery,
    AzureRetailPriceRecord,
    fetch_azure_retail_prices,
    normalize_azure_region,
    select_azure_retail_price,
)
from .sku import arm_sku_candidates_from_node_type

logger = logging.getLogger(__name__)

# Default TTL: 4 hours
DEFAULT_CACHE_TTL_SECONDS = 4 * 60 * 60

# Max SKU names per Azure Retail Prices API query.
# The Azure Retail Prices API returns HTTP 400 when the OData filter
# string gets too long (~19+ armSkuName OR clauses).  10 is safe.
_MAX_SKUS_PER_QUERY = 10


@dataclass(slots=True)
class PriceCacheEntry:
    """A cached VM price for a single node type."""

    node_type_id: str
    arm_sku_name: str | None
    hourly_rate: float
    currency_code: str
    region: str
    fetched_at: float


@dataclass(slots=True)
class PriceCache:
    """In-memory price cache with TTL expiry."""

    entries: dict[str, PriceCacheEntry] = field(default_factory=dict)
    region: str | None = None
    fetched_at: float = 0.0
    ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS

    @property
    def is_expired(self) -> bool:
        if self.fetched_at == 0.0:
            return True
        return (time.time() - self.fetched_at) > self.ttl_seconds

    @property
    def is_populated(self) -> bool:
        return len(self.entries) > 0 and not self.is_expired

    def get_rate(self, node_type_id: str) -> float | None:
        """Get the cached hourly rate for a node type, or None if not cached/expired."""
        entry = self.entries.get(node_type_id)
        if entry is None:
            return None
        if self.is_expired:
            return None
        return entry.hourly_rate

    def as_vm_pricing_dict(self) -> dict[str, float]:
        """Return a {node_type_id: hourly_rate} dict for the recommender."""
        if self.is_expired:
            return {}
        return {
            entry.node_type_id: entry.hourly_rate
            for entry in self.entries.values()
        }

    def age_minutes(self) -> float:
        """Return how many minutes ago the cache was populated."""
        if self.fetched_at == 0.0:
            return 0.0
        return (time.time() - self.fetched_at) / 60.0


def _cache_file_path() -> Path:
    """Default file cache path."""
    import os
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "dbx-model-planner" / "price_cache.json"


def save_price_cache(cache: PriceCache, path: Path | None = None) -> None:
    """Persist cache to a JSON file."""
    target = path or _cache_file_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "region": cache.region,
            "fetched_at": cache.fetched_at,
            "ttl_seconds": cache.ttl_seconds,
            "entries": {
                node_id: {
                    "node_type_id": e.node_type_id,
                    "arm_sku_name": e.arm_sku_name,
                    "hourly_rate": e.hourly_rate,
                    "currency_code": e.currency_code,
                    "region": e.region,
                    "fetched_at": e.fetched_at,
                }
                for node_id, e in cache.entries.items()
            },
        }
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Saved price cache to %s (%d entries)", target, len(cache.entries))
    except Exception:
        logger.warning("Failed to save price cache to %s", target, exc_info=True)


def load_price_cache(
    path: Path | None = None,
    ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
) -> PriceCache:
    """Load cache from a JSON file. Returns empty cache if file missing/corrupt."""
    target = path or _cache_file_path()
    cache = PriceCache(ttl_seconds=ttl_seconds)
    try:
        if not target.exists():
            return cache
        data = json.loads(target.read_text(encoding="utf-8"))
        cache.region = data.get("region")
        cache.fetched_at = float(data.get("fetched_at", 0.0))
        for node_id, entry_data in data.get("entries", {}).items():
            cache.entries[node_id] = PriceCacheEntry(
                node_type_id=entry_data["node_type_id"],
                arm_sku_name=entry_data.get("arm_sku_name"),
                hourly_rate=float(entry_data["hourly_rate"]),
                currency_code=entry_data.get("currency_code", "USD"),
                region=entry_data.get("region", ""),
                fetched_at=float(entry_data.get("fetched_at", cache.fetched_at)),
            )
        logger.debug(
            "Loaded price cache from %s (%d entries, age %.0f min)",
            target, len(cache.entries), cache.age_minutes(),
        )
    except Exception:
        logger.warning("Failed to load price cache from %s", target, exc_info=True)
    return cache


def fetch_bulk_vm_prices(
    node_type_ids: list[str],
    region: str,
    *,
    currency_code: str = "USD",
    timeout: float = 30.0,
) -> dict[str, tuple[float, str | None, AzureRetailPriceRecord | None]]:
    """Fetch Azure retail prices for a list of node types in a region.

    Returns a dict of {node_type_id: (hourly_rate, arm_sku_name, record)}.
    Node types with no matching price are omitted.

    Batches requests to stay within OData filter limits.
    """
    normalized_region = normalize_azure_region(region)
    if normalized_region is None:
        logger.warning("Could not normalize region '%s'", region)
        return {}

    # Build deduplicated list of ARM SKU candidates across all node types
    node_to_skus: dict[str, list[str]] = {}
    all_skus: list[str] = []
    seen_skus: set[str] = set()

    for node_id in node_type_ids:
        candidates = arm_sku_candidates_from_node_type(node_id)
        node_to_skus[node_id] = candidates
        for sku in candidates:
            if sku not in seen_skus:
                seen_skus.add(sku)
                all_skus.append(sku)

    if not all_skus:
        return {}

    # Fetch in batches
    all_records: list[AzureRetailPriceRecord] = []
    for batch_start in range(0, len(all_skus), _MAX_SKUS_PER_QUERY):
        batch = all_skus[batch_start:batch_start + _MAX_SKUS_PER_QUERY]
        query = AzureRetailPriceQuery(
            arm_region_name=normalized_region,
            arm_sku_names=batch,
            currency_code=currency_code,
        )
        try:
            records = fetch_azure_retail_prices(query, timeout=timeout)
            all_records.extend(records)
        except Exception:
            logger.warning(
                "Price fetch failed for batch %d-%d",
                batch_start, batch_start + len(batch),
                exc_info=True,
            )

    # Match each node type to the best price record
    result: dict[str, tuple[float, str | None, AzureRetailPriceRecord | None]] = {}
    for node_id, sku_candidates in node_to_skus.items():
        selected = select_azure_retail_price(
            all_records,
            arm_region_name=normalized_region,
            arm_sku_names=sku_candidates,
        )
        if selected is not None and selected.unit_price > 0:
            result[node_id] = (selected.unit_price, selected.arm_sku_name, selected)

    return result


def refresh_price_cache(
    node_type_ids: list[str],
    region: str,
    *,
    currency_code: str = "USD",
    ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    timeout: float = 30.0,
    persist: bool = True,
) -> PriceCache:
    """Fetch prices for all node types and build a fresh cache.

    This is the main entry point for the TUI to call from a background thread.
    """
    normalized_region = normalize_azure_region(region) or region
    now = time.time()

    cache = PriceCache(
        region=normalized_region,
        fetched_at=now,
        ttl_seconds=ttl_seconds,
    )

    prices = fetch_bulk_vm_prices(
        node_type_ids,
        region,
        currency_code=currency_code,
        timeout=timeout,
    )

    for node_id, (rate, arm_sku, _record) in prices.items():
        cache.entries[node_id] = PriceCacheEntry(
            node_type_id=node_id,
            arm_sku_name=arm_sku,
            hourly_rate=rate,
            currency_code=currency_code,
            region=normalized_region,
            fetched_at=now,
        )

    logger.info(
        "Refreshed price cache: %d/%d nodes priced in %s",
        len(cache.entries), len(node_type_ids), normalized_region,
    )

    if persist:
        save_price_cache(cache)

    return cache
