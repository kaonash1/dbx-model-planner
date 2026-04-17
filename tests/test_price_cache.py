"""Tests for the Azure price cache module."""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from dbx_model_planner.adapters.azure.price_cache import (
    DEFAULT_CACHE_TTL_SECONDS,
    PriceCache,
    PriceCacheEntry,
    fetch_bulk_vm_prices,
    load_price_cache,
    refresh_price_cache,
    save_price_cache,
)


class PriceCacheTests(unittest.TestCase):
    """Test the PriceCache dataclass behaviour."""

    def test_empty_cache_is_expired(self) -> None:
        cache = PriceCache()
        self.assertTrue(cache.is_expired)
        self.assertFalse(cache.is_populated)

    def test_populated_cache_is_not_expired(self) -> None:
        cache = PriceCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["node1"] = PriceCacheEntry(
            node_type_id="node1",
            arm_sku_name="Standard_NC24ads_A100_v4",
            hourly_rate=3.67,
            currency_code="USD",
            region="westeurope",
            fetched_at=time.time(),
        )
        self.assertFalse(cache.is_expired)
        self.assertTrue(cache.is_populated)

    def test_expired_cache(self) -> None:
        cache = PriceCache(
            fetched_at=time.time() - 10000,
            ttl_seconds=100,
        )
        cache.entries["node1"] = PriceCacheEntry(
            node_type_id="node1",
            arm_sku_name=None,
            hourly_rate=1.0,
            currency_code="USD",
            region="westeurope",
            fetched_at=time.time() - 10000,
        )
        self.assertTrue(cache.is_expired)
        self.assertFalse(cache.is_populated)

    def test_get_rate_returns_value_when_populated(self) -> None:
        cache = PriceCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["node1"] = PriceCacheEntry(
            node_type_id="node1",
            arm_sku_name=None,
            hourly_rate=5.50,
            currency_code="USD",
            region="westeurope",
            fetched_at=time.time(),
        )
        self.assertEqual(cache.get_rate("node1"), 5.50)

    def test_get_rate_returns_none_for_missing_node(self) -> None:
        cache = PriceCache(fetched_at=time.time(), ttl_seconds=3600)
        self.assertIsNone(cache.get_rate("nonexistent"))

    def test_get_rate_returns_none_when_expired(self) -> None:
        cache = PriceCache(fetched_at=time.time() - 10000, ttl_seconds=100)
        cache.entries["node1"] = PriceCacheEntry(
            node_type_id="node1",
            arm_sku_name=None,
            hourly_rate=5.50,
            currency_code="USD",
            region="westeurope",
            fetched_at=time.time() - 10000,
        )
        self.assertIsNone(cache.get_rate("node1"))

    def test_as_vm_pricing_dict(self) -> None:
        cache = PriceCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["node1"] = PriceCacheEntry(
            node_type_id="node1",
            arm_sku_name=None,
            hourly_rate=3.67,
            currency_code="USD",
            region="westeurope",
            fetched_at=time.time(),
        )
        cache.entries["node2"] = PriceCacheEntry(
            node_type_id="node2",
            arm_sku_name=None,
            hourly_rate=7.34,
            currency_code="USD",
            region="westeurope",
            fetched_at=time.time(),
        )

        pricing = cache.as_vm_pricing_dict()
        self.assertEqual(pricing, {"node1": 3.67, "node2": 7.34})

    def test_as_vm_pricing_dict_empty_when_expired(self) -> None:
        cache = PriceCache(fetched_at=time.time() - 10000, ttl_seconds=100)
        cache.entries["node1"] = PriceCacheEntry(
            node_type_id="node1",
            arm_sku_name=None,
            hourly_rate=3.67,
            currency_code="USD",
            region="westeurope",
            fetched_at=time.time() - 10000,
        )
        self.assertEqual(cache.as_vm_pricing_dict(), {})

    def test_age_minutes(self) -> None:
        cache = PriceCache(fetched_at=time.time() - 120, ttl_seconds=3600)
        self.assertAlmostEqual(cache.age_minutes(), 2.0, delta=0.1)

    def test_age_minutes_zero_when_unfetched(self) -> None:
        cache = PriceCache()
        self.assertEqual(cache.age_minutes(), 0.0)


class PriceCacheSerializationTests(unittest.TestCase):
    """Test save/load of price cache to/from JSON file."""

    def test_save_and_load_roundtrip(self, tmp_path: Path | None = None) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"

            cache = PriceCache(
                region="westeurope",
                fetched_at=time.time(),
                ttl_seconds=7200,
            )
            cache.entries["Standard_NC24ads_A100_v4"] = PriceCacheEntry(
                node_type_id="Standard_NC24ads_A100_v4",
                arm_sku_name="Standard_NC24ads_A100_v4",
                hourly_rate=3.67,
                currency_code="USD",
                region="westeurope",
                fetched_at=cache.fetched_at,
            )

            save_price_cache(cache, path=path)
            self.assertTrue(path.exists())

            loaded = load_price_cache(path=path, ttl_seconds=7200)
            self.assertEqual(loaded.region, "westeurope")
            self.assertEqual(len(loaded.entries), 1)
            entry = loaded.entries["Standard_NC24ads_A100_v4"]
            self.assertEqual(entry.hourly_rate, 3.67)
            self.assertEqual(entry.currency_code, "USD")

    def test_load_missing_file_returns_empty_cache(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            cache = load_price_cache(path=path)
            self.assertEqual(len(cache.entries), 0)
            self.assertTrue(cache.is_expired)

    def test_load_corrupt_file_returns_empty_cache(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("not valid json{{{", encoding="utf-8")
            cache = load_price_cache(path=path)
            self.assertEqual(len(cache.entries), 0)


class FetchBulkVmPricesTests(unittest.TestCase):
    """Test fetch_bulk_vm_prices with mocked API responses."""

    @patch("dbx_model_planner.adapters.azure.price_cache.fetch_azure_retail_prices")
    def test_returns_prices_for_matched_nodes(self, mock_fetch: MagicMock) -> None:
        from dbx_model_planner.adapters.azure.pricing import AzureRetailPriceRecord

        mock_fetch.return_value = [
            AzureRetailPriceRecord(
                currency_code="USD",
                retail_price=3.67,
                unit_price=3.67,
                arm_region_name="westeurope",
                arm_sku_name="Standard_NC24ads_A100_v4",
                normalized_region="westeurope",
                normalized_price_type="consumption",
                is_primary_meter_region=True,
            ),
        ]

        result = fetch_bulk_vm_prices(
            ["Standard_NC24ads_A100_v4"],
            "westeurope",
        )

        self.assertIn("Standard_NC24ads_A100_v4", result)
        rate, sku, record = result["Standard_NC24ads_A100_v4"]
        self.assertEqual(rate, 3.67)
        self.assertEqual(sku, "Standard_NC24ads_A100_v4")

    @patch("dbx_model_planner.adapters.azure.price_cache.fetch_azure_retail_prices")
    def test_returns_empty_for_no_match(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = []

        result = fetch_bulk_vm_prices(
            ["Standard_NC24ads_A100_v4"],
            "westeurope",
        )
        self.assertEqual(result, {})

    def test_returns_empty_for_invalid_region(self) -> None:
        # normalize_azure_region returns None for garbage strings
        result = fetch_bulk_vm_prices([], "")
        self.assertEqual(result, {})


class RefreshPriceCacheTests(unittest.TestCase):
    """Test refresh_price_cache with mocked fetch."""

    @patch("dbx_model_planner.adapters.azure.price_cache.fetch_bulk_vm_prices")
    def test_builds_cache_from_fetched_prices(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {
            "node1": (3.67, "Standard_NC24ads_A100_v4", None),
            "node2": (7.34, "Standard_ND96asr_v4", None),
        }

        cache = refresh_price_cache(
            ["node1", "node2"],
            "westeurope",
            persist=False,
        )

        self.assertEqual(cache.region, "westeurope")
        self.assertEqual(len(cache.entries), 2)
        self.assertFalse(cache.is_expired)
        self.assertTrue(cache.is_populated)
        self.assertEqual(cache.get_rate("node1"), 3.67)
        self.assertEqual(cache.get_rate("node2"), 7.34)

    @patch("dbx_model_planner.adapters.azure.price_cache.fetch_bulk_vm_prices")
    def test_empty_result_creates_empty_cache(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {}

        cache = refresh_price_cache(
            ["node1"],
            "westeurope",
            persist=False,
        )

        self.assertEqual(len(cache.entries), 0)
        # Cache was fetched (fetched_at is set) but has no entries
        self.assertFalse(cache.is_expired)


if __name__ == "__main__":
    unittest.main()
