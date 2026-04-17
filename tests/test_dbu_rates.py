"""Tests for the Azure DBU rates parser module."""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from dbx_model_planner.adapters.azure.dbu_rates import (
    DbuRateCache,
    DbuRateEntry,
    _instance_name_to_node_type_id,
    build_dbu_rate_cache,
    fetch_dbu_unit_prices,
    load_dbu_cache,
    parse_dbu_rates_from_html,
    save_dbu_cache,
)
from dbx_model_planner.collectors.databricks.inventory import enrich_dbu_rates
from dbx_model_planner.domain import Cloud, WorkspaceComputeProfile


# -- Minimal HTML fixture replicating the Azure pricing page structure ------

_GPU_HTML = """
<h3>GPU</h3>
<div class="row row-size3 column">
    <h4 class="text-heading3">NC A100 v4 series</h4>
</div>
<div class="row row-size3 column">
    <div class="data-table-base data-table--pricing">
        <table class="data-table__table data-table__table--pricing">
            <thead><tr>
                <th>Instance</th><th>vCPU(s)</th><th>RAM</th>
                <th>DBU Count</th><th>DBU Price</th>
            </tr></thead>
            <tbody>
                <tr>
                    <td>NC24ads A100 v4</td>
                    <td>24</td>
                    <td>220.00 GiB</td>
                    <td>
10.00                    </td>
                    <td><span class='price-data'>$5.50</span></td>
                </tr>
                <tr>
                    <td>NC48ads A100 v4</td>
                    <td>48</td>
                    <td>440.00 GiB</td>
                    <td>
20.00                    </td>
                    <td><span class='price-data'>$11.00</span></td>
                </tr>
            </tbody>
        </table>
    </div>
</div>
<div class="row row-size3 column">
    <h4 class="text-heading3">NCas_T4_v3 series</h4>
</div>
<div class="row row-size3 column">
    <div class="data-table-base data-table--pricing">
        <table class="data-table__table data-table__table--pricing">
            <thead><tr>
                <th>Instance</th><th>vCPU(s)</th><th>RAM</th>
                <th>DBU Count</th><th>DBU Price</th>
            </tr></thead>
            <tbody>
                <tr>
                    <td>NC4as T4 v3</td>
                    <td>4</td>
                    <td>28.00 GiB</td>
                    <td>
1.00                    </td>
                    <td><span class='price-data'>$0.55</span></td>
                </tr>
            </tbody>
        </table>
    </div>
</div>
<h3>Confidential Compute</h3>
"""

_GENERAL_PURPOSE_HTML = """
<h3>General purpose</h3>
<div class="row row-size3 column">
    <h4 class="text-heading3">DSv2 series</h4>
</div>
<div class="row row-size3 column">
    <div class="data-table-base data-table--pricing">
        <table class="data-table__table data-table__table--pricing">
            <thead><tr>
                <th>Instance</th><th>vCPU(s)</th><th>RAM</th>
                <th>DBU Count</th><th>DBU Price</th>
            </tr></thead>
            <tbody>
                <tr>
                    <td>DS3 v2</td>
                    <td>4</td>
                    <td>14.00 GiB</td>
                    <td>
0.75                    </td>
                    <td><span class='price-data'>$-</span></td>
                </tr>
            </tbody>
        </table>
    </div>
</div>
<h3>Memory optimized</h3>
<h3>Storage optimized</h3>
"""

_FULL_HTML = _GENERAL_PURPOSE_HTML + _GPU_HTML


class InstanceNameMappingTests(unittest.TestCase):
    """Test instance name to node_type_id conversion."""

    def test_a100_mapping(self) -> None:
        self.assertEqual(
            _instance_name_to_node_type_id("NC24ads A100 v4"),
            "Standard_NC24ads_A100_v4",
        )

    def test_t4_mapping(self) -> None:
        self.assertEqual(
            _instance_name_to_node_type_id("NC4as T4 v3"),
            "Standard_NC4as_T4_v3",
        )

    def test_simple_mapping(self) -> None:
        self.assertEqual(
            _instance_name_to_node_type_id("NC12"),
            "Standard_NC12",
        )

    def test_v3_mapping(self) -> None:
        self.assertEqual(
            _instance_name_to_node_type_id("NC6s v3"),
            "Standard_NC6s_v3",
        )

    def test_general_purpose_mapping(self) -> None:
        self.assertEqual(
            _instance_name_to_node_type_id("DS3 v2"),
            "Standard_DS3_v2",
        )

    def test_whitespace_handling(self) -> None:
        self.assertEqual(
            _instance_name_to_node_type_id("  NC24ads A100 v4  "),
            "Standard_NC24ads_A100_v4",
        )


class ParseDbuRatesTests(unittest.TestCase):
    """Test HTML parsing of DBU rates."""

    def test_parse_gpu_section(self) -> None:
        entries = parse_dbu_rates_from_html(_GPU_HTML)
        self.assertEqual(len(entries), 3)

        by_name = {e.instance_name: e for e in entries}
        self.assertIn("NC24ads A100 v4", by_name)
        self.assertIn("NC48ads A100 v4", by_name)
        self.assertIn("NC4as T4 v3", by_name)

        a100 = by_name["NC24ads A100 v4"]
        self.assertEqual(a100.dbu_per_hour, 10.0)
        self.assertEqual(a100.vcpu_count, 24)
        self.assertEqual(a100.node_type_id, "Standard_NC24ads_A100_v4")

        a100_48 = by_name["NC48ads A100 v4"]
        self.assertEqual(a100_48.dbu_per_hour, 20.0)

        t4 = by_name["NC4as T4 v3"]
        self.assertEqual(t4.dbu_per_hour, 1.0)
        self.assertEqual(t4.vcpu_count, 4)

    def test_parse_general_purpose(self) -> None:
        entries = parse_dbu_rates_from_html(_GENERAL_PURPOSE_HTML)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].instance_name, "DS3 v2")
        self.assertEqual(entries[0].dbu_per_hour, 0.75)

    def test_parse_full_html(self) -> None:
        entries = parse_dbu_rates_from_html(_FULL_HTML)
        self.assertEqual(len(entries), 4)  # 1 general + 3 GPU

    def test_parse_empty_html(self) -> None:
        entries = parse_dbu_rates_from_html("<html><body>nothing</body></html>")
        self.assertEqual(entries, [])

    def test_no_duplicate_entries(self) -> None:
        # If the same instance appears in multiple sections, it should only appear once
        double_html = _GPU_HTML + _GPU_HTML.replace("Confidential Compute", "Other")
        entries = parse_dbu_rates_from_html(double_html)
        names = [e.instance_name for e in entries]
        self.assertEqual(len(names), len(set(names)))


class DbuRateCacheTests(unittest.TestCase):
    """Test DbuRateCache behavior."""

    def test_empty_cache_is_expired(self) -> None:
        cache = DbuRateCache()
        self.assertTrue(cache.is_expired)
        self.assertFalse(cache.is_populated)

    def test_populated_cache(self) -> None:
        cache = DbuRateCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["Standard_NC24ads_A100_v4"] = DbuRateEntry(
            instance_name="NC24ads A100 v4",
            node_type_id="Standard_NC24ads_A100_v4",
            dbu_per_hour=10.0,
            vcpu_count=24,
            ram_gib="220.00 GiB",
        )
        self.assertFalse(cache.is_expired)
        self.assertTrue(cache.is_populated)

    def test_get_rate(self) -> None:
        cache = DbuRateCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["Standard_NC24ads_A100_v4"] = DbuRateEntry(
            instance_name="NC24ads A100 v4",
            node_type_id="Standard_NC24ads_A100_v4",
            dbu_per_hour=10.0,
            vcpu_count=24,
            ram_gib="220.00 GiB",
        )
        self.assertEqual(cache.get_rate("Standard_NC24ads_A100_v4"), 10.0)
        self.assertIsNone(cache.get_rate("Standard_UNKNOWN"))

    def test_expired_cache_returns_none(self) -> None:
        cache = DbuRateCache(
            fetched_at=time.time() - 100000,
            ttl_seconds=100,
        )
        cache.entries["node1"] = DbuRateEntry(
            instance_name="NC12",
            node_type_id="node1",
            dbu_per_hour=3.0,
            vcpu_count=12,
            ram_gib="112 GiB",
        )
        self.assertTrue(cache.is_expired)
        self.assertIsNone(cache.get_rate("node1"))

    def test_as_dict(self) -> None:
        cache = DbuRateCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["n1"] = DbuRateEntry("NC12", "n1", 3.0, 12, "112 GiB")
        cache.entries["n2"] = DbuRateEntry("NC24", "n2", 6.0, 24, "224 GiB")
        d = cache.as_dict()
        # Non-GPU-model keys have no aliases to add
        self.assertIn("n1", d)
        self.assertIn("n2", d)
        self.assertEqual(d["n1"], 3.0)
        self.assertEqual(d["n2"], 6.0)

    def test_as_dict_expired_returns_empty(self) -> None:
        cache = DbuRateCache(fetched_at=time.time() - 100000, ttl_seconds=100)
        cache.entries["n1"] = DbuRateEntry("NC12", "n1", 3.0, 12, "112 GiB")
        self.assertEqual(cache.as_dict(), {})


class BuildDbuRateCacheTests(unittest.TestCase):
    """Test build_dbu_rate_cache from entries."""

    def test_build_from_entries(self) -> None:
        entries = [
            DbuRateEntry("NC24ads A100 v4", "Standard_NC24ads_A100_v4", 10.0, 24, "220 GiB"),
            DbuRateEntry("NC4as T4 v3", "Standard_NC4as_T4_v3", 1.0, 4, "28 GiB"),
        ]
        cache = build_dbu_rate_cache(entries, ttl_seconds=3600)
        self.assertTrue(cache.is_populated)
        self.assertEqual(len(cache.entries), 2)
        self.assertEqual(cache.get_rate("Standard_NC24ads_A100_v4"), 10.0)
        self.assertEqual(cache.get_rate("Standard_NC4as_T4_v3"), 1.0)


class DbuCachePersistenceTests(unittest.TestCase):
    """Test save/load of DBU rate cache."""

    def test_round_trip(self) -> None:
        import tempfile
        cache = DbuRateCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["Standard_NC24ads_A100_v4"] = DbuRateEntry(
            instance_name="NC24ads A100 v4",
            node_type_id="Standard_NC24ads_A100_v4",
            dbu_per_hour=10.0,
            vcpu_count=24,
            ram_gib="220.00 GiB",
        )
        cache.entries["Standard_NC4as_T4_v3"] = DbuRateEntry(
            instance_name="NC4as T4 v3",
            node_type_id="Standard_NC4as_T4_v3",
            dbu_per_hour=1.0,
            vcpu_count=4,
            ram_gib="28.00 GiB",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dbu_cache.json"
            save_dbu_cache(cache, path)

            loaded = load_dbu_cache(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.entries), 2)
            self.assertEqual(loaded.get_rate("Standard_NC24ads_A100_v4"), 10.0)
            self.assertEqual(loaded.get_rate("Standard_NC4as_T4_v3"), 1.0)

    def test_load_missing_file(self) -> None:
        result = load_dbu_cache(Path("/nonexistent/path.json"))
        self.assertIsNone(result)

    def test_load_corrupt_file(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("not json at all")
            result = load_dbu_cache(path)
            self.assertIsNone(result)


class EnrichDbuRatesTests(unittest.TestCase):
    """Test the enrich_dbu_rates function."""

    def test_enriches_matching_nodes(self) -> None:
        nodes = [
            WorkspaceComputeProfile(
                node_type_id="Standard_NC24ads_A100_v4",
                cloud=Cloud.AZURE,
                gpu_count=1,
                vcpu_count=24,
            ),
            WorkspaceComputeProfile(
                node_type_id="Standard_NC4as_T4_v3",
                cloud=Cloud.AZURE,
                gpu_count=1,
                vcpu_count=4,
            ),
            WorkspaceComputeProfile(
                node_type_id="Standard_UNKNOWN",
                cloud=Cloud.AZURE,
                gpu_count=0,
                vcpu_count=8,
            ),
        ]
        rates = {
            "Standard_NC24ads_A100_v4": 10.0,
            "Standard_NC4as_T4_v3": 1.0,
        }
        count = enrich_dbu_rates(nodes, rates)
        self.assertEqual(count, 2)
        self.assertEqual(nodes[0].dbu_per_hour, 10.0)
        self.assertEqual(nodes[1].dbu_per_hour, 1.0)
        self.assertIsNone(nodes[2].dbu_per_hour)

    def test_enriches_zero_when_no_matches(self) -> None:
        nodes = [
            WorkspaceComputeProfile(
                node_type_id="Standard_UNKNOWN",
                cloud=Cloud.AZURE,
                gpu_count=0,
            ),
        ]
        count = enrich_dbu_rates(nodes, {"Standard_OTHER": 5.0})
        self.assertEqual(count, 0)
        self.assertIsNone(nodes[0].dbu_per_hour)


class GpuModelAliasTests(unittest.TestCase):
    """Test GPU model token alias matching for node_type_id mismatches."""

    def test_get_rate_alias_nd96asr_v4(self) -> None:
        """ND96asr_v4 in workspace should match ND96asr_A100_v4 in cache."""
        cache = DbuRateCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["Standard_ND96asr_A100_v4"] = DbuRateEntry(
            instance_name="ND96asr A100 v4",
            node_type_id="Standard_ND96asr_A100_v4",
            dbu_per_hour=44.0,
            vcpu_count=96,
            ram_gib="900 GiB",
        )
        # Exact match
        self.assertEqual(cache.get_rate("Standard_ND96asr_A100_v4"), 44.0)
        # Alias match (Databricks drops "A100")
        self.assertEqual(cache.get_rate("Standard_ND96asr_v4"), 44.0)

    def test_as_dict_includes_aliases(self) -> None:
        """as_dict() should include short-form aliases for GPU model names."""
        cache = DbuRateCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["Standard_NC24ads_A100_v4"] = DbuRateEntry(
            "NC24ads A100 v4", "Standard_NC24ads_A100_v4", 10.0, 24, "220 GiB",
        )
        cache.entries["Standard_ND96asr_A100_v4"] = DbuRateEntry(
            "ND96asr A100 v4", "Standard_ND96asr_A100_v4", 44.0, 96, "900 GiB",
        )
        d = cache.as_dict()
        # Full names present
        self.assertIn("Standard_NC24ads_A100_v4", d)
        self.assertIn("Standard_ND96asr_A100_v4", d)
        # Short aliases present
        self.assertIn("Standard_NC24ads_v4", d)
        self.assertIn("Standard_ND96asr_v4", d)
        # Values match
        self.assertEqual(d["Standard_ND96asr_v4"], 44.0)
        self.assertEqual(d["Standard_NC24ads_v4"], 10.0)

    def test_alias_does_not_override_existing_entry(self) -> None:
        """If the short form already exists as a real entry, don't overwrite it."""
        cache = DbuRateCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["Standard_NC24ads_A100_v4"] = DbuRateEntry(
            "NC24ads A100 v4", "Standard_NC24ads_A100_v4", 10.0, 24, "220 GiB",
        )
        # Hypothetical: short form is also a real entry with different rate
        cache.entries["Standard_NC24ads_v4"] = DbuRateEntry(
            "NC24ads v4", "Standard_NC24ads_v4", 99.0, 24, "220 GiB",
        )
        d = cache.as_dict()
        # Real entry takes precedence
        self.assertEqual(d["Standard_NC24ads_v4"], 99.0)

    def test_enrich_with_alias_from_as_dict(self) -> None:
        """enrich_dbu_rates should work with aliases from cache.as_dict()."""
        cache = DbuRateCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["Standard_ND96asr_A100_v4"] = DbuRateEntry(
            "ND96asr A100 v4", "Standard_ND96asr_A100_v4", 44.0, 96, "900 GiB",
        )
        nodes = [
            WorkspaceComputeProfile(
                node_type_id="Standard_ND96asr_v4",
                cloud=Cloud.AZURE,
                gpu_count=8,
                vcpu_count=96,
            ),
        ]
        count = enrich_dbu_rates(nodes, cache.as_dict())
        self.assertEqual(count, 1)
        self.assertEqual(nodes[0].dbu_per_hour, 44.0)

    def test_no_alias_for_non_gpu_token(self) -> None:
        """Non-GPU tokens like 'DS3' should not create spurious aliases."""
        cache = DbuRateCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["Standard_DS3_v2"] = DbuRateEntry(
            "DS3 v2", "Standard_DS3_v2", 0.75, 4, "14 GiB",
        )
        d = cache.as_dict()
        # Should only have the original key, no aliases
        ds_keys = [k for k in d if "DS3" in k]
        self.assertEqual(ds_keys, ["Standard_DS3_v2"])


class DbuUnitPriceCacheTests(unittest.TestCase):
    """Test DbuRateCache with per-DBU unit prices."""

    def test_get_unit_price_returns_cached_price(self) -> None:
        """get_unit_price should return the correct per-DBU price."""
        cache = DbuRateCache(
            fetched_at=time.time(),
            ttl_seconds=3600,
            dbu_unit_prices={"all_purpose": 0.4774, "jobs_compute": 0.2604},
            unit_price_currency="EUR",
        )
        cache.entries["n1"] = DbuRateEntry("NC12", "n1", 3.0, 12, "112 GiB")
        self.assertEqual(cache.get_unit_price("all_purpose", "EUR"), 0.4774)
        self.assertEqual(cache.get_unit_price("jobs_compute", "EUR"), 0.2604)

    def test_get_unit_price_wrong_currency_returns_none(self) -> None:
        """get_unit_price should return None if currency doesn't match."""
        cache = DbuRateCache(
            fetched_at=time.time(),
            ttl_seconds=3600,
            dbu_unit_prices={"all_purpose": 0.4774},
            unit_price_currency="EUR",
        )
        cache.entries["n1"] = DbuRateEntry("NC12", "n1", 3.0, 12, "112 GiB")
        self.assertIsNone(cache.get_unit_price("all_purpose", "USD"))

    def test_get_unit_price_missing_workload_returns_none(self) -> None:
        """get_unit_price should return None for unknown workload type."""
        cache = DbuRateCache(
            fetched_at=time.time(),
            ttl_seconds=3600,
            dbu_unit_prices={"all_purpose": 0.4774},
            unit_price_currency="EUR",
        )
        cache.entries["n1"] = DbuRateEntry("NC12", "n1", 3.0, 12, "112 GiB")
        self.assertIsNone(cache.get_unit_price("jobs_compute", "EUR"))

    def test_get_unit_price_empty_prices_returns_none(self) -> None:
        """get_unit_price should return None if no prices are cached."""
        cache = DbuRateCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["n1"] = DbuRateEntry("NC12", "n1", 3.0, 12, "112 GiB")
        self.assertIsNone(cache.get_unit_price("all_purpose", "EUR"))

    def test_get_unit_price_expired_returns_none(self) -> None:
        """get_unit_price should return None when cache is expired."""
        cache = DbuRateCache(
            fetched_at=time.time() - 100000,
            ttl_seconds=100,
            dbu_unit_prices={"all_purpose": 0.4774},
            unit_price_currency="EUR",
        )
        cache.entries["n1"] = DbuRateEntry("NC12", "n1", 3.0, 12, "112 GiB")
        self.assertIsNone(cache.get_unit_price("all_purpose", "EUR"))

    def test_save_load_preserves_unit_prices(self) -> None:
        """save/load round-trip should preserve dbu_unit_prices."""
        import tempfile

        cache = DbuRateCache(
            fetched_at=time.time(),
            ttl_seconds=3600,
            dbu_unit_prices={"all_purpose": 0.4774, "jobs_compute": 0.2604},
            unit_price_currency="EUR",
        )
        cache.entries["Standard_NC24ads_A100_v4"] = DbuRateEntry(
            instance_name="NC24ads A100 v4",
            node_type_id="Standard_NC24ads_A100_v4",
            dbu_per_hour=10.0,
            vcpu_count=24,
            ram_gib="220.00 GiB",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dbu_cache.json"
            save_dbu_cache(cache, path)

            loaded = load_dbu_cache(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.dbu_unit_prices, {"all_purpose": 0.4774, "jobs_compute": 0.2604})
            self.assertEqual(loaded.unit_price_currency, "EUR")
            self.assertEqual(loaded.get_unit_price("all_purpose", "EUR"), 0.4774)

    def test_load_old_cache_without_unit_prices(self) -> None:
        """Loading a cache file without dbu_unit_prices should work (backward compat)."""
        import tempfile

        old_data = {
            "fetched_at": time.time(),
            "ttl_seconds": 3600,
            "entries": {
                "n1": {
                    "instance_name": "NC12",
                    "node_type_id": "n1",
                    "dbu_per_hour": 3.0,
                    "vcpu_count": 12,
                    "ram_gib": "112 GiB",
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old_cache.json"
            path.write_text(json.dumps(old_data), encoding="utf-8")

            loaded = load_dbu_cache(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.dbu_unit_prices, {})
            self.assertIsNone(loaded.unit_price_currency)
            self.assertEqual(loaded.get_rate("n1"), 3.0)

    def test_save_load_round_trip(self) -> None:
        """save/load round-trip should preserve all fields."""
        import tempfile

        cache = DbuRateCache(fetched_at=time.time(), ttl_seconds=3600)
        cache.entries["Standard_NC24ads_A100_v4"] = DbuRateEntry(
            instance_name="NC24ads A100 v4",
            node_type_id="Standard_NC24ads_A100_v4",
            dbu_per_hour=10.0,
            vcpu_count=24,
            ram_gib="220.00 GiB",
        )
        cache.entries["Standard_DS3_v2"] = DbuRateEntry(
            instance_name="DS3 v2",
            node_type_id="Standard_DS3_v2",
            dbu_per_hour=0.75,
            vcpu_count=4,
            ram_gib="14.00 GiB",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dbu_cache.json"
            save_dbu_cache(cache, path)

            loaded = load_dbu_cache(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.entries["Standard_NC24ads_A100_v4"].dbu_per_hour, 10.0)
            self.assertEqual(loaded.entries["Standard_DS3_v2"].dbu_per_hour, 0.75)


class FetchDbuUnitPricesTests(unittest.TestCase):
    """Test fetch_dbu_unit_prices with mocked API responses."""

    def _mock_records(self, items: list[dict]) -> list:
        """Build mock AzureRetailPriceRecord objects from dicts."""
        records = []
        for item in items:
            record = MagicMock()
            record.meter_name = item.get("meterName", "")
            record.unit_price = item.get("unitPrice", 0.0)
            records.append(record)
        return records

    @patch("dbx_model_planner.adapters.azure.pricing.fetch_azure_retail_prices")
    def test_parses_all_purpose_and_jobs(self, mock_fetch: MagicMock) -> None:
        """Should extract per-DBU prices for all_purpose and jobs_compute."""
        mock_fetch.return_value = self._mock_records([
            {"meterName": "Premium All-purpose Compute DBU", "unitPrice": 0.4774},
            {"meterName": "Premium Jobs Compute DBU", "unitPrice": 0.2604},
            {"meterName": "Premium Jobs Light Compute DBU", "unitPrice": 0.1200},
            {"meterName": "Standard All-purpose Compute DBU", "unitPrice": 0.40},  # Ignored (not Premium)
        ])
        result = fetch_dbu_unit_prices(region="westeurope", currency_code="EUR")
        self.assertEqual(result["all_purpose"], 0.4774)
        self.assertEqual(result["jobs_compute"], 0.2604)
        self.assertEqual(result["jobs_light"], 0.1200)
        self.assertNotIn("standard", result)

    @patch("dbx_model_planner.adapters.azure.pricing.fetch_azure_retail_prices")
    def test_ignores_non_dbu_meters(self, mock_fetch: MagicMock) -> None:
        """Should skip records that don't contain 'dbu' in meter name."""
        mock_fetch.return_value = self._mock_records([
            {"meterName": "Premium All-purpose Compute", "unitPrice": 1.50},  # No "DBU"
            {"meterName": "Premium All-purpose Compute DBU", "unitPrice": 0.4774},
        ])
        result = fetch_dbu_unit_prices(region="westeurope", currency_code="EUR")
        self.assertEqual(len(result), 1)
        self.assertEqual(result["all_purpose"], 0.4774)

    @patch("dbx_model_planner.adapters.azure.pricing.fetch_azure_retail_prices")
    def test_empty_response(self, mock_fetch: MagicMock) -> None:
        """Should return empty dict when API returns no records."""
        mock_fetch.return_value = []
        result = fetch_dbu_unit_prices(region="westeurope", currency_code="EUR")
        self.assertEqual(result, {})

    @patch("dbx_model_planner.adapters.azure.pricing.fetch_azure_retail_prices")
    def test_api_failure_returns_empty(self, mock_fetch: MagicMock) -> None:
        """Should return empty dict on API failure."""
        mock_fetch.side_effect = RuntimeError("API error")
        result = fetch_dbu_unit_prices(region="westeurope", currency_code="EUR")
        self.assertEqual(result, {})

    @patch("dbx_model_planner.adapters.azure.pricing.fetch_azure_retail_prices")
    def test_skips_zero_price_records(self, mock_fetch: MagicMock) -> None:
        """Should skip records with zero or negative unit prices."""
        mock_fetch.return_value = self._mock_records([
            {"meterName": "Premium All-purpose Compute DBU", "unitPrice": 0.0},
            {"meterName": "Premium Jobs Compute DBU", "unitPrice": 0.2604},
        ])
        result = fetch_dbu_unit_prices(region="westeurope", currency_code="EUR")
        self.assertNotIn("all_purpose", result)
        self.assertEqual(result["jobs_compute"], 0.2604)

    def test_invalid_region_returns_empty(self) -> None:
        """Should return empty dict for a region that can't be normalized."""
        result = fetch_dbu_unit_prices(region="", currency_code="EUR")
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
