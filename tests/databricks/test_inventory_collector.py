from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dbx_model_planner.collectors.databricks import DatabricksInventoryCollector, load_inventory_manifest, parse_inventory_manifest
from dbx_model_planner.domain import Cloud, HostingMode


class DatabricksInventoryCollectorTests(unittest.TestCase):
    def test_mock_fixture_collects_domain_snapshot_and_pool_gap(self) -> None:
        collector = DatabricksInventoryCollector()

        collection = collector.collect()

        self.assertEqual(collection.snapshot.cloud, Cloud.AZURE)
        self.assertEqual(collection.snapshot.region, "eastus2")
        self.assertEqual(len(collection.snapshot.compute), 2)
        self.assertEqual(len(collection.snapshot.runtimes), 3)
        self.assertEqual(len(collection.snapshot.policies), 2)
        self.assertEqual(len(collection.pools), 2)
        self.assertTrue(any("pool field" in note for note in collection.notes))

        node_types = {compute.node_type_id: compute for compute in collection.snapshot.compute}
        self.assertIn("Standard_D3_v2", node_types)
        self.assertEqual(node_types["Standard_D3_v2"].supported_hosting_modes, [HostingMode.CLASSIC_COMPUTE, HostingMode.BATCH_COMPUTE])
        self.assertEqual(node_types["Standard_NC6s_v3"].gpu_count, 1)

    def test_parse_inventory_manifest_supports_custom_fixture_path(self) -> None:
        payload = {
            "workspace_url": "https://example.azuredatabricks.net",
            "node_types": [
                {
                    "node_type_id": "Standard_D8_v3",
                    "vcpu_count": 8,
                    "memory_gb": 32.0,
                    "runtime_ids": [],
                    "supported_hosting_modes": [],
                    "policy_ids": [],
                }
            ],
            "dbr_versions": [],
            "policies": [],
            "pools": [],
            "notes": []
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "inventory.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            manifest = load_inventory_manifest(path)
            collection = parse_inventory_manifest(manifest, source_path=path)

        self.assertEqual(collection.snapshot.workspace_url, "https://example.azuredatabricks.net")
        self.assertEqual(collection.snapshot.compute[0].node_type_id, "Standard_D8_v3")
        self.assertEqual(collection.pools, [])
        self.assertEqual(collection.notes, [])

    def test_non_mock_mode_is_explicitly_blocked(self) -> None:
        with self.assertRaises(NotImplementedError):
            DatabricksInventoryCollector(mode="live")


if __name__ == "__main__":
    unittest.main()
