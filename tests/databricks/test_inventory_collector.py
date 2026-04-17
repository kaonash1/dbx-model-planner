from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from dbx_model_planner.auth import DatabricksCredentials
from dbx_model_planner.collectors.databricks import DatabricksAPIError, DatabricksInventoryCollector
from dbx_model_planner.collectors.databricks.inventory import _extract_gpu_family, _extract_gpu_memory
from dbx_model_planner.domain import Cloud


_MOCK_CREDS = DatabricksCredentials(
    host="https://adb-mock.azuredatabricks.net",
    token="dapi_mock_token",
)


def _node_types_response() -> dict:
    return {
        "node_types": [
            {
                "node_type_id": "Standard_D3_v2",
                "num_gpus": 0,
                "num_cores": 4,
                "memory_mb": 14336,
            },
            {
                "node_type_id": "Standard_NC6s_v3",
                "num_gpus": 1,
                "num_cores": 6,
                "memory_mb": 114688,
            },
            {
                "node_type_id": "Standard_NC24ads_A100_v4",
                "num_gpus": 1,
                "num_cores": 24,
                "memory_mb": 220160,
            },
        ],
    }


def _spark_versions_response() -> dict:
    return {
        "versions": [
            {"key": "15.4.x-ml-scala2.12", "name": "15.4 ML (includes Apache Spark 3.5.0, GPU)"},
            {"key": "15.4.x-scala2.12", "name": "15.4 (includes Apache Spark 3.5.0)"},
        ],
    }


def _policies_response() -> dict:
    return {
        "policies": [
            {
                "policy_id": "abc123",
                "name": "default-gpu",
                "definition": {
                    "node_type_id": {
                        "type": "allowlist",
                        "values": ["Standard_NC6s_v3", "Standard_NC24ads_A100_v4"],
                    }
                },
            },
            {
                "policy_id": "def456",
                "name": "fixed-cpu",
                "definition": {
                    "node_type_id": {
                        "type": "fixed",
                        "value": "Standard_D3_v2",
                    }
                },
            },
        ],
    }


def _mock_urlopen_side_effect(responses: list[dict]):
    """Return a side_effect callable that pops responses in order."""
    queue = list(responses)

    def side_effect(request, timeout=30):
        data = queue.pop(0)
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    return side_effect


class DatabricksInventoryCollectorTests(unittest.TestCase):
    """Test the live inventory collector with mocked API responses."""

    @patch("dbx_model_planner.collectors.databricks.inventory.urllib.request.urlopen")
    def test_collect_returns_valid_snapshot(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _mock_urlopen_side_effect([
            _node_types_response(),
            _spark_versions_response(),
            _policies_response(),
        ])

        collector = DatabricksInventoryCollector(credentials=_MOCK_CREDS)
        collection = collector.collect()
        snapshot = collection.snapshot

        self.assertEqual(snapshot.cloud, Cloud.AZURE)
        self.assertEqual(len(snapshot.compute), 3)
        self.assertEqual(len(snapshot.runtimes), 2)
        self.assertEqual(len(snapshot.policies), 2)
        self.assertEqual(len(collection.notes), 3)

        node_types = {c.node_type_id: c for c in snapshot.compute}
        self.assertIn("Standard_D3_v2", node_types)
        self.assertIn("Standard_NC6s_v3", node_types)
        self.assertIn("Standard_NC24ads_A100_v4", node_types)

        # CPU node: no GPU
        d3 = node_types["Standard_D3_v2"]
        self.assertEqual(d3.gpu_count, 0)
        self.assertIsNone(d3.gpu_family)
        self.assertIsNone(d3.gpu_memory_gb)

        # V100 node
        nc6 = node_types["Standard_NC6s_v3"]
        self.assertEqual(nc6.gpu_count, 1)
        self.assertEqual(nc6.gpu_family, "V100")
        self.assertEqual(nc6.gpu_memory_gb, 16.0)

        # A100 node
        a100 = node_types["Standard_NC24ads_A100_v4"]
        self.assertEqual(a100.gpu_count, 1)
        self.assertEqual(a100.gpu_family, "A100_80")
        self.assertEqual(a100.gpu_memory_gb, 80.0)

    @patch("dbx_model_planner.collectors.databricks.inventory.urllib.request.urlopen")
    def test_collect_snapshot_convenience_method(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _mock_urlopen_side_effect([
            _node_types_response(),
            _spark_versions_response(),
            _policies_response(),
        ])

        collector = DatabricksInventoryCollector(credentials=_MOCK_CREDS)
        snapshot = collector.collect_snapshot()

        self.assertEqual(snapshot.cloud, Cloud.AZURE)
        self.assertGreater(len(snapshot.compute), 0)

    @patch("dbx_model_planner.collectors.databricks.inventory.urllib.request.urlopen")
    def test_api_error_raises_databricks_api_error(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://adb-mock.azuredatabricks.net/api/2.0/clusters/list-node-types",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )

        collector = DatabricksInventoryCollector(credentials=_MOCK_CREDS)
        with self.assertRaises(DatabricksAPIError) as ctx:
            collector.collect()
        self.assertIn("Authentication failed", str(ctx.exception))

    @patch("dbx_model_planner.collectors.databricks.inventory.urllib.request.urlopen")
    def test_runtimes_ml_and_gpu_flags(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _mock_urlopen_side_effect([
            _node_types_response(),
            _spark_versions_response(),
            _policies_response(),
        ])

        collector = DatabricksInventoryCollector(credentials=_MOCK_CREDS)
        snapshot = collector.collect_snapshot()

        runtimes = {r.runtime_id: r for r in snapshot.runtimes}

        ml_rt = runtimes["15.4.x-ml-scala2.12"]
        self.assertTrue(ml_rt.ml_runtime)
        self.assertTrue(ml_rt.gpu_enabled)

        std_rt = runtimes["15.4.x-scala2.12"]
        self.assertFalse(std_rt.ml_runtime)
        self.assertFalse(std_rt.gpu_enabled)

    @patch("dbx_model_planner.collectors.databricks.inventory.urllib.request.urlopen")
    def test_policies_allowlist_and_fixed(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _mock_urlopen_side_effect([
            _node_types_response(),
            _spark_versions_response(),
            _policies_response(),
        ])

        collector = DatabricksInventoryCollector(credentials=_MOCK_CREDS)
        snapshot = collector.collect_snapshot()

        policies = {p.policy_name: p for p in snapshot.policies}

        gpu_policy = policies["default-gpu"]
        self.assertEqual(gpu_policy.allowed_node_types, ["Standard_NC6s_v3", "Standard_NC24ads_A100_v4"])

        cpu_policy = policies["fixed-cpu"]
        self.assertEqual(cpu_policy.allowed_node_types, ["Standard_D3_v2"])


class ExtractGpuFamilyTests(unittest.TestCase):
    """Test the module-level _extract_gpu_family helper."""

    def test_a100_80_node(self) -> None:
        self.assertEqual(_extract_gpu_family("Standard_NC24ads_A100_v4"), "A100_80")

    def test_a100_40_node(self) -> None:
        self.assertEqual(_extract_gpu_family("Standard_ND96asr_v4"), "A100_40")

    def test_t4_node(self) -> None:
        self.assertEqual(_extract_gpu_family("Standard_NC8as_T4_v3"), "T4")

    def test_v100_node(self) -> None:
        self.assertEqual(_extract_gpu_family("Standard_NC6s_v3"), "V100")

    def test_k80_node(self) -> None:
        self.assertEqual(_extract_gpu_family("Standard_NC12"), "K80")
        self.assertEqual(_extract_gpu_family("Standard_NC24"), "K80")

    def test_h100_nd_node(self) -> None:
        self.assertEqual(_extract_gpu_family("Standard_ND96isr_H100_v5"), "H100")

    def test_cpu_node_returns_none(self) -> None:
        self.assertIsNone(_extract_gpu_family("Standard_D3_v2"))


class ExtractGpuMemoryTests(unittest.TestCase):
    """Test the module-level _extract_gpu_memory helper."""

    def test_v100_single_gpu(self) -> None:
        self.assertEqual(_extract_gpu_memory("Standard_NC6s_v3", 1), 16.0)

    def test_a100_single_gpu(self) -> None:
        self.assertEqual(_extract_gpu_memory("Standard_NC24ads_A100_v4", 1), 80.0)

    def test_a100_40_single_gpu(self) -> None:
        self.assertEqual(_extract_gpu_memory("Standard_ND96asr_v4", 1), 40.0)

    def test_no_gpu_returns_none(self) -> None:
        self.assertIsNone(_extract_gpu_memory("Standard_D3_v2", 0))

    def test_unknown_gpu_returns_none(self) -> None:
        self.assertIsNone(_extract_gpu_memory("Standard_UNKNOWN_v1", 1))

    def test_multi_gpu_returns_per_gpu_memory(self) -> None:
        """Multi-GPU nodes should return per-GPU memory, not total."""
        # NC96ads A100 v4 has 4 A100-80GB GPUs
        mem = _extract_gpu_memory("Standard_NC96ads_A100_v4", 4)
        self.assertEqual(mem, 80.0)  # Per-GPU, not 320

        # ND96asr v4 has 4 A100-40GB GPUs
        mem = _extract_gpu_memory("Standard_ND96asr_v4", 4)
        self.assertEqual(mem, 40.0)  # Per-GPU, not 160


if __name__ == "__main__":
    unittest.main()
