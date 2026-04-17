from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from dbx_model_planner.auth import DatabricksCredentials, HuggingFaceCredentials
from dbx_model_planner.collectors.databricks import DatabricksInventoryCollection
from dbx_model_planner.config import AppConfig
from dbx_model_planner.domain import (
    Cloud,
    RuntimeProfile,
    WorkspaceComputeProfile,
    WorkspaceInventorySnapshot,
    WorkspacePolicyProfile,
)
from dbx_model_planner.terminal_app import run_terminal_app


_MOCK_DBX_CREDS = DatabricksCredentials(
    host="https://adb-mock.azuredatabricks.net",
    token="dapi_mock_token",
)

_MOCK_HF_CREDS = HuggingFaceCredentials(token=None)


def _mock_inventory() -> WorkspaceInventorySnapshot:
    return WorkspaceInventorySnapshot(
        workspace_url="https://adb-mock.azuredatabricks.net",
        cloud=Cloud.AZURE,
        region="eastus2",
        compute=[
            WorkspaceComputeProfile(
                node_type_id="Standard_D3_v2",
                cloud=Cloud.AZURE,
                region="eastus2",
                gpu_count=0,
            ),
            WorkspaceComputeProfile(
                node_type_id="Standard_NC6s_v3",
                cloud=Cloud.AZURE,
                region="eastus2",
                gpu_family="V100",
                gpu_count=1,
                gpu_memory_gb=16.0,
            ),
        ],
        runtimes=[
            RuntimeProfile(runtime_id="15.4.x-ml-scala2.12", dbr_version="15.4 ML", ml_runtime=True, gpu_enabled=True),
            RuntimeProfile(runtime_id="15.4.x-scala2.12", dbr_version="15.4", ml_runtime=False, gpu_enabled=False),
            RuntimeProfile(runtime_id="14.3.x-ml-scala2.12", dbr_version="14.3 ML", ml_runtime=True, gpu_enabled=True),
        ],
        policies=[
            WorkspacePolicyProfile(policy_id="p1", policy_name="default", allowed_node_types=["Standard_NC6s_v3"]),
            WorkspacePolicyProfile(policy_id="p2", policy_name="cpu-only", allowed_node_types=["Standard_D3_v2"]),
        ],
    )


def _mock_collection() -> DatabricksInventoryCollection:
    return DatabricksInventoryCollection(
        snapshot=_mock_inventory(),
        notes=["Fetched 2 node types", "Fetched 3 runtime versions", "Fetched 2 cluster policies"],
    )


class TerminalAppTests(unittest.TestCase):
    @patch("dbx_model_planner.terminal_app.DatabricksInventoryCollector")
    @patch("dbx_model_planner.terminal_app.load_stored_credentials")
    @patch("dbx_model_planner.terminal_app.credential_exists", return_value=True)
    def test_terminal_app_inventory_then_quit(
        self,
        mock_exists: MagicMock,
        mock_load_creds: MagicMock,
        mock_collector_cls: MagicMock,
    ) -> None:
        mock_load_creds.return_value = (_MOCK_DBX_CREDS, _MOCK_HF_CREDS)
        mock_collector_cls.return_value.collect.return_value = _mock_collection()

        inputs = iter(["1", "q"])
        outputs: list[str] = []

        exit_code = run_terminal_app(
            config=AppConfig(),
            input_fn=lambda _: next(inputs),
            output_fn=outputs.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(any("Compute profiles: 2" in output for output in outputs))
        self.assertEqual(outputs[-1], "Bye.")

    @patch("dbx_model_planner.terminal_app.fetch_huggingface_metadata")
    @patch("dbx_model_planner.terminal_app.DatabricksInventoryCollector")
    @patch("dbx_model_planner.terminal_app.load_stored_credentials")
    @patch("dbx_model_planner.terminal_app.credential_exists", return_value=True)
    def test_terminal_app_model_fit_path(
        self,
        mock_exists: MagicMock,
        mock_load_creds: MagicMock,
        mock_collector_cls: MagicMock,
        mock_fetch: MagicMock,
    ) -> None:
        mock_load_creds.return_value = (_MOCK_DBX_CREDS, _MOCK_HF_CREDS)
        mock_collector_cls.return_value.collect.return_value = _mock_collection()

        mock_fetch.return_value = {
            "pipeline_tag": "text-generation",
            "repository_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "revision": "main",
            "commit_sha": "abc123",
            "library_name": "transformers",
            "tags": ["llm"],
            "siblings": [{"rfilename": "model.safetensors"}],
            "config": {
                "architectures": ["MistralForCausalLM"],
                "model_type": "mistral",
                "num_parameters": 7_000_000_000,
                "max_position_embeddings": 32768,
                "torch_dtype": "float16",
            },
            "tokenizer": {"model_max_length": 32768},
            "processor": {},
            "card_data": {},
            "license_name": "apache-2.0",
            "gated": False,
            "sha": "abc123",
        }

        inputs = iter(["2", "mistralai/Mistral-7B-Instruct-v0.3", "q"])
        outputs: list[str] = []

        exit_code = run_terminal_app(
            config=AppConfig(),
            input_fn=lambda _: next(inputs),
            output_fn=outputs.append,
        )

        self.assertEqual(exit_code, 0)
        mock_fetch.assert_called_once()
        self.assertTrue(any("Fetching model metadata" in output for output in outputs))


if __name__ == "__main__":
    unittest.main()
