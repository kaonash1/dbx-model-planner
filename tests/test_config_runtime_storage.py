from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dbx_model_planner.config import (
    AppConfig,
    WorkloadType,
    WORKLOAD_DBU_PRESETS,
    WORKLOAD_LABELS,
    _WORKLOAD_CYCLE,
    load_app_config,
    render_default_config_template,
)
from dbx_model_planner.domain.common import Cloud, EstimateSource, HostingMode, ModelFamily, ModelModality
from dbx_model_planner.domain.profiles import (
    ModelArtifactProfile,
    ModelProfile,
    RuntimeProfile,
    WorkspaceComputeProfile,
    WorkspaceInventorySnapshot,
    WorkspacePolicyProfile,
)
from dbx_model_planner.runtime.context import build_runtime_context
from dbx_model_planner.storage.sqlite import SQLiteSnapshotStore


class ConfigRuntimeStorageTests(unittest.TestCase):
    def test_load_app_config_with_file_and_env_overrides(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text(
                """
                [pricing]
                discount_rate = 0.25
                vat_rate = 0.2
                currency_code = "EUR"

                [databricks]
                dbu_rate_per_unit = 3.5

                [workspace]
                preferred_regions = ["eastus", "westeurope"]
                blocked_skus = ["Standard_NC6s_v3"]
                approved_runtimes = ["15.4"]

                [profiles]
                config = "prod"
                inventory = "sync"
                model = "hf"
                runtime = "cluster"
                """,
                encoding="utf-8",
            )

            env = {
                "DBX_MODEL_PLANNER_WORKSPACE_BLOCKED_SKUS": "Standard_NC24s_v3,Standard_NC48s_v3",
                "DBX_MODEL_PLANNER_PROFILE_RUNTIME": "runtime-override",
            }

            config = load_app_config(config_path=config_path, env=env)

            self.assertEqual(config.pricing.discount_rate, 0.25)
            self.assertEqual(config.pricing.vat_rate, 0.2)
            self.assertEqual(config.pricing.currency_code, "EUR")
            self.assertEqual(config.databricks.dbu_rate_per_unit, 3.5)
            self.assertEqual(config.workspace.preferred_regions, ["eastus", "westeurope"])
            self.assertEqual(
                config.workspace.blocked_skus,
                ["Standard_NC24s_v3", "Standard_NC48s_v3"],
            )
            self.assertEqual(config.profiles.runtime, "runtime-override")

    def test_runtime_context_uses_resolved_paths(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            data_dir = Path(tmp_dir) / "data"
            context = build_runtime_context(AppConfig(), config_path=config_path, data_dir=data_dir)

            self.assertEqual(context.paths.config_path, config_path)
            self.assertEqual(context.paths.config_dir, config_path.parent)
            self.assertEqual(context.paths.data_dir, data_dir)
            self.assertEqual(context.paths.snapshot_db_path, data_dir / "snapshots.sqlite3")
            self.assertEqual(context.profiles, AppConfig().profiles)

    def test_sqlite_snapshot_store_round_trips_inventory_and_model(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = SQLiteSnapshotStore(Path(tmp_dir) / "snapshots.sqlite3")

            inventory = WorkspaceInventorySnapshot(
                workspace_url="https://example.azuredatabricks.net",
                cloud=Cloud.AZURE,
                region="eastus",
                compute=[
                    WorkspaceComputeProfile(
                        node_type_id="Standard_NC6s_v3",
                        cloud=Cloud.AZURE,
                        region="eastus",
                        gpu_family="nvidia",
                        gpu_count=1,
                        runtime_ids=["15.4"],
                        supported_hosting_modes=[HostingMode.CLASSIC_COMPUTE],
                        policy_ids=["policy-1"],
                        availability_source=EstimateSource.DISCOVERED,
                    )
                ],
                runtimes=[
                    RuntimeProfile(
                        runtime_id="15.4",
                        dbr_version="15.4",
                        ml_runtime=True,
                        gpu_enabled=True,
                        photon_supported=False,
                        cuda_version="12.2",
                        python_version="3.11",
                        supported_engines=["spark"],
                        notes=["lts"],
                    )
                ],
                policies=[
                    WorkspacePolicyProfile(
                        policy_id="policy-1",
                        policy_name="default",
                        allowed_node_types=["Standard_NC6s_v3"],
                        blocked_node_types=[],
                        allowed_runtime_ids=["15.4"],
                        required_tags={"team": "ml"},
                    )
                ],
            )

            model = ModelProfile(
                model_id="mistralai/Mistral-7B-Instruct-v0.3",
                family=ModelFamily.LLM,
                modality=ModelModality.TEXT,
                source="hugging_face",
                task="text-generation",
                parameter_count=7_000_000_000,
                context_length=32_768,
                dtype_options=["float16", "bfloat16"],
                quantization_options=["int8"],
                capabilities=["chat", "tool-use"],
                artifacts=[
                    ModelArtifactProfile(
                        source="hugging_face",
                        repository_id="mistralai/Mistral-7B-Instruct-v0.3",
                        revision="main",
                        format="safetensors",
                        quantization="int8",
                        artifact_size_gb=13.0,
                        license_name="apache-2.0",
                        gated=False,
                        dependency_hints=["transformers"],
                        processor_required=False,
                    )
                ],
                metadata_sources=[EstimateSource.USER_PROVIDED],
            )

            inventory_record = store.save_inventory_snapshot(inventory)
            model_record = store.save_model_snapshot(model)

            self.assertEqual(store.load_inventory_snapshot(snapshot_id=inventory_record.snapshot_id), inventory)
            self.assertEqual(store.load_model_snapshot(snapshot_id=model_record.snapshot_id), model)
            self.assertEqual(store.load_inventory_snapshot(workspace_url=inventory.workspace_url), inventory)
            self.assertEqual(store.load_model_snapshot(model_id=model.model_id), model)

    def test_template_is_valid_toml_text(self) -> None:
        template = render_default_config_template()
        self.assertIn("[pricing]", template)
        self.assertIn("DATABRICKS_HOST", template)


class WorkloadTypeTests(unittest.TestCase):
    """Test WorkloadType enum and presets."""

    def test_enum_values(self) -> None:
        self.assertEqual(WorkloadType.ALL_PURPOSE.value, "all_purpose")
        self.assertEqual(WorkloadType.JOBS_COMPUTE.value, "jobs_compute")

    def test_presets_have_both_types(self) -> None:
        self.assertIn(WorkloadType.ALL_PURPOSE, WORKLOAD_DBU_PRESETS)
        self.assertIn(WorkloadType.JOBS_COMPUTE, WORKLOAD_DBU_PRESETS)
        self.assertEqual(WORKLOAD_DBU_PRESETS[WorkloadType.ALL_PURPOSE], 0.55)
        self.assertEqual(WORKLOAD_DBU_PRESETS[WorkloadType.JOBS_COMPUTE], 0.30)

    def test_labels_have_both_types(self) -> None:
        self.assertIn(WorkloadType.ALL_PURPOSE, WORKLOAD_LABELS)
        self.assertIn(WorkloadType.JOBS_COMPUTE, WORKLOAD_LABELS)
        self.assertEqual(WORKLOAD_LABELS[WorkloadType.ALL_PURPOSE], "All-Purpose Compute")
        self.assertEqual(WORKLOAD_LABELS[WorkloadType.JOBS_COMPUTE], "Jobs Compute")

    def test_cycle_order(self) -> None:
        self.assertEqual(len(_WORKLOAD_CYCLE), 2)
        self.assertEqual(_WORKLOAD_CYCLE[0], WorkloadType.ALL_PURPOSE)
        self.assertEqual(_WORKLOAD_CYCLE[1], WorkloadType.JOBS_COMPUTE)

    def test_jobs_compute_is_cheaper(self) -> None:
        """Jobs Compute per-DBU rate must be lower than All-Purpose."""
        self.assertLess(
            WORKLOAD_DBU_PRESETS[WorkloadType.JOBS_COMPUTE],
            WORKLOAD_DBU_PRESETS[WorkloadType.ALL_PURPOSE],
        )

    def test_config_loads_workload_type_from_toml(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text(
                """
                [databricks]
                dbu_rate_per_unit = 0.30
                workload_type = "jobs_compute"
                """,
                encoding="utf-8",
            )
            config = load_app_config(config_path=config_path, env={})
            self.assertEqual(config.databricks.workload_type, "jobs_compute")
            self.assertEqual(config.databricks.dbu_rate_per_unit, 0.30)

    def test_config_loads_workload_type_from_env(self) -> None:
        config = load_app_config(
            config_path=Path("/nonexistent/config.toml"),
            env={"DBX_MODEL_PLANNER_DATABRICKS_WORKLOAD_TYPE": "jobs_compute"},
        )
        self.assertEqual(config.databricks.workload_type, "jobs_compute")

    def test_default_workload_type_is_all_purpose(self) -> None:
        config = AppConfig()
        self.assertEqual(config.databricks.workload_type, "all_purpose")

    def test_template_contains_workload_type(self) -> None:
        template = render_default_config_template()
        self.assertIn("workload_type", template)


if __name__ == "__main__":
    unittest.main()

