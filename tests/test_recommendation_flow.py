from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from dbx_model_planner.catalog import resolve_example_model
from dbx_model_planner.collectors.databricks import DatabricksInventoryCollector
from dbx_model_planner.config import AppConfig
from dbx_model_planner.domain import WorkloadProfile
from dbx_model_planner.planners import build_deployment_hint, recommend_compute_for_model, recommend_models_for_compute
from dbx_model_planner.presentation import render_compute_fit, render_deployment_hint, render_inventory, render_model_recommendation
from dbx_model_planner.runtime import build_runtime_context
from dbx_model_planner.storage import SQLiteSnapshotStore


class RecommendationFlowTests(unittest.TestCase):
    def test_example_llm_recommends_gpu_compute(self) -> None:
        config = AppConfig()
        snapshot = DatabricksInventoryCollector().collect_snapshot()
        model = resolve_example_model("mistral-7b-instruct").model_profile

        recommendation = recommend_compute_for_model(
            config=config,
            inventory=snapshot,
            model=model,
            workload=WorkloadProfile(workload_name="offline-fit", online=True),
        )

        self.assertEqual(recommendation.candidates[0].compute.node_type_id, "Standard_NC6s_v3")
        self.assertIn("Recommended compute", recommendation.summary)
        self.assertIn("Standard_NC6s_v3", render_model_recommendation(recommendation))

    def test_compute_fit_reports_embedding_range(self) -> None:
        config = AppConfig()
        snapshot = DatabricksInventoryCollector().collect_snapshot()
        gpu_compute = next(item for item in snapshot.compute if item.node_type_id == "Standard_NC6s_v3")
        models = [
            resolve_example_model("mistral-7b-instruct").model_profile,
            resolve_example_model("bge-large-en-v1.5").model_profile,
            resolve_example_model("qwen2-vl-2b-instruct").model_profile,
        ]

        report = recommend_models_for_compute(config=config, compute=gpu_compute, models=models)

        self.assertIn("embedding", report.model_family_ranges)
        self.assertIn("Compute Standard_NC6s_v3", report.summary)
        self.assertIn("Example models:", render_compute_fit(report))

    def test_deployment_hint_uses_catalog_defaults(self) -> None:
        config = AppConfig()
        config.catalog.catalog = "main"
        config.catalog.schema = "serving"
        config.catalog.volume = "artifacts"
        snapshot = DatabricksInventoryCollector().collect_snapshot()
        model = resolve_example_model("qwen2-vl-2b-instruct").model_profile
        recommendation = recommend_compute_for_model(
            config=config,
            inventory=snapshot,
            model=model,
            workload=WorkloadProfile(workload_name="deploy", online=True),
        )

        hint = build_deployment_hint(config, snapshot, model, recommendation)

        self.assertEqual(hint.recommended_node_type_id, "Standard_NC6s_v3")
        self.assertTrue(hint.target)
        self.assertEqual(
            hint.target.volume_path,
            "/Volumes/main/serving/artifacts/Qwen--Qwen2-VL-2B-Instruct",
        )
        self.assertIn("Volume path:", render_deployment_hint(hint))

    def test_runtime_context_and_snapshot_store_work_with_inventory(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            context = build_runtime_context(AppConfig(), data_dir=tmp_dir)
            store = SQLiteSnapshotStore(context.paths.snapshot_db_path)
            snapshot = DatabricksInventoryCollector().collect_snapshot()
            store.save_inventory_snapshot(snapshot)

            loaded = store.load_inventory_snapshot()

        self.assertEqual(snapshot, loaded)
        self.assertIn("Workspace:", render_inventory(loaded))


if __name__ == "__main__":
    unittest.main()
