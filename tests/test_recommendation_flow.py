from __future__ import annotations

import unittest

from dbx_model_planner.config import AppConfig
from dbx_model_planner.domain import (
    Cloud,
    ModelFamily,
    ModelModality,
    ModelProfile,
    WorkloadProfile,
    WorkspaceComputeProfile,
    WorkspaceInventorySnapshot,
    WorkspacePolicyProfile,
)
from dbx_model_planner.planners import recommend_compute_for_model, recommend_models_for_compute


def _mock_inventory() -> WorkspaceInventorySnapshot:
    """Build a lightweight workspace inventory with known GPU and CPU node types."""
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
        runtimes=[],
        policies=[
            WorkspacePolicyProfile(policy_id="p1", policy_name="default", allowed_node_types=["Standard_NC6s_v3"]),
            WorkspacePolicyProfile(policy_id="p2", policy_name="cpu-only", allowed_node_types=["Standard_D3_v2"]),
        ],
    )


def _mock_llm() -> ModelProfile:
    return ModelProfile(
        model_id="meta-llama/Mock-LLM-7B",
        family=ModelFamily.LLM,
        modality=ModelModality.TEXT,
        source="mock",
        task="text-generation",
        parameter_count=7_000_000_000,
        context_length=32768,
        quantization_options=["fp16", "int8", "int4"],
    )


def _mock_embedding() -> ModelProfile:
    return ModelProfile(
        model_id="mock/embedding-335M",
        family=ModelFamily.EMBEDDING,
        modality=ModelModality.TEXT_EMBEDDING,
        source="mock",
        task="feature-extraction",
        parameter_count=335_000_000,
        context_length=512,
        quantization_options=["fp16"],
    )


def _mock_vlm() -> ModelProfile:
    return ModelProfile(
        model_id="mock/VLM-2B",
        family=ModelFamily.VLM,
        modality=ModelModality.IMAGE_TEXT,
        source="mock",
        task="image-text-to-text",
        parameter_count=2_000_000_000,
        context_length=32768,
        quantization_options=["bf16", "int4"],
    )


class RecommendationFlowTests(unittest.TestCase):
    def test_mock_llm_recommends_gpu_compute(self) -> None:
        config = AppConfig()
        snapshot = _mock_inventory()
        model = _mock_llm()

        recommendation = recommend_compute_for_model(
            config=config,
            inventory=snapshot,
            model=model,
            workload=WorkloadProfile(workload_name="offline-fit", online=True),
        )

        self.assertEqual(recommendation.candidates[0].compute.node_type_id, "Standard_NC6s_v3")
        self.assertIn("Recommended compute", recommendation.summary)

    def test_compute_fit_reports_embedding_range(self) -> None:
        config = AppConfig()
        snapshot = _mock_inventory()
        gpu_compute = next(item for item in snapshot.compute if item.node_type_id == "Standard_NC6s_v3")
        models = [
            _mock_llm(),
            _mock_embedding(),
            _mock_vlm(),
        ]

        report = recommend_models_for_compute(config=config, compute=gpu_compute, models=models)

        self.assertIn("embedding", report.model_family_ranges)
        self.assertIn("Compute Standard_NC6s_v3", report.summary)


if __name__ == "__main__":
    unittest.main()
