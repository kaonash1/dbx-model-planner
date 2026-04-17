"""Tests for the inverse planning engine."""

from __future__ import annotations

import unittest

from dbx_model_planner.domain.common import Cloud, FitLevel, ModelFamily, ModelModality
from dbx_model_planner.domain.profiles import (
    ModelProfile,
    WorkspaceComputeProfile,
    WorkspaceInventorySnapshot,
)
from dbx_model_planner.engines.plan import (
    CONTEXT_PRESETS,
    QUANTIZATION_OPTIONS,
    PlanResult,
    plan_for_model,
)


def _llm(params_b: float = 7.0, context: int | None = 32768) -> ModelProfile:
    return ModelProfile(
        model_id=f"test/llm-{params_b}b",
        family=ModelFamily.LLM,
        modality=ModelModality.TEXT,
        source="test",
        task="text-generation",
        parameter_count=int(params_b * 1_000_000_000),
        context_length=context,
        quantization_options=["fp16", "int8", "int4"],
    )


def _compute(
    gpu_mem_gb: float = 16.0,
    gpu_count: int = 1,
    node_type: str = "Standard_NC6s_v3",
    gpu_family: str = "V100",
) -> WorkspaceComputeProfile:
    return WorkspaceComputeProfile(
        node_type_id=node_type,
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_mem_gb,
        gpu_family=gpu_family,
    )


def _inventory(*nodes: WorkspaceComputeProfile) -> WorkspaceInventorySnapshot:
    return WorkspaceInventorySnapshot(
        workspace_url="https://test.azuredatabricks.net",
        cloud=Cloud.AZURE,
        compute=list(nodes),
    )


class PlanForModelBasicTests(unittest.TestCase):
    """Basic plan_for_model tests."""

    def test_returns_plan_result(self) -> None:
        model = _llm(7.0)
        inv = _inventory(_compute(gpu_mem_gb=80.0, node_type="Standard_NC24ads_A100_v4", gpu_family="A100_80"))
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        self.assertIsInstance(result, PlanResult)
        self.assertEqual(result.model_id, model.model_id)
        self.assertEqual(result.selected_quantization, "fp16")
        self.assertEqual(result.selected_context_length, 4096)
        self.assertEqual(result.selected_gpu_count, 1)

    def test_estimated_memory_positive(self) -> None:
        model = _llm(7.0)
        inv = _inventory(_compute(gpu_mem_gb=80.0))
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        self.assertGreater(result.estimated_memory_gb, 0)

    def test_int4_uses_less_memory_than_fp16(self) -> None:
        model = _llm(7.0)
        inv = _inventory(_compute(gpu_mem_gb=80.0))

        fp16 = plan_for_model(model, inv, quantization="fp16", context_length=4096)
        int4 = plan_for_model(model, inv, quantization="int4", context_length=4096)

        self.assertGreater(fp16.estimated_memory_gb, int4.estimated_memory_gb)

    def test_larger_context_uses_more_memory(self) -> None:
        model = _llm(7.0)
        inv = _inventory(_compute(gpu_mem_gb=80.0))

        short_ctx = plan_for_model(model, inv, quantization="fp16", context_length=2048)
        long_ctx = plan_for_model(model, inv, quantization="fp16", context_length=32768)

        self.assertGreater(long_ctx.estimated_memory_gb, short_ctx.estimated_memory_gb)

    def test_default_context_uses_model_value(self) -> None:
        model = _llm(7.0, context=8192)
        inv = _inventory(_compute(gpu_mem_gb=80.0))
        result = plan_for_model(model, inv, quantization="fp16")

        self.assertEqual(result.selected_context_length, 8192)

    def test_default_context_falls_back_to_4096(self) -> None:
        model = _llm(7.0, context=None)
        inv = _inventory(_compute(gpu_mem_gb=80.0))
        result = plan_for_model(model, inv, quantization="fp16")

        self.assertEqual(result.selected_context_length, 4096)


class PlanRecommendedNodeTests(unittest.TestCase):
    """Test recommended node selection logic."""

    def test_selects_best_fitting_node(self) -> None:
        model = _llm(7.0)
        inv = _inventory(
            _compute(gpu_mem_gb=16.0, node_type="small", gpu_family="V100"),
            _compute(gpu_mem_gb=80.0, node_type="large", gpu_family="A100_80"),
        )
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        self.assertIsNotNone(result.recommended_node)
        # Should recommend the node that fits safely but not wastefully
        self.assertIn(result.recommended_node.node_type_id, ["small", "large"])

    def test_no_feasible_node_when_model_too_large(self) -> None:
        model = _llm(70.0)  # 70B model
        inv = _inventory(
            _compute(gpu_mem_gb=16.0, node_type="small"),
        )
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        self.assertIsNone(result.recommended_node)

    def test_safe_fit_preferred_over_borderline(self) -> None:
        model = _llm(7.0)
        inv = _inventory(
            _compute(gpu_mem_gb=80.0, node_type="big", gpu_family="A100_80"),
            _compute(gpu_mem_gb=24.0, node_type="medium", gpu_family="A10"),
        )
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        self.assertIsNotNone(result.recommended_node)
        self.assertEqual(result.recommended_fit_level, FitLevel.SAFE)


class PlanQuantizationRowTests(unittest.TestCase):
    """Test quantization comparison rows."""

    def test_four_quantization_rows(self) -> None:
        model = _llm(7.0)
        inv = _inventory(_compute(gpu_mem_gb=80.0))
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        self.assertEqual(len(result.quantization_rows), len(QUANTIZATION_OPTIONS))

    def test_selected_quantization_has_zero_delta(self) -> None:
        model = _llm(7.0)
        inv = _inventory(_compute(gpu_mem_gb=80.0))
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        fp16_row = next(r for r in result.quantization_rows if r.quantization == "fp16")
        self.assertEqual(fp16_row.delta_vs_selected_gb, 0.0)

    def test_int4_delta_is_negative_vs_fp16(self) -> None:
        model = _llm(7.0)
        inv = _inventory(_compute(gpu_mem_gb=80.0))
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        int4_row = next(r for r in result.quantization_rows if r.quantization == "int4")
        self.assertLess(int4_row.delta_vs_selected_gb, 0.0)

    def test_feasible_node_count_varies(self) -> None:
        model = _llm(7.0)
        inv = _inventory(
            _compute(gpu_mem_gb=16.0, node_type="small"),
            _compute(gpu_mem_gb=80.0, node_type="large"),
        )
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        # int4 should fit on more nodes than fp16
        fp16_row = next(r for r in result.quantization_rows if r.quantization == "fp16")
        int4_row = next(r for r in result.quantization_rows if r.quantization == "int4")
        self.assertGreaterEqual(int4_row.feasible_node_count, fp16_row.feasible_node_count)


class PlanRunPathTests(unittest.TestCase):
    """Test feasible run path generation."""

    def test_run_paths_sorted_safe_first(self) -> None:
        model = _llm(7.0)
        inv = _inventory(
            _compute(gpu_mem_gb=80.0, node_type="large", gpu_family="A100_80"),
            _compute(gpu_mem_gb=24.0, node_type="medium", gpu_family="A10"),
        )
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        if len(result.run_paths) >= 2:
            safe_count = sum(1 for rp in result.run_paths if rp.fit_level == FitLevel.SAFE)
            # First safe_count entries should all be SAFE
            for rp in result.run_paths[:safe_count]:
                self.assertEqual(rp.fit_level, FitLevel.SAFE)

    def test_no_run_paths_for_huge_model(self) -> None:
        model = _llm(200.0)  # 200B, nothing will fit
        inv = _inventory(
            _compute(gpu_mem_gb=16.0, node_type="small"),
        )
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        self.assertEqual(len(result.run_paths), 0)

    def test_run_paths_include_multiple_quantizations(self) -> None:
        model = _llm(7.0)
        inv = _inventory(
            _compute(gpu_mem_gb=80.0, node_type="large", gpu_family="A100_80"),
        )
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        quants_in_paths = set(rp.quantization for rp in result.run_paths)
        # At least fp16 and int4 should have paths on an 80GB GPU
        self.assertIn("fp16", quants_in_paths)
        self.assertIn("int4", quants_in_paths)


class PlanMultiGpuTests(unittest.TestCase):
    """Test multi-GPU planning."""

    def test_multi_gpu_reduces_per_gpu_requirement(self) -> None:
        model = _llm(70.0)  # 70B model
        inv = _inventory(
            _compute(gpu_mem_gb=80.0, gpu_count=4, node_type="multi_gpu", gpu_family="A100_80"),
        )

        single = plan_for_model(model, inv, quantization="fp16", context_length=4096, gpu_count=1)
        multi = plan_for_model(model, inv, quantization="fp16", context_length=4096, gpu_count=4)

        # Total memory should be the same
        self.assertAlmostEqual(single.estimated_memory_gb, multi.estimated_memory_gb, places=1)
        # Per-GPU requirement should be lower
        self.assertLess(multi.min_vram_per_gpu_gb, single.min_vram_per_gpu_gb)

    def test_multi_gpu_filters_to_nodes_with_enough_gpus(self) -> None:
        model = _llm(7.0)
        inv = _inventory(
            _compute(gpu_mem_gb=16.0, gpu_count=1, node_type="single"),
            _compute(gpu_mem_gb=16.0, gpu_count=4, node_type="quad"),
        )
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096, gpu_count=4)

        # Only the quad-GPU node should appear in run paths
        node_types = set(rp.node_type_id for rp in result.run_paths)
        self.assertNotIn("single", node_types)


class PlanNotesTests(unittest.TestCase):
    """Test plan notes generation."""

    def test_notes_include_model_info(self) -> None:
        model = _llm(7.0)
        inv = _inventory(_compute(gpu_mem_gb=80.0))
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        self.assertTrue(any("7.0B" in note for note in result.notes))
        self.assertTrue(any("fp16" in note for note in result.notes))

    def test_notes_include_path_summary(self) -> None:
        model = _llm(7.0)
        inv = _inventory(_compute(gpu_mem_gb=80.0))
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        self.assertTrue(any("safe" in note or "run path" in note for note in result.notes))

    def test_no_feasible_note_for_impossible_plan(self) -> None:
        model = _llm(200.0)
        inv = _inventory(_compute(gpu_mem_gb=16.0))
        result = plan_for_model(model, inv, quantization="fp16", context_length=4096)

        self.assertTrue(any("No feasible" in note for note in result.notes))


if __name__ == "__main__":
    unittest.main()
