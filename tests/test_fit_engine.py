from __future__ import annotations

import unittest

from dbx_model_planner.domain.common import FitLevel, ModelFamily, ModelModality, RiskLevel
from dbx_model_planner.domain.profiles import ModelArtifactProfile, ModelProfile, WorkloadProfile, WorkspaceComputeProfile
from dbx_model_planner.engines.fit import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_PARAMETER_HEADROOM,
    DEFAULT_RUNTIME_OVERHEAD_GB,
    MemoryEstimate,
    assess_compute_for_models,
    assess_model_on_compute,
    estimate_model_memory_gb,
    infer_model_family_range,
    rank_compute_candidates,
)


def _llm(params_b: float = 7.0, context: int | None = 32768, quant_opts: list[str] | None = None) -> ModelProfile:
    return ModelProfile(
        model_id=f"test/llm-{params_b}b",
        family=ModelFamily.LLM,
        modality=ModelModality.TEXT,
        source="test",
        task="text-generation",
        parameter_count=int(params_b * 1_000_000_000),
        context_length=context,
        quantization_options=quant_opts or ["fp16", "int8", "int4"],
    )


def _embedding(params_b: float = 0.335) -> ModelProfile:
    return ModelProfile(
        model_id=f"test/embedding-{params_b}b",
        family=ModelFamily.EMBEDDING,
        modality=ModelModality.TEXT_EMBEDDING,
        source="test",
        task="feature-extraction",
        parameter_count=int(params_b * 1_000_000_000),
        context_length=512,
        quantization_options=["fp16"],
    )


def _vlm(params_b: float = 2.0) -> ModelProfile:
    return ModelProfile(
        model_id=f"test/vlm-{params_b}b",
        family=ModelFamily.VLM,
        modality=ModelModality.IMAGE_TEXT,
        source="test",
        task="image-text-to-text",
        parameter_count=int(params_b * 1_000_000_000),
        context_length=32768,
        quantization_options=["bf16", "int4"],
    )


def _compute(gpu_mem_gb: float = 16.0, gpu_count: int = 1, node_type: str = "Standard_NC6s_v3") -> WorkspaceComputeProfile:
    return WorkspaceComputeProfile(
        node_type_id=node_type,
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_mem_gb,
        gpu_family="V100" if gpu_count > 0 else None,
    )


def _workload(online: bool = False) -> WorkloadProfile:
    return WorkloadProfile(workload_name="test", online=online)


class EstimateModelMemoryTests(unittest.TestCase):
    """Test estimate_model_memory_gb across families and quantizations."""

    def test_llm_fp16_estimate(self) -> None:
        model = _llm(params_b=7.0)
        est = estimate_model_memory_gb(model, "fp16")

        # 7B * 2 bytes / 1e9 = 14 GB model weight
        # KV cache + runtime overhead on top
        self.assertGreater(est.total_gb, 14.0)
        self.assertGreater(est.kv_cache_gb, 0.0)
        self.assertEqual(est.runtime_overhead_gb, DEFAULT_RUNTIME_OVERHEAD_GB)

    def test_llm_int4_uses_less_memory_than_fp16(self) -> None:
        model = _llm(params_b=7.0)
        fp16 = estimate_model_memory_gb(model, "fp16")
        int4 = estimate_model_memory_gb(model, "int4")

        self.assertLess(int4.total_gb, fp16.total_gb)

    def test_llm_int8_between_fp16_and_int4(self) -> None:
        model = _llm(params_b=7.0)
        fp16 = estimate_model_memory_gb(model, "fp16")
        int8 = estimate_model_memory_gb(model, "int8")
        int4 = estimate_model_memory_gb(model, "int4")

        self.assertLess(int8.total_gb, fp16.total_gb)
        self.assertGreater(int8.total_gb, int4.total_gb)

    def test_embedding_has_lower_overhead_than_llm(self) -> None:
        emb = _embedding(params_b=0.335)
        est = estimate_model_memory_gb(emb, "fp16")

        self.assertEqual(est.runtime_overhead_gb, 0.6)
        self.assertLess(est.kv_cache_gb, 0.5)

    def test_vlm_includes_extra_overhead(self) -> None:
        vlm = _vlm(params_b=2.0)
        est = estimate_model_memory_gb(vlm, "bf16")

        self.assertEqual(est.runtime_overhead_gb, 2.0)
        self.assertEqual(est.kv_cache_gb, 0.6)

    def test_zero_params_returns_minimal_estimate(self) -> None:
        model = ModelProfile(
            model_id="test/zero",
            family=ModelFamily.OTHER,
            modality=ModelModality.TEXT,
            source="test",
            task="unknown",
            parameter_count=0,
        )
        est = estimate_model_memory_gb(model)

        # With 0 params the model weight is 0, but overhead remains
        self.assertGreater(est.total_gb, 0.0)

    def test_no_params_with_artifact_uses_artifact_size(self) -> None:
        model = ModelProfile(
            model_id="test/artifact-only",
            family=ModelFamily.LLM,
            modality=ModelModality.TEXT,
            source="test",
            task="text-generation",
            parameter_count=None,
            artifacts=[
                ModelArtifactProfile(
                    source="test",
                    repository_id="test/artifact-only",
                    artifact_size_gb=5.0,
                )
            ],
        )
        est = estimate_model_memory_gb(model)

        self.assertGreaterEqual(est.total_gb, 5.0)
        self.assertEqual(est.kv_cache_gb, 0.0)

    def test_default_context_length_used_when_none(self) -> None:
        model = _llm(params_b=7.0, context=None)
        est = estimate_model_memory_gb(model, "fp16")

        # Should still produce a valid estimate using DEFAULT_CONTEXT_LENGTH
        self.assertGreater(est.total_gb, 0.0)

    def test_active_parameter_count_preferred_over_total(self) -> None:
        model = ModelProfile(
            model_id="test/moe",
            family=ModelFamily.LLM,
            modality=ModelModality.TEXT,
            source="test",
            task="text-generation",
            parameter_count=46_000_000_000,
            active_parameter_count=12_000_000_000,
            context_length=4096,
            quantization_options=["fp16"],
        )

        est = estimate_model_memory_gb(model, "fp16")
        # Should use 12B active, not 46B total → ~24GB model weight, not ~92GB
        self.assertLess(est.total_gb, 40.0)

    def test_large_context_is_bounded(self) -> None:
        model_large_ctx = _llm(params_b=7.0, context=131072)
        model_small_ctx = _llm(params_b=7.0, context=4096)

        est_large = estimate_model_memory_gb(model_large_ctx, "fp16")
        est_small = estimate_model_memory_gb(model_small_ctx, "fp16")

        # Context is bounded to 8192 internally, so large context shouldn't
        # cause unbounded KV cache inflation relative to the 4096 baseline
        ratio = est_large.kv_cache_gb / est_small.kv_cache_gb
        self.assertLess(ratio, 3.0)


class AssessModelOnComputeTests(unittest.TestCase):
    """Test assess_model_on_compute for fit level classification."""

    def test_safe_fit_with_large_gpu(self) -> None:
        model = _embedding(params_b=0.335)
        compute = _compute(gpu_mem_gb=16.0)

        result = assess_model_on_compute(model, _workload(), compute)

        self.assertEqual(result.fit_level, FitLevel.SAFE)
        self.assertEqual(result.risk_level, RiskLevel.LOW)
        self.assertGreater(result.estimated_headroom_gb, 0.0)

    def test_no_gpu_is_unlikely(self) -> None:
        model = _llm(params_b=7.0)
        compute = _compute(gpu_mem_gb=0.0, gpu_count=0, node_type="Standard_D3_v2")

        result = assess_model_on_compute(model, _workload(), compute)

        self.assertEqual(result.fit_level, FitLevel.UNLIKELY)
        self.assertEqual(result.risk_level, RiskLevel.HIGH)
        self.assertTrue(any("No GPU" in note for note in result.notes))

    def test_borderline_when_tight_memory(self) -> None:
        model = _llm(params_b=7.0)
        # Find an estimate to figure out a tight GPU size
        est = estimate_model_memory_gb(model, "int4")
        # GPU just barely enough
        compute = _compute(gpu_mem_gb=est.total_gb + 0.5)

        result = assess_model_on_compute(model, _workload(), compute)

        self.assertIn(result.fit_level, {FitLevel.BORDERLINE, FitLevel.SAFE})

    def test_unlikely_when_memory_exceeded(self) -> None:
        model = _llm(params_b=70.0)
        compute = _compute(gpu_mem_gb=16.0)

        result = assess_model_on_compute(model, _workload(), compute)

        self.assertEqual(result.fit_level, FitLevel.UNLIKELY)
        self.assertLess(result.estimated_headroom_gb, 0.0)

    def test_embedding_note_about_throughput(self) -> None:
        model = _embedding()
        compute = _compute(gpu_mem_gb=16.0)

        result = assess_model_on_compute(model, _workload(), compute)

        self.assertTrue(any("Throughput" in note or "batching" in note for note in result.notes))

    def test_vlm_note_about_processor(self) -> None:
        model = _vlm()
        compute = _compute(gpu_mem_gb=16.0)

        result = assess_model_on_compute(model, _workload(), compute)

        self.assertTrue(any("Processor" in note or "multimodal" in note for note in result.notes))

    def test_qps_note_on_embedding_with_expected_qps(self) -> None:
        model = _embedding()
        compute = _compute(gpu_mem_gb=16.0)
        workload = WorkloadProfile(workload_name="test", online=False, expected_qps=100.0)

        result = assess_model_on_compute(model, workload, compute)

        self.assertTrue(any("QPS" in note for note in result.notes))


class RankComputeCandidatesTests(unittest.TestCase):
    """Test rank_compute_candidates ordering."""

    def test_safe_ranked_before_borderline_before_unlikely(self) -> None:
        model = _llm(params_b=7.0)
        computes = [
            _compute(gpu_mem_gb=0.0, gpu_count=0, node_type="cpu-only"),
            _compute(gpu_mem_gb=80.0, node_type="big-gpu"),
            _compute(gpu_mem_gb=16.0, node_type="medium-gpu"),
        ]

        ranked = rank_compute_candidates(model, _workload(), computes)

        self.assertEqual(len(ranked), 3)
        # big-gpu should be first (safe), cpu-only should be last (unlikely)
        self.assertEqual(ranked[0].compute.node_type_id, "big-gpu")
        self.assertEqual(ranked[-1].compute.node_type_id, "cpu-only")

    def test_empty_computes_returns_empty(self) -> None:
        ranked = rank_compute_candidates(_llm(), _workload(), [])
        self.assertEqual(ranked, [])


class AssessComputeForModelsTests(unittest.TestCase):
    """Test assess_compute_for_models."""

    def test_mixed_models_sorted_by_fit(self) -> None:
        compute = _compute(gpu_mem_gb=16.0)
        models = [_llm(params_b=70.0), _embedding(params_b=0.335), _vlm(params_b=2.0)]

        results = assess_compute_for_models(compute, models)

        self.assertEqual(len(results), 3)
        # Embedding and 2B VLM should fit, 70B LLM should not
        fit_levels = [r.fit_level for r in results]
        self.assertIn(FitLevel.SAFE, fit_levels)
        self.assertIn(FitLevel.UNLIKELY, fit_levels)


class InferModelFamilyRangeTests(unittest.TestCase):
    """Test infer_model_family_range."""

    def test_ranges_computed_from_safe_candidates(self) -> None:
        compute = _compute(gpu_mem_gb=80.0)
        models = [_llm(params_b=7.0), _llm(params_b=13.0), _embedding(params_b=0.335)]

        candidates = assess_compute_for_models(compute, models)
        ranges = infer_model_family_range(candidates)

        self.assertIn("llm", ranges)
        self.assertIn("embedding", ranges)
        self.assertIn("7.0B", ranges["llm"])

    def test_unlikely_models_excluded_from_ranges(self) -> None:
        compute = _compute(gpu_mem_gb=4.0)
        models = [_llm(params_b=70.0)]

        candidates = assess_compute_for_models(compute, models)
        ranges = infer_model_family_range(candidates)

        self.assertEqual(ranges, {})


if __name__ == "__main__":
    unittest.main()
