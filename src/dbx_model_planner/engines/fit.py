from __future__ import annotations

from dataclasses import dataclass
from math import floor

from dbx_model_planner.domain import (
    CandidateCompute,
    CandidateModel,
    FitLevel,
    ModelFamily,
    ModelProfile,
    RiskLevel,
    WorkloadProfile,
    WorkspaceComputeProfile,
)

DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_PARAMETER_HEADROOM = 1.20
DEFAULT_RUNTIME_OVERHEAD_GB = 1.5

QUANTIZATION_BYTES_PER_PARAMETER: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "8bit": 1.0,
    "int4": 0.5,
    "4bit": 0.5,
    "awq-4bit": 0.5,
    "gptq-int4": 0.5,
}


@dataclass(slots=True)
class MemoryEstimate:
    total_gb: float
    kv_cache_gb: float
    runtime_overhead_gb: float


def _family_default_quantization(model: ModelProfile) -> str:
    if model.family == ModelFamily.LLM:
        return "fp16"
    if model.family in {ModelFamily.EMBEDDING, ModelFamily.RERANKER}:
        return "fp16"
    if model.family == ModelFamily.VLM:
        return "fp16"
    return "fp16"


def _recommended_quantization_for_budget(
    model: ModelProfile,
    available_gpu_memory_gb: float | None,
) -> str:
    options = [option.lower() for option in model.quantization_options]
    ordered = ["fp16", "bf16", "int8", "8bit", "int4", "4bit", "awq-4bit", "gptq-int4"]
    if available_gpu_memory_gb is None or model.parameter_count is None:
        return options[0] if options else _family_default_quantization(model)

    for quant in ordered:
        if options and quant not in options:
            continue
        estimate = estimate_model_memory_gb(model, quant)
        if estimate.total_gb <= available_gpu_memory_gb:
            return quant
    return options[-1] if options else "int4"


def _bytes_per_parameter(quantization: str | None) -> float:
    if not quantization:
        return QUANTIZATION_BYTES_PER_PARAMETER["fp16"]
    return QUANTIZATION_BYTES_PER_PARAMETER.get(
        quantization.lower(),
        QUANTIZATION_BYTES_PER_PARAMETER["fp16"],
    )


def estimate_model_memory_gb(model: ModelProfile, quantization: str | None = None) -> MemoryEstimate:
    quant = quantization or _family_default_quantization(model)

    if model.parameter_count is None and model.artifacts:
        artifact_size = max(
            (artifact.artifact_size_gb or 0.0) for artifact in model.artifacts
        )
        total = max(artifact_size, DEFAULT_RUNTIME_OVERHEAD_GB)
        return MemoryEstimate(
            total_gb=round(total, 2),
            kv_cache_gb=0.0,
            runtime_overhead_gb=DEFAULT_RUNTIME_OVERHEAD_GB,
        )

    parameter_count = model.active_parameter_count or model.parameter_count or 0
    model_gb = parameter_count * _bytes_per_parameter(quant) / 1_000_000_000

    if model.family == ModelFamily.LLM:
        context = model.context_length or DEFAULT_CONTEXT_LENGTH
        bounded_context = min(context, 8192)
        kv_cache_gb = max((parameter_count / 1_000_000_000) * (bounded_context / 4096) * 0.35, 0.4)
    elif model.family == ModelFamily.VLM:
        kv_cache_gb = 0.6
    else:
        kv_cache_gb = 0.2

    if model.family in {ModelFamily.EMBEDDING, ModelFamily.RERANKER}:
        runtime_overhead_gb = 0.6
    elif model.family == ModelFamily.VLM:
        runtime_overhead_gb = 2.0
    else:
        runtime_overhead_gb = DEFAULT_RUNTIME_OVERHEAD_GB

    total = model_gb * DEFAULT_PARAMETER_HEADROOM + kv_cache_gb + runtime_overhead_gb
    return MemoryEstimate(
        total_gb=round(total, 2),
        kv_cache_gb=round(kv_cache_gb, 2),
        runtime_overhead_gb=round(runtime_overhead_gb, 2),
    )


def assess_model_on_compute(
    model: ModelProfile,
    workload: WorkloadProfile,
    compute: WorkspaceComputeProfile,
) -> CandidateCompute:
    recommended_quantization = _recommended_quantization_for_budget(model, compute.gpu_memory_gb)
    estimate = estimate_model_memory_gb(model, recommended_quantization)

    notes: list[str] = []
    available_gpu_memory_gb = compute.gpu_memory_gb or 0.0
    headroom = round(available_gpu_memory_gb - estimate.total_gb, 2)

    if model.family in {ModelFamily.EMBEDDING, ModelFamily.RERANKER}:
        notes.append("Throughput and batching usually matter more than token generation speed.")
    elif model.family == ModelFamily.VLM:
        notes.append("Processor and multimodal dependencies must be present in addition to weights.")
    else:
        notes.append("Context-length assumptions are conservatively capped for estimation.")

    if compute.gpu_count <= 0:
        fit_level = FitLevel.UNLIKELY
        risk_level = RiskLevel.HIGH
        notes.append("No GPU is present on this compute profile.")
    elif headroom >= max(available_gpu_memory_gb * 0.15, 2.0):
        fit_level = FitLevel.SAFE
        risk_level = RiskLevel.LOW
    elif headroom >= 0:
        fit_level = FitLevel.BORDERLINE
        risk_level = RiskLevel.MEDIUM
        notes.append("Fits, but with limited GPU memory headroom.")
    else:
        fit_level = FitLevel.UNLIKELY
        risk_level = RiskLevel.HIGH
        notes.append("Estimated memory exceeds available GPU memory.")

    if workload.expected_qps and model.family in {ModelFamily.EMBEDDING, ModelFamily.RERANKER}:
        notes.append(f"Requested QPS={workload.expected_qps} should be validated with batching assumptions.")

    return CandidateCompute(
        compute=compute,
        fit_level=fit_level,
        risk_level=risk_level,
        recommended_quantization=recommended_quantization,
        estimated_memory_gb=estimate.total_gb,
        estimated_headroom_gb=headroom,
        notes=notes,
    )


def rank_compute_candidates(
    model: ModelProfile,
    workload: WorkloadProfile,
    compute_options: list[WorkspaceComputeProfile],
) -> list[CandidateCompute]:
    candidates = [
        assess_model_on_compute(model=model, workload=workload, compute=compute)
        for compute in compute_options
    ]
    return sorted(
        candidates,
        key=lambda candidate: (
            0 if candidate.fit_level == FitLevel.SAFE else 1 if candidate.fit_level == FitLevel.BORDERLINE else 2,
            -(candidate.estimated_headroom_gb or -9999.0),
            candidate.compute.node_type_id,
        ),
    )


def assess_compute_for_models(
    compute: WorkspaceComputeProfile,
    models: list[ModelProfile],
) -> list[CandidateModel]:
    workload = WorkloadProfile(workload_name="compute-fit", online=False)
    candidates: list[CandidateModel] = []
    for model in models:
        compute_assessment = assess_model_on_compute(model, workload, compute)
        candidates.append(
            CandidateModel(
                model=model,
                fit_level=compute_assessment.fit_level,
                risk_level=compute_assessment.risk_level,
                recommended_quantization=compute_assessment.recommended_quantization,
                estimated_memory_gb=compute_assessment.estimated_memory_gb,
                notes=compute_assessment.notes,
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            0 if candidate.fit_level == FitLevel.SAFE else 1 if candidate.fit_level == FitLevel.BORDERLINE else 2,
            candidate.model.family.value,
            candidate.model.model_id,
        ),
    )


def infer_model_family_range(models: list[CandidateModel]) -> dict[str, str]:
    ranges: dict[str, str] = {}
    grouped: dict[str, list[ModelProfile]] = {}
    for candidate in models:
        if candidate.fit_level == FitLevel.UNLIKELY or candidate.model.parameter_count is None:
            continue
        grouped.setdefault(candidate.model.family.value, []).append(candidate.model)

    for family, family_models in grouped.items():
        params = sorted(model.parameter_count or 0 for model in family_models)
        if not params:
            continue
        min_b = floor((params[0] / 1_000_000_000) * 10) / 10
        max_b = floor((params[-1] / 1_000_000_000) * 10) / 10
        ranges[family] = f"{min_b}B to {max_b}B"
    return ranges
