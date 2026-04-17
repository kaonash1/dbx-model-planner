"""Inverse planning engine: 'What hardware do I need for this model?'

Given a model and configurable parameters (quantization, context length,
GPU count), compute the minimum VRAM required and find the best matching
node types from the workspace inventory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain import (
    FitLevel,
    ModelFamily,
    ModelProfile,
    WorkspaceComputeProfile,
    WorkspaceInventorySnapshot,
)
from .fit import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_RUNTIME_OVERHEAD_GB,
    QUANTIZATION_BYTES_PER_PARAMETER,
    MemoryEstimate,
    _bytes_per_parameter,
    _estimate_kv_cache_gb,
)

# Context length presets the user can cycle through
CONTEXT_PRESETS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

# Quantization options in quality order (highest to lowest)
QUANTIZATION_OPTIONS = [
    "fp16", "bf16",
    "q8_0", "int8",
    "q6_k", "q5_k_m",
    "q4_k_m", "q4_0", "int4",
    "q3_k_m", "q2_k",
]


@dataclass(slots=True)
class PlanRunPath:
    """A feasible combination of quantization + node type."""

    quantization: str
    node_type_id: str
    gpu_family: str | None
    gpu_count: int
    gpu_memory_gb: float
    estimated_model_memory_gb: float
    headroom_gb: float
    fit_level: FitLevel


@dataclass(slots=True)
class PlanQuantizationRow:
    """Memory requirement for a specific quantization level."""

    quantization: str
    estimated_memory_gb: float
    min_vram_per_gpu_gb: float  # With the configured GPU count
    feasible_node_count: int  # How many workspace nodes can run this
    delta_vs_selected_gb: float  # Difference from selected quantization


@dataclass(slots=True)
class PlanResult:
    """Output of the inverse planning computation."""

    model_id: str
    parameter_count: int | None
    family: ModelFamily

    # User-selected parameters
    selected_quantization: str
    selected_context_length: int
    selected_gpu_count: int

    # Computed requirements
    estimated_memory_gb: float
    min_vram_per_gpu_gb: float  # estimated_memory / gpu_count

    # Best matching node from workspace
    recommended_node: WorkspaceComputeProfile | None
    recommended_fit_level: FitLevel | None

    # All quantization rows (upgrade deltas)
    quantization_rows: list[PlanQuantizationRow] = field(default_factory=list)

    # Feasible run paths (quantization + node combos that work)
    run_paths: list[PlanRunPath] = field(default_factory=list)

    # Explanatory notes
    notes: list[str] = field(default_factory=list)


def _estimate_plan_memory_gb(
    model: ModelProfile,
    quantization: str,
    context_length: int,
) -> MemoryEstimate:
    """Estimate memory for a model with explicit quantization and context length.

    Similar to fit.estimate_model_memory_gb but uses the plan's explicit
    context length instead of the model's default.
    """
    parameter_count = model.active_parameter_count or model.parameter_count or 0
    model_gb = parameter_count * _bytes_per_parameter(quantization) / 1_000_000_000

    if model.family == ModelFamily.LLM:
        kv_cache_gb = _estimate_kv_cache_gb(model, context_length)
    elif model.family == ModelFamily.VLM:
        kv_cache_gb = _estimate_kv_cache_gb(model, context_length) if model.num_hidden_layers else 0.6
    else:
        kv_cache_gb = 0.2

    if model.family in {ModelFamily.EMBEDDING, ModelFamily.RERANKER}:
        runtime_overhead_gb = 0.6
    elif model.family == ModelFamily.VLM:
        runtime_overhead_gb = 2.0
    else:
        runtime_overhead_gb = DEFAULT_RUNTIME_OVERHEAD_GB

    total = model_gb + kv_cache_gb + runtime_overhead_gb
    return MemoryEstimate(
        total_gb=round(total, 2),
        kv_cache_gb=round(kv_cache_gb, 2),
        runtime_overhead_gb=round(runtime_overhead_gb, 2),
    )


def _assess_node_fit(
    estimated_memory_gb: float,
    node: WorkspaceComputeProfile,
) -> FitLevel:
    """Quick fit assessment for a node given a memory estimate."""
    available = node.gpu_memory_gb or 0.0
    if node.gpu_count <= 0 or available <= 0:
        return FitLevel.UNLIKELY
    headroom = available - estimated_memory_gb
    if headroom >= max(available * 0.15, 2.0):
        return FitLevel.SAFE
    elif headroom >= 0:
        return FitLevel.BORDERLINE
    return FitLevel.UNLIKELY


def plan_for_model(
    model: ModelProfile,
    inventory: WorkspaceInventorySnapshot,
    quantization: str = "fp16",
    context_length: int | None = None,
    gpu_count: int = 1,
) -> PlanResult:
    """Compute inverse plan: what hardware is needed for this model?

    Args:
        model: The model to plan for.
        inventory: Workspace inventory with available node types.
        quantization: Target quantization level.
        context_length: Target context length (None = model default or 4096).
        gpu_count: Number of GPUs to assume (for multi-GPU splits).

    Returns:
        PlanResult with requirements, recommendations, and upgrade deltas.
    """
    ctx = context_length or model.context_length or DEFAULT_CONTEXT_LENGTH
    gpu_count = max(gpu_count, 1)

    # -- Compute memory for selected quantization --------------------------
    estimate = _estimate_plan_memory_gb(model, quantization, ctx)
    min_vram_per_gpu = round(estimate.total_gb / gpu_count, 2)

    # -- GPU nodes from inventory ------------------------------------------
    gpu_nodes = [n for n in inventory.compute if n.gpu_count >= gpu_count]

    # -- Find best matching node -------------------------------------------
    recommended_node: WorkspaceComputeProfile | None = None
    recommended_fit: FitLevel | None = None
    best_headroom: float | None = None

    for node in gpu_nodes:
        per_gpu_mem = (node.gpu_memory_gb or 0.0)
        if per_gpu_mem <= 0:
            continue
        # For multi-GPU, assume model parallel across gpu_count GPUs
        effective_mem = per_gpu_mem * min(node.gpu_count, gpu_count)
        headroom = effective_mem - estimate.total_gb
        fit = _assess_node_fit(estimate.total_gb, node) if gpu_count <= 1 else (
            FitLevel.SAFE if headroom >= max(effective_mem * 0.15, 2.0)
            else FitLevel.BORDERLINE if headroom >= 0
            else FitLevel.UNLIKELY
        )

        if fit == FitLevel.UNLIKELY:
            continue

        # Pick the tightest safe fit, then tightest borderline
        if recommended_node is None:
            recommended_node = node
            recommended_fit = fit
            best_headroom = headroom
        else:
            current_rank = 0 if recommended_fit == FitLevel.SAFE else 1
            new_rank = 0 if fit == FitLevel.SAFE else 1
            if new_rank < current_rank:
                recommended_node = node
                recommended_fit = fit
                best_headroom = headroom
            elif new_rank == current_rank and best_headroom is not None and headroom < best_headroom:
                recommended_node = node
                recommended_fit = fit
                best_headroom = headroom

    # -- Build quantization comparison rows --------------------------------
    selected_mem = estimate.total_gb
    quant_rows: list[PlanQuantizationRow] = []

    for q in QUANTIZATION_OPTIONS:
        q_est = _estimate_plan_memory_gb(model, q, ctx)
        q_per_gpu = round(q_est.total_gb / gpu_count, 2)
        feasible_count = sum(
            1 for n in gpu_nodes
            if (n.gpu_memory_gb or 0) * min(n.gpu_count, gpu_count) >= q_est.total_gb
        )
        quant_rows.append(PlanQuantizationRow(
            quantization=q,
            estimated_memory_gb=round(q_est.total_gb, 2),
            min_vram_per_gpu_gb=q_per_gpu,
            feasible_node_count=feasible_count,
            delta_vs_selected_gb=round(q_est.total_gb - selected_mem, 2),
        ))

    # -- Build feasible run paths ------------------------------------------
    run_paths: list[PlanRunPath] = []
    for q in QUANTIZATION_OPTIONS:
        q_est = _estimate_plan_memory_gb(model, q, ctx)
        for node in gpu_nodes:
            per_gpu_mem = node.gpu_memory_gb or 0.0
            if per_gpu_mem <= 0:
                continue
            effective_mem = per_gpu_mem * min(node.gpu_count, gpu_count)
            headroom = effective_mem - q_est.total_gb
            if headroom < 0:
                continue

            fit = (
                FitLevel.SAFE if headroom >= max(effective_mem * 0.15, 2.0)
                else FitLevel.BORDERLINE
            )

            run_paths.append(PlanRunPath(
                quantization=q,
                node_type_id=node.node_type_id,
                gpu_family=node.gpu_family,
                gpu_count=node.gpu_count,
                gpu_memory_gb=per_gpu_mem,
                estimated_model_memory_gb=round(q_est.total_gb, 2),
                headroom_gb=round(headroom, 2),
                fit_level=fit,
            ))

    # Sort: safe first, then by least headroom (tightest fit)
    run_paths.sort(key=lambda rp: (
        0 if rp.fit_level == FitLevel.SAFE else 1,
        rp.headroom_gb,
        rp.node_type_id,
    ))

    # -- Notes -------------------------------------------------------------
    notes: list[str] = []
    param_b = (model.parameter_count or 0) / 1e9
    notes.append(f"Model: {param_b:.1f}B parameters, {quantization} quantization")
    notes.append(f"Context length: {ctx:,} tokens")
    if gpu_count > 1:
        notes.append(f"Assuming model parallelism across {gpu_count} GPUs")
    notes.append(f"Estimated total VRAM: {estimate.total_gb:.1f} GB")
    if gpu_count > 1:
        notes.append(f"Per-GPU requirement: {min_vram_per_gpu:.1f} GB")

    safe_paths = sum(1 for rp in run_paths if rp.fit_level == FitLevel.SAFE)
    total_paths = len(run_paths)
    if total_paths == 0:
        notes.append("No feasible run path exists in this workspace")
    else:
        notes.append(f"{safe_paths} safe, {total_paths - safe_paths} borderline run paths")

    return PlanResult(
        model_id=model.model_id,
        parameter_count=model.parameter_count,
        family=model.family,
        selected_quantization=quantization,
        selected_context_length=ctx,
        selected_gpu_count=gpu_count,
        estimated_memory_gb=estimate.total_gb,
        min_vram_per_gpu_gb=min_vram_per_gpu,
        recommended_node=recommended_node,
        recommended_fit_level=recommended_fit,
        quantization_rows=quant_rows,
        run_paths=run_paths,
        notes=notes,
    )
