from __future__ import annotations

from dbx_model_planner.config import AppConfig
from dbx_model_planner.domain import (
    CandidateCompute,
    ComputeFitReport,
    FitLevel,
    HostingMode,
    HostingRecommendation,
    ModelProfile,
    WorkloadProfile,
    WorkspaceComputeProfile,
    WorkspaceInventorySnapshot,
    WorkspacePolicyProfile,
)
from dbx_model_planner.engines.cost import compose_cost_profile
from dbx_model_planner.engines.fit import assess_compute_for_models, estimate_tokens_per_second, find_best_quantization, infer_model_family_range, rank_compute_candidates
from dbx_model_planner.engines.score import compute_candidate_score


def _default_hosting_mode(workload: WorkloadProfile) -> HostingMode:
    return HostingMode.CUSTOM_SERVING if workload.online else HostingMode.BATCH_COMPUTE


def _filter_compute_by_preferences(
    config: AppConfig,
    compute_options: list[WorkspaceComputeProfile],
    policies: list[WorkspacePolicyProfile] | None = None,
) -> tuple[list[WorkspaceComputeProfile], list[str]]:
    blocked_node_types = set(config.workspace.blocked_node_types)
    blocked_skus = set(config.workspace.blocked_skus)
    blocked_gpu_families = {value.lower() for value in config.workspace.blocked_gpu_families}

    # Collect policy-level constraints
    policy_blocked: set[str] = set()
    policy_allowed: set[str] | None = None
    for policy in policies or []:
        policy_blocked.update(policy.blocked_node_types)
        if policy.allowed_node_types:
            if policy_allowed is None:
                policy_allowed = set(policy.allowed_node_types)
            else:
                policy_allowed |= set(policy.allowed_node_types)

    filtered: list[WorkspaceComputeProfile] = []
    notes: list[str] = []

    for compute in compute_options:
        sku_or_node_type = compute.vm_sku_name or compute.node_type_id
        gpu_family = (compute.gpu_family or "").lower()
        if compute.node_type_id in blocked_node_types:
            notes.append(f"Skipped {compute.node_type_id} because it is blocked in config.")
            continue
        if sku_or_node_type in blocked_skus:
            notes.append(f"Skipped {compute.node_type_id} because its SKU is blocked in config.")
            continue
        if gpu_family and gpu_family in blocked_gpu_families:
            notes.append(f"Skipped {compute.node_type_id} because its GPU family is blocked in config.")
            continue
        if compute.node_type_id in policy_blocked:
            notes.append(f"Skipped {compute.node_type_id} because it is blocked by workspace policy.")
            continue
        if policy_allowed is not None and compute.node_type_id not in policy_allowed:
            notes.append(f"Skipped {compute.node_type_id} because it is not in the workspace policy allow list.")
            continue
        filtered.append(compute)

    return filtered, notes


def recommend_compute_for_model(
    config: AppConfig,
    inventory: WorkspaceInventorySnapshot,
    model: ModelProfile,
    workload: WorkloadProfile,
    vm_pricing: dict[str, float] | None = None,
    dbu_pricing: dict[str, float] | None = None,
    forced_quantization: str | None = None,
) -> HostingRecommendation:
    vm_pricing = vm_pricing or {}
    dbu_pricing = dbu_pricing or {}
    eligible_compute, filtering_notes = _filter_compute_by_preferences(
        config, inventory.compute, policies=inventory.policies,
    )

    candidates = rank_compute_candidates(
        model, workload, eligible_compute,
        forced_quantization=forced_quantization,
    )
    enriched_candidates: list[CandidateCompute] = []
    for candidate in candidates:
        candidate.best_quantization = find_best_quantization(model, candidate.compute)
        candidate.estimated_tok_s = estimate_tokens_per_second(
            model, candidate.compute, candidate.recommended_quantization or "fp16",
        )
        vm_hourly_rate = vm_pricing.get(candidate.compute.node_type_id)
        dbu_hourly_rate = dbu_pricing.get(candidate.compute.node_type_id)
        if vm_hourly_rate is not None:
            candidate.cost = compose_cost_profile(
                config=config,
                vm_hourly_rate=vm_hourly_rate,
                dbu_hourly_rate=dbu_hourly_rate,
                pricing_reference="azure_vm_plus_configured_dbu",
            )
        enriched_candidates.append(candidate)

    # Compute composite scores and re-sort by score descending
    for candidate in enriched_candidates:
        total_gpu_mem = (candidate.compute.gpu_memory_gb or 0.0) * max(candidate.compute.gpu_count, 1)
        if candidate.cost:
            r = candidate.cost.vat_adjusted_hourly_rate
            cost_per_hour = r if r is not None else candidate.cost.estimated_hourly_rate
        else:
            cost_per_hour = None
        candidate.composite_score = compute_candidate_score(
            fit_level=candidate.fit_level,
            estimated_memory_gb=candidate.estimated_memory_gb,
            total_gpu_memory_gb=total_gpu_mem,
            estimated_tok_s=candidate.estimated_tok_s,
            cost_per_hour=cost_per_hour,
        )
    # Sort: fitting nodes first (SAFE, BORDERLINE), then by smallest GPU memory.
    # This puts the smallest viable node at the top.
    _FIT_ORDER = {FitLevel.SAFE: 0, FitLevel.BORDERLINE: 1, FitLevel.UNLIKELY: 2}
    enriched_candidates.sort(
        key=lambda c: (
            _FIT_ORDER.get(c.fit_level, 9),
            (c.compute.gpu_memory_gb or 0.0) * max(c.compute.gpu_count, 1),
        ),
    )

    if enriched_candidates:
        best = enriched_candidates[0]
        summary = (
            f"Recommended compute for {model.model_id}: {best.compute.node_type_id} "
            f"({best.fit_level.value}, {best.recommended_quantization})."
        )
    else:
        summary = f"No compute candidates are available for {model.model_id}."

    assumptions = [
        "Fit heuristics are conservative estimates, not benchmark results.",
        "Cost is only attached when a VM price is available.",
    ]
    blocking_issues = list(filtering_notes)
    if enriched_candidates and enriched_candidates[0].fit_level == FitLevel.UNLIKELY:
        blocking_issues.append("No discovered compute option is a safe fit for the current model assumptions.")

    return HostingRecommendation(
        hosting_mode=_default_hosting_mode(workload),
        summary=summary,
        candidates=enriched_candidates,
        blocking_issues=blocking_issues,
        assumptions=assumptions,
    )


def recommend_models_for_compute(
    config: AppConfig,
    compute: WorkspaceComputeProfile,
    models: list[ModelProfile],
    vm_hourly_rate: float | None = None,
    dbu_hourly_rate: float | None = None,
) -> ComputeFitReport:
    candidates = assess_compute_for_models(compute, models)
    cost = None
    if vm_hourly_rate is not None:
        cost = compose_cost_profile(
            config=config,
            vm_hourly_rate=vm_hourly_rate,
            dbu_hourly_rate=dbu_hourly_rate,
            pricing_reference="azure_vm_plus_configured_dbu",
        )
        for candidate in candidates:
            candidate.cost = cost

    family_ranges = infer_model_family_range(candidates)
    summary = f"Compute {compute.node_type_id} can safely or borderline run {len([c for c in candidates if c.fit_level != FitLevel.UNLIKELY])} of {len(candidates)} evaluated models."
    blocking_issues = []
    if not family_ranges:
        blocking_issues.append("No evaluated model family appears to be a realistic fit for this compute.")

    return ComputeFitReport(
        compute=compute,
        summary=summary,
        candidates=candidates,
        model_family_ranges=family_ranges,
        blocking_issues=blocking_issues,
        assumptions=[
            "Model ranges are derived from currently evaluated models, not the full Hugging Face universe.",
            "Cost shown per candidate reuses the compute-level price estimate when a VM price is available.",
        ],
    )
