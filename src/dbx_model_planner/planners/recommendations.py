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
from dbx_model_planner.engines.fit import assess_compute_for_models, infer_model_family_range, rank_compute_candidates


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
) -> HostingRecommendation:
    vm_pricing = vm_pricing or {}
    dbu_pricing = dbu_pricing or {}
    eligible_compute, filtering_notes = _filter_compute_by_preferences(
        config, inventory.compute, policies=inventory.policies,
    )

    candidates = rank_compute_candidates(model, workload, eligible_compute)
    enriched_candidates: list[CandidateCompute] = []
    for candidate in candidates:
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
