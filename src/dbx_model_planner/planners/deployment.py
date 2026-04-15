from __future__ import annotations

from dbx_model_planner.config import AppConfig
from dbx_model_planner.domain import DeploymentHint, DeploymentTarget, HostingRecommendation, ModelProfile, WorkspaceInventorySnapshot


def build_deployment_hint(
    config: AppConfig,
    inventory: WorkspaceInventorySnapshot,
    model: ModelProfile,
    recommendation: HostingRecommendation,
) -> DeploymentHint:
    best_candidate = recommendation.candidates[0] if recommendation.candidates else None
    target = DeploymentTarget(
        catalog=config.catalog.catalog,
        schema=config.catalog.schema,
        volume=config.catalog.volume,
        volume_path=(
            f"/Volumes/{config.catalog.catalog}/{config.catalog.schema}/{config.catalog.volume}/{model.model_id.replace('/', '--')}"
            if config.catalog.catalog and config.catalog.schema and config.catalog.volume
            else None
        ),
    )

    dependency_notes = [
        "Download and stage model artifacts only after revision pinning is confirmed.",
        "Validate tokenizer, config, and processor assets before cluster startup.",
    ]
    if model.quantization_options:
        dependency_notes.append(f"Preferred quantization candidates: {', '.join(model.quantization_options[:3])}.")
    if best_candidate:
        dependency_notes.append(f"Recommended node type: {best_candidate.compute.node_type_id}.")

    starter_commands = [
        "# Example next step",
        f"# stage artifacts for {model.model_id}",
    ]
    if target.volume_path:
        starter_commands.append(f"# target volume path: {target.volume_path}")

    runtime_id = best_candidate.compute.runtime_ids[0] if best_candidate and best_candidate.compute.runtime_ids else None
    node_type_id = best_candidate.compute.node_type_id if best_candidate else None
    summary = (
        f"Stage {model.model_id} into a Unity Catalog volume and start from a simple inference script."
        if best_candidate
        else f"Prepare a staging path for {model.model_id}; no compute recommendation is available yet."
    )
    return DeploymentHint(
        summary=summary,
        target=target,
        recommended_node_type_id=node_type_id,
        recommended_runtime_id=runtime_id,
        dependency_notes=dependency_notes,
        starter_commands=starter_commands,
    )
