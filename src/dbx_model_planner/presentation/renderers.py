from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from dbx_model_planner.domain import ComputeFitReport, DeploymentHint, HostingRecommendation, WorkspaceInventorySnapshot


def render_json(payload: Any) -> str:
    return json.dumps(_jsonable(payload), indent=2, sort_keys=True)


def _jsonable(payload: Any) -> Any:
    if is_dataclass(payload):
        return _jsonable(asdict(payload))
    if isinstance(payload, dict):
        return {str(key): _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_jsonable(value) for value in payload]
    return payload


def render_inventory(snapshot: WorkspaceInventorySnapshot) -> str:
    lines = [
        f"Workspace: {snapshot.workspace_url}",
        f"Compute profiles: {len(snapshot.compute)}",
        f"Runtimes: {len(snapshot.runtimes)}",
        f"Policies: {len(snapshot.policies)}",
    ]
    for compute in snapshot.compute[:10]:
        gpu = f"{compute.gpu_family or 'gpu-unknown'} x{compute.gpu_count}" if compute.gpu_count else "cpu-only"
        lines.append(f"- {compute.node_type_id}: {gpu}, {compute.gpu_memory_gb or 0} GB GPU RAM")
    return "\n".join(lines)


def render_model_recommendation(recommendation: HostingRecommendation) -> str:
    lines = [recommendation.summary]
    for candidate in recommendation.candidates[:5]:
        cost = candidate.cost.vat_adjusted_hourly_rate if candidate.cost else None
        cost_text = f", gross/hr {cost}" if cost is not None else ""
        lines.append(
            f"- {candidate.compute.node_type_id}: {candidate.fit_level.value}, "
            f"{candidate.recommended_quantization}, est mem {candidate.estimated_memory_gb} GB{cost_text}"
        )
    if recommendation.blocking_issues:
        lines.append("Blocking issues:")
        lines.extend(f"- {issue}" for issue in recommendation.blocking_issues)
    return "\n".join(lines)


def render_compute_fit(report: ComputeFitReport) -> str:
    lines = [report.summary]
    if report.model_family_ranges:
        lines.append("Model family ranges:")
        lines.extend(f"- {family}: {value}" for family, value in sorted(report.model_family_ranges.items()))
    lines.append("Example models:")
    for candidate in report.candidates[:8]:
        lines.append(
            f"- {candidate.model.model_id}: {candidate.fit_level.value}, "
            f"{candidate.recommended_quantization}, est mem {candidate.estimated_memory_gb} GB"
        )
    return "\n".join(lines)


def render_deployment_hint(hint: DeploymentHint) -> str:
    lines = [hint.summary]
    if hint.target and hint.target.volume_path:
        lines.append(f"Volume path: {hint.target.volume_path}")
    if hint.recommended_node_type_id:
        lines.append(f"Node type: {hint.recommended_node_type_id}")
    if hint.recommended_runtime_id:
        lines.append(f"Runtime: {hint.recommended_runtime_id}")
    if hint.dependency_notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in hint.dependency_notes)
    if hint.starter_commands:
        lines.append("Starter:")
        lines.extend(hint.starter_commands)
    return "\n".join(lines)
