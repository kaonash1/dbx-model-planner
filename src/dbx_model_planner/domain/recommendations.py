from __future__ import annotations

from dataclasses import dataclass, field

from .common import FitLevel, HostingMode, RiskLevel
from .profiles import CostProfile, ModelProfile, WorkspaceComputeProfile


@dataclass(slots=True)
class CandidateCompute:
    """A compute option evaluated for a specific model/workload pair."""

    compute: WorkspaceComputeProfile
    fit_level: FitLevel = FitLevel.BORDERLINE
    risk_level: RiskLevel = RiskLevel.MEDIUM
    recommended_quantization: str | None = None
    estimated_memory_gb: float | None = None
    estimated_headroom_gb: float | None = None
    notes: list[str] = field(default_factory=list)
    cost: CostProfile | None = None


@dataclass(slots=True)
class HostingRecommendation:
    """Top-level planner output for a model and workload."""

    hosting_mode: HostingMode
    summary: str
    candidates: list[CandidateCompute] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CandidateModel:
    """A model candidate evaluated for a specific compute option."""

    model: ModelProfile
    fit_level: FitLevel = FitLevel.BORDERLINE
    risk_level: RiskLevel = RiskLevel.MEDIUM
    recommended_quantization: str | None = None
    estimated_memory_gb: float | None = None
    notes: list[str] = field(default_factory=list)
    cost: CostProfile | None = None


@dataclass(slots=True)
class ComputeFitReport:
    """Top-level planner output for a compute-first workflow."""

    compute: WorkspaceComputeProfile
    summary: str
    candidates: list[CandidateModel] = field(default_factory=list)
    model_family_ranges: dict[str, str] = field(default_factory=dict)
    blocking_issues: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
