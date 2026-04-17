from __future__ import annotations

from dataclasses import dataclass, field

from .common import Cloud, EstimateSource, HostingMode, ModelFamily, ModelModality


@dataclass(slots=True)
class ModelArtifactProfile:
    """Artifact and packaging characteristics for a model."""

    source: str
    repository_id: str
    revision: str | None = None
    format: str | None = None
    quantization: str | None = None
    artifact_size_gb: float | None = None
    license_name: str | None = None
    gated: bool = False
    dependency_hints: list[str] = field(default_factory=list)
    processor_required: bool = False


@dataclass(slots=True)
class ModelProfile:
    """Normalized model metadata used by the planner."""

    model_id: str
    family: ModelFamily
    modality: ModelModality
    source: str
    task: str
    parameter_count: int | None = None
    active_parameter_count: int | None = None
    context_length: int | None = None
    max_batch_size_hint: int | None = None
    architecture: str | None = None
    # Architecture details for precise KV cache estimation (from config.json).
    num_hidden_layers: int | None = None
    num_kv_heads: int | None = None  # GQA key-value heads (often < num_attention_heads)
    head_dim: int | None = None      # Per-head dimension in attention
    dtype_options: list[str] = field(default_factory=list)
    quantization_options: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    artifacts: list[ModelArtifactProfile] = field(default_factory=list)
    metadata_sources: list[EstimateSource] = field(default_factory=list)


@dataclass(slots=True)
class WorkloadProfile:
    """Operational intent for evaluating a hosting recommendation."""

    workload_name: str
    online: bool = True
    expected_qps: float | None = None
    target_latency_ms: int | None = None
    target_concurrency: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    input_sequence_length: int | None = None
    scale_to_zero_tolerated: bool = False


@dataclass(slots=True)
class RuntimeProfile:
    """Databricks runtime and engine compatibility facts."""

    runtime_id: str
    dbr_version: str
    ml_runtime: bool = False
    gpu_enabled: bool = False
    photon_supported: bool = False
    cuda_version: str | None = None
    python_version: str | None = None
    supported_engines: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WorkspacePolicyProfile:
    """Policy restrictions that affect whether a recommendation is deployable."""

    policy_id: str
    policy_name: str
    allowed_node_types: list[str] = field(default_factory=list)
    blocked_node_types: list[str] = field(default_factory=list)
    allowed_runtime_ids: list[str] = field(default_factory=list)
    required_tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class WorkspaceComputeProfile:
    """A discovered Databricks compute option."""

    node_type_id: str
    cloud: Cloud = Cloud.AZURE
    region: str | None = None
    vm_sku_name: str | None = None
    gpu_family: str | None = None
    gpu_count: int = 0
    gpu_memory_gb: float | None = None
    vcpu_count: int | None = None
    memory_gb: float | None = None
    local_disk_gb: float | None = None
    dbu_per_hour: float | None = None  # Estimated DBUs consumed per hour
    runtime_ids: list[str] = field(default_factory=list)
    supported_hosting_modes: list[HostingMode] = field(default_factory=list)
    policy_ids: list[str] = field(default_factory=list)
    availability_notes: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    availability_source: EstimateSource = EstimateSource.DISCOVERED


@dataclass(slots=True)
class CostProfile:
    """Cost facts or estimates for a compute or serving option."""

    currency_code: str
    vm_hourly_rate: float | None = None
    dbu_hourly_rate: float | None = None
    estimated_hourly_rate: float | None = None
    discounted_hourly_rate: float | None = None
    vat_adjusted_hourly_rate: float | None = None
    pricing_reference: str | None = None
    source: EstimateSource = EstimateSource.INFERRED


@dataclass(slots=True)
class DeploymentTarget:
    """Storage and governance target for a deployment plan."""

    catalog: str | None = None
    schema: str | None = None
    volume: str | None = None
    volume_path: str | None = None
    unity_catalog_enabled: bool = True


@dataclass(slots=True)
class WorkspaceInventorySnapshot:
    """A locally cached view of Databricks workspace facts."""

    workspace_url: str
    cloud: Cloud = Cloud.AZURE
    region: str | None = None
    compute: list[WorkspaceComputeProfile] = field(default_factory=list)
    runtimes: list[RuntimeProfile] = field(default_factory=list)
    policies: list[WorkspacePolicyProfile] = field(default_factory=list)
