from __future__ import annotations

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping

from ...domain import (
    Cloud,
    EstimateSource,
    HostingMode,
    RuntimeProfile,
    WorkspaceComputeProfile,
    WorkspaceInventorySnapshot,
    WorkspacePolicyProfile,
)

_DEFAULT_MOCK_FIXTURE = Path(__file__).with_name("fixtures") / "mock_inventory.json"
_DEFAULT_WORKSPACE_URL = "mock://databricks-workspace"
_ALLOWED_MANIFEST_KEYS = {
    "workspace_url",
    "cloud",
    "region",
    "node_types",
    "dbr_versions",
    "policies",
    "pools",
    "notes",
}


class DatabricksInventoryManifestError(ValueError):
    """Raised when a fixture-backed inventory manifest cannot be parsed."""


@dataclass(slots=True)
class DatabricksPoolProfile:
    """Collector-local representation of a Databricks pool.

    The current domain model does not have a dedicated pool object, so the
    collector preserves pools here and records a gap note in the snapshot
    wrapper.
    """

    pool_id: str
    pool_name: str
    node_type_id: str | None = None
    runtime_id: str | None = None
    policy_id: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DatabricksInventoryCollection:
    """Normalized inventory facts plus collector-local gap coverage."""

    snapshot: WorkspaceInventorySnapshot
    pools: list[DatabricksPoolProfile] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source_path: Path | None = None


def load_inventory_manifest(fixture_path: str | Path | None = None) -> dict[str, Any]:
    """Load a JSON inventory manifest from a fixture path."""

    path = Path(fixture_path) if fixture_path is not None else _DEFAULT_MOCK_FIXTURE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise DatabricksInventoryManifestError(f"Inventory fixture not found: {path}") from exc
    except JSONDecodeError as exc:
        raise DatabricksInventoryManifestError(f"Inventory fixture is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise DatabricksInventoryManifestError("Inventory fixture must decode to a JSON object.")
    return payload


def parse_inventory_manifest(manifest: Mapping[str, Any], *, source_path: Path | None = None) -> DatabricksInventoryCollection:
    """Convert a manifest into domain objects and collector-local pool records."""

    if not isinstance(manifest, Mapping):
        raise DatabricksInventoryManifestError("Inventory manifest must be a mapping.")

    unknown_keys = _validate_manifest_keys(manifest)
    workspace_url = _string_value(manifest, "workspace_url", default=_DEFAULT_WORKSPACE_URL)
    cloud = _cloud_value(manifest.get("cloud", Cloud.AZURE))
    region = _optional_string_value(manifest.get("region"))

    compute = [_parse_node_type(entry) for entry in _sequence_value(manifest.get("node_types", []), "node_types")]
    runtimes = [_parse_runtime(entry) for entry in _sequence_value(manifest.get("dbr_versions", []), "dbr_versions")]
    policies = [_parse_policy(entry) for entry in _sequence_value(manifest.get("policies", []), "policies")]
    pools = [_parse_pool(entry) for entry in _sequence_value(manifest.get("pools", []), "pools")]

    notes = [str(note) for note in _sequence_value(manifest.get("notes", []), "notes")]
    if unknown_keys:
        notes.append(f"Ignored unsupported inventory manifest keys: {', '.join(unknown_keys)}")
    if pools:
        notes.append(
            "Pools are preserved in DatabricksInventoryCollection because WorkspaceInventorySnapshot "
            "does not yet have a pool field."
        )

    snapshot = WorkspaceInventorySnapshot(
        workspace_url=workspace_url,
        cloud=cloud,
        region=region,
        compute=compute,
        runtimes=runtimes,
        policies=policies,
    )
    return DatabricksInventoryCollection(snapshot=snapshot, pools=pools, notes=notes, source_path=source_path)


class DatabricksInventoryCollector:
    """Offline-first Databricks inventory collector.

    This skeleton only supports mock/fixture mode. It is intentionally read-only
    and leaves live Databricks API access for a later ticket.
    """

    def __init__(
        self,
        *,
        fixture_path: str | Path | None = None,
        mode: str = "mock",
    ) -> None:
        if mode != "mock":
            raise NotImplementedError("Only mock mode is implemented for the Databricks inventory skeleton.")
        self._fixture_path = Path(fixture_path) if fixture_path is not None else None
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    def collect(self) -> DatabricksInventoryCollection:
        manifest = load_inventory_manifest(self._fixture_path)
        collection = parse_inventory_manifest(manifest, source_path=self._fixture_path or _DEFAULT_MOCK_FIXTURE)
        return collection

    def collect_snapshot(self) -> WorkspaceInventorySnapshot:
        """Return the normalized domain snapshot for callers that only need the core model."""

        return self.collect().snapshot


def _validate_manifest_keys(manifest: Mapping[str, Any]) -> list[str]:
    unknown_keys = sorted(set(manifest.keys()) - _ALLOWED_MANIFEST_KEYS)
    return unknown_keys


def _parse_node_type(entry: Any) -> WorkspaceComputeProfile:
    mapping = _dict_entry(entry, "node_types")
    node_type_id = _string_value(mapping, "node_type_id")
    return WorkspaceComputeProfile(
        node_type_id=node_type_id,
        cloud=_cloud_value(mapping.get("cloud", Cloud.AZURE)),
        region=_optional_string_value(mapping.get("region")),
        gpu_family=_optional_string_value(mapping.get("gpu_family")),
        gpu_count=_int_value(mapping.get("gpu_count", 0), "gpu_count"),
        gpu_memory_gb=_optional_float_value(mapping.get("gpu_memory_gb")),
        vcpu_count=_optional_int_value(mapping.get("vcpu_count"), "vcpu_count"),
        memory_gb=_optional_float_value(mapping.get("memory_gb")),
        local_disk_gb=_optional_float_value(mapping.get("local_disk_gb")),
        runtime_ids=[str(value) for value in _sequence_value(mapping.get("runtime_ids", []), "runtime_ids")],
        supported_hosting_modes=[_hosting_mode_value(value) for value in _sequence_value(mapping.get("supported_hosting_modes", []), "supported_hosting_modes")],
        policy_ids=[str(value) for value in _sequence_value(mapping.get("policy_ids", []), "policy_ids")],
        availability_source=EstimateSource.DISCOVERED,
    )


def _parse_runtime(entry: Any) -> RuntimeProfile:
    mapping = _dict_entry(entry, "dbr_versions")
    runtime_id = _string_value(mapping, "runtime_id")
    dbr_version = _string_value(mapping, "dbr_version")
    return RuntimeProfile(
        runtime_id=runtime_id,
        dbr_version=dbr_version,
        ml_runtime=_bool_value(mapping.get("ml_runtime", False), "ml_runtime"),
        gpu_enabled=_bool_value(mapping.get("gpu_enabled", False), "gpu_enabled"),
        photon_supported=_bool_value(mapping.get("photon_supported", False), "photon_supported"),
        cuda_version=_optional_string_value(mapping.get("cuda_version")),
        python_version=_optional_string_value(mapping.get("python_version")),
        supported_engines=[str(value) for value in _sequence_value(mapping.get("supported_engines", []), "supported_engines")],
        notes=[str(value) for value in _sequence_value(mapping.get("notes", []), "notes")],
    )


def _parse_policy(entry: Any) -> WorkspacePolicyProfile:
    mapping = _dict_entry(entry, "policies")
    policy_id = _string_value(mapping, "policy_id")
    policy_name = _string_value(mapping, "policy_name")
    return WorkspacePolicyProfile(
        policy_id=policy_id,
        policy_name=policy_name,
        allowed_node_types=[str(value) for value in _sequence_value(mapping.get("allowed_node_types", []), "allowed_node_types")],
        blocked_node_types=[str(value) for value in _sequence_value(mapping.get("blocked_node_types", []), "blocked_node_types")],
        allowed_runtime_ids=[str(value) for value in _sequence_value(mapping.get("allowed_runtime_ids", []), "allowed_runtime_ids")],
        required_tags=_dict_or_empty(mapping.get("required_tags"), "required_tags"),
    )


def _parse_pool(entry: Any) -> DatabricksPoolProfile:
    mapping = _dict_entry(entry, "pools")
    pool_id = _string_value(mapping, "pool_id")
    pool_name = _string_value(mapping, "pool_name")
    return DatabricksPoolProfile(
        pool_id=pool_id,
        pool_name=pool_name,
        node_type_id=_optional_string_value(mapping.get("node_type_id")),
        runtime_id=_optional_string_value(mapping.get("runtime_id")),
        policy_id=_optional_string_value(mapping.get("policy_id")),
        notes=[str(value) for value in _sequence_value(mapping.get("notes", []), "notes")],
    )


def _dict_entry(entry: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        raise DatabricksInventoryManifestError(f"Entries in '{section}' must be JSON objects.")
    return entry


def _sequence_value(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    raise DatabricksInventoryManifestError(f"Field '{field_name}' must be a JSON array when present.")


def _dict_or_empty(value: Any, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DatabricksInventoryManifestError(f"Field '{field_name}' must be a JSON object when present.")
    return {str(key): str(item) for key, item in value.items()}


def _string_value(mapping_or_value: Mapping[str, Any] | Any, field_name: str, *, default: str | None = None) -> str:
    if isinstance(mapping_or_value, Mapping):
        value = mapping_or_value.get(field_name, default)
    else:
        value = mapping_or_value if mapping_or_value is not None else default
    if value is None:
        raise DatabricksInventoryManifestError(f"Missing required field '{field_name}'.")
    if not isinstance(value, str):
        raise DatabricksInventoryManifestError(f"Field '{field_name}' must be a string.")
    return value


def _optional_string_value(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatabricksInventoryManifestError("Expected a string or null value.")
    return value


def _bool_value(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise DatabricksInventoryManifestError(f"Field '{field_name}' must be a boolean.")


def _int_value(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatabricksInventoryManifestError(f"Field '{field_name}' must be an integer.")
    return value


def _optional_int_value(value: Any, field_name: str = "integer_field") -> int | None:
    if value is None:
        return None
    return _int_value(value, field_name)


def _optional_float_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatabricksInventoryManifestError("Expected a numeric value or null.")
    return float(value)


def _cloud_value(value: Any) -> Cloud:
    if isinstance(value, Cloud):
        return value
    if isinstance(value, str):
        try:
            return Cloud(value)
        except ValueError as exc:
            raise DatabricksInventoryManifestError(f"Unsupported cloud value: {value}") from exc
    raise DatabricksInventoryManifestError("Cloud value must be a string or Cloud enum.")


def _hosting_mode_value(value: Any) -> HostingMode:
    if isinstance(value, HostingMode):
        return value
    if isinstance(value, str):
        try:
            return HostingMode(value)
        except ValueError as exc:
            raise DatabricksInventoryManifestError(f"Unsupported hosting mode value: {value}") from exc
    raise DatabricksInventoryManifestError("Hosting mode values must be strings or HostingMode enums.")
