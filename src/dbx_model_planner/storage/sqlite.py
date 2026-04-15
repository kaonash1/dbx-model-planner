from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from ..domain.common import Cloud, EstimateSource, HostingMode, ModelFamily, ModelModality
from ..domain.profiles import (
    ModelArtifactProfile,
    ModelProfile,
    RuntimeProfile,
    WorkspaceComputeProfile,
    WorkspaceInventorySnapshot,
    WorkspacePolicyProfile,
)


@dataclass(slots=True, frozen=True)
class SnapshotRecord:
    """Metadata for a stored snapshot row."""

    snapshot_id: str
    kind: str
    subject_id: str
    created_at: datetime


class SQLiteSnapshotStore:
    """SQLite-backed persistence for inventory and model snapshots."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path).expanduser()
        self.initialize()

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> None:
        """Create the schema if it does not already exist."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_snapshots_kind_subject_created
                ON snapshots(kind, subject_id, created_at DESC)
                """
            )

    def save_inventory_snapshot(
        self,
        snapshot: WorkspaceInventorySnapshot,
        *,
        snapshot_id: str | None = None,
        created_at: datetime | None = None,
    ) -> SnapshotRecord:
        """Persist an inventory snapshot and return the stored record metadata."""

        record = self._save(
            kind="inventory",
            subject_id=snapshot.workspace_url,
            payload=_inventory_snapshot_to_payload(snapshot),
            snapshot_id=snapshot_id,
            created_at=created_at,
        )
        return record

    def load_inventory_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        workspace_url: str | None = None,
    ) -> WorkspaceInventorySnapshot | None:
        """Load an inventory snapshot by id or latest workspace snapshot."""

        row = self._fetch_snapshot(kind="inventory", subject_id=workspace_url, snapshot_id=snapshot_id)
        if row is None:
            return None
        return _inventory_snapshot_from_payload(json.loads(row["payload"]))

    def save_model_snapshot(
        self,
        model: ModelProfile,
        *,
        snapshot_id: str | None = None,
        created_at: datetime | None = None,
    ) -> SnapshotRecord:
        """Persist a model metadata snapshot and return the stored record metadata."""

        return self._save(
            kind="model",
            subject_id=model.model_id,
            payload=_model_snapshot_to_payload(model),
            snapshot_id=snapshot_id,
            created_at=created_at,
        )

    def load_model_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        model_id: str | None = None,
    ) -> ModelProfile | None:
        """Load a model snapshot by id or latest model snapshot."""

        row = self._fetch_snapshot(kind="model", subject_id=model_id, snapshot_id=snapshot_id)
        if row is None:
            return None
        return _model_snapshot_from_payload(json.loads(row["payload"]))

    def _save(
        self,
        *,
        kind: str,
        subject_id: str,
        payload: Mapping[str, Any],
        snapshot_id: str | None = None,
        created_at: datetime | None = None,
    ) -> SnapshotRecord:
        record_id = snapshot_id or uuid4().hex
        created = created_at or datetime.now(timezone.utc)
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO snapshots (snapshot_id, kind, subject_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    kind = excluded.kind,
                    subject_id = excluded.subject_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (record_id, kind, subject_id, created.isoformat(), serialized),
            )

        return SnapshotRecord(snapshot_id=record_id, kind=kind, subject_id=subject_id, created_at=created)

    def _fetch_snapshot(
        self,
        *,
        kind: str,
        subject_id: str | None,
        snapshot_id: str | None,
    ) -> sqlite3.Row | None:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            if snapshot_id is not None:
                row = connection.execute(
                    """
                    SELECT snapshot_id, kind, subject_id, created_at, payload
                    FROM snapshots
                    WHERE snapshot_id = ? AND kind = ?
                    """,
                    (snapshot_id, kind),
                ).fetchone()
                return row

            if subject_id is not None:
                row = connection.execute(
                    """
                    SELECT snapshot_id, kind, subject_id, created_at, payload
                    FROM snapshots
                    WHERE kind = ? AND subject_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (kind, subject_id),
                ).fetchone()
                return row

            row = connection.execute(
                """
                SELECT snapshot_id, kind, subject_id, created_at, payload
                FROM snapshots
                WHERE kind = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (kind,),
            ).fetchone()
            return row

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection


def _inventory_snapshot_to_payload(snapshot: WorkspaceInventorySnapshot) -> dict[str, Any]:
    return {
        "workspace_url": snapshot.workspace_url,
        "cloud": snapshot.cloud.value,
        "region": snapshot.region,
        "compute": [_workspace_compute_profile_to_payload(item) for item in snapshot.compute],
        "runtimes": [_runtime_profile_to_payload(item) for item in snapshot.runtimes],
        "policies": [_workspace_policy_profile_to_payload(item) for item in snapshot.policies],
    }


def _inventory_snapshot_from_payload(payload: Mapping[str, Any]) -> WorkspaceInventorySnapshot:
    return WorkspaceInventorySnapshot(
        workspace_url=str(payload["workspace_url"]),
        cloud=Cloud(str(payload.get("cloud", Cloud.AZURE))),
        region=_optional_str(payload.get("region")),
        compute=[_workspace_compute_profile_from_payload(item) for item in payload.get("compute", [])],
        runtimes=[_runtime_profile_from_payload(item) for item in payload.get("runtimes", [])],
        policies=[_workspace_policy_profile_from_payload(item) for item in payload.get("policies", [])],
    )


def _workspace_compute_profile_to_payload(profile: WorkspaceComputeProfile) -> dict[str, Any]:
    return {
        "node_type_id": profile.node_type_id,
        "cloud": profile.cloud.value,
        "region": profile.region,
        "vm_sku_name": profile.vm_sku_name,
        "gpu_family": profile.gpu_family,
        "gpu_count": profile.gpu_count,
        "gpu_memory_gb": profile.gpu_memory_gb,
        "vcpu_count": profile.vcpu_count,
        "memory_gb": profile.memory_gb,
        "local_disk_gb": profile.local_disk_gb,
        "runtime_ids": list(profile.runtime_ids),
        "supported_hosting_modes": [mode.value for mode in profile.supported_hosting_modes],
        "policy_ids": list(profile.policy_ids),
        "availability_notes": list(profile.availability_notes),
        "attributes": dict(profile.attributes),
        "availability_source": profile.availability_source.value,
    }


def _workspace_compute_profile_from_payload(payload: Mapping[str, Any]) -> WorkspaceComputeProfile:
    return WorkspaceComputeProfile(
        node_type_id=str(payload["node_type_id"]),
        cloud=Cloud(str(payload.get("cloud", Cloud.AZURE))),
        region=_optional_str(payload.get("region")),
        vm_sku_name=_optional_str(payload.get("vm_sku_name")),
        gpu_family=_optional_str(payload.get("gpu_family")),
        gpu_count=int(payload.get("gpu_count", 0)),
        gpu_memory_gb=_optional_float(payload.get("gpu_memory_gb")),
        vcpu_count=_optional_int(payload.get("vcpu_count")),
        memory_gb=_optional_float(payload.get("memory_gb")),
        local_disk_gb=_optional_float(payload.get("local_disk_gb")),
        runtime_ids=[str(item) for item in payload.get("runtime_ids", [])],
        supported_hosting_modes=[
            HostingMode(str(item)) for item in payload.get("supported_hosting_modes", [])
        ],
        policy_ids=[str(item) for item in payload.get("policy_ids", [])],
        availability_notes=[str(item) for item in payload.get("availability_notes", [])],
        attributes={str(key): str(value) for key, value in dict(payload.get("attributes", {})).items()},
        availability_source=EstimateSource(str(payload.get("availability_source", EstimateSource.DISCOVERED))),
    )


def _runtime_profile_to_payload(profile: RuntimeProfile) -> dict[str, Any]:
    return {
        "runtime_id": profile.runtime_id,
        "dbr_version": profile.dbr_version,
        "ml_runtime": profile.ml_runtime,
        "gpu_enabled": profile.gpu_enabled,
        "photon_supported": profile.photon_supported,
        "cuda_version": profile.cuda_version,
        "python_version": profile.python_version,
        "supported_engines": list(profile.supported_engines),
        "notes": list(profile.notes),
    }


def _runtime_profile_from_payload(payload: Mapping[str, Any]) -> RuntimeProfile:
    return RuntimeProfile(
        runtime_id=str(payload["runtime_id"]),
        dbr_version=str(payload["dbr_version"]),
        ml_runtime=bool(payload.get("ml_runtime", False)),
        gpu_enabled=bool(payload.get("gpu_enabled", False)),
        photon_supported=bool(payload.get("photon_supported", False)),
        cuda_version=_optional_str(payload.get("cuda_version")),
        python_version=_optional_str(payload.get("python_version")),
        supported_engines=[str(item) for item in payload.get("supported_engines", [])],
        notes=[str(item) for item in payload.get("notes", [])],
    )


def _workspace_policy_profile_to_payload(profile: WorkspacePolicyProfile) -> dict[str, Any]:
    return {
        "policy_id": profile.policy_id,
        "policy_name": profile.policy_name,
        "allowed_node_types": list(profile.allowed_node_types),
        "blocked_node_types": list(profile.blocked_node_types),
        "allowed_runtime_ids": list(profile.allowed_runtime_ids),
        "required_tags": dict(profile.required_tags),
    }


def _workspace_policy_profile_from_payload(payload: Mapping[str, Any]) -> WorkspacePolicyProfile:
    required_tags = payload.get("required_tags", {})
    return WorkspacePolicyProfile(
        policy_id=str(payload["policy_id"]),
        policy_name=str(payload["policy_name"]),
        allowed_node_types=[str(item) for item in payload.get("allowed_node_types", [])],
        blocked_node_types=[str(item) for item in payload.get("blocked_node_types", [])],
        allowed_runtime_ids=[str(item) for item in payload.get("allowed_runtime_ids", [])],
        required_tags={str(key): str(value) for key, value in dict(required_tags).items()},
    )


def _model_snapshot_to_payload(model: ModelProfile) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "family": model.family.value,
        "modality": model.modality.value,
        "source": model.source,
        "task": model.task,
        "parameter_count": model.parameter_count,
        "active_parameter_count": model.active_parameter_count,
        "context_length": model.context_length,
        "max_batch_size_hint": model.max_batch_size_hint,
        "architecture": model.architecture,
        "dtype_options": list(model.dtype_options),
        "quantization_options": list(model.quantization_options),
        "capabilities": list(model.capabilities),
        "artifacts": [_model_artifact_profile_to_payload(item) for item in model.artifacts],
        "metadata_sources": [source.value for source in model.metadata_sources],
    }


def _model_snapshot_from_payload(payload: Mapping[str, Any]) -> ModelProfile:
    return ModelProfile(
        model_id=str(payload["model_id"]),
        family=ModelFamily(str(payload["family"])),
        modality=ModelModality(str(payload["modality"])),
        source=str(payload["source"]),
        task=str(payload["task"]),
        parameter_count=_optional_int(payload.get("parameter_count")),
        active_parameter_count=_optional_int(payload.get("active_parameter_count")),
        context_length=_optional_int(payload.get("context_length")),
        max_batch_size_hint=_optional_int(payload.get("max_batch_size_hint")),
        architecture=_optional_str(payload.get("architecture")),
        dtype_options=[str(item) for item in payload.get("dtype_options", [])],
        quantization_options=[str(item) for item in payload.get("quantization_options", [])],
        capabilities=[str(item) for item in payload.get("capabilities", [])],
        artifacts=[_model_artifact_profile_from_payload(item) for item in payload.get("artifacts", [])],
        metadata_sources=[EstimateSource(str(item)) for item in payload.get("metadata_sources", [])],
    )


def _model_artifact_profile_to_payload(profile: ModelArtifactProfile) -> dict[str, Any]:
    return {
        "source": profile.source,
        "repository_id": profile.repository_id,
        "revision": profile.revision,
        "format": profile.format,
        "quantization": profile.quantization,
        "artifact_size_gb": profile.artifact_size_gb,
        "license_name": profile.license_name,
        "gated": profile.gated,
        "dependency_hints": list(profile.dependency_hints),
        "processor_required": profile.processor_required,
    }


def _model_artifact_profile_from_payload(payload: Mapping[str, Any]) -> ModelArtifactProfile:
    return ModelArtifactProfile(
        source=str(payload["source"]),
        repository_id=str(payload["repository_id"]),
        revision=_optional_str(payload.get("revision")),
        format=_optional_str(payload.get("format")),
        quantization=_optional_str(payload.get("quantization")),
        artifact_size_gb=_optional_float(payload.get("artifact_size_gb")),
        license_name=_optional_str(payload.get("license_name")),
        gated=bool(payload.get("gated", False)),
        dependency_hints=[str(item) for item in payload.get("dependency_hints", [])],
        processor_required=bool(payload.get("processor_required", False)),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
