from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ...auth import DatabricksCredentials
from ...domain import (
    Cloud,
    EstimateSource,
    RuntimeProfile,
    WorkspaceComputeProfile,
    WorkspaceInventorySnapshot,
    WorkspacePolicyProfile,
)


class DatabricksAPIError(Exception):
    """Raised when a Databricks REST API call fails."""


@dataclass(slots=True)
class DatabricksInventoryCollection:
    """Normalized inventory facts from a live workspace."""

    snapshot: WorkspaceInventorySnapshot
    notes: list[str] = field(default_factory=list)


# Known GPU memory per device (GB) for common Azure GPU SKUs.
_GPU_MEMORY_MAP: dict[str, float] = {
    "A100_80": 80.0,
    "A100_40": 40.0,
    "A10": 24.0,
    "H100": 80.0,
    "V100": 16.0,
    "T4": 16.0,
    "K80": 12.0,
    "L4": 24.0,
    "L40S": 48.0,
}

# Patterns to extract GPU family from Azure node type IDs.
# Order matters: more specific patterns first, catch-all last.
_GPU_PATTERNS: list[tuple[str, str]] = [
    (r"NC(\d+)ads_A100", "A100_80"),
    (r"ND(\d+)asr_v4", "A100_40"),
    (r"ND(\d+)rs_v2", "A100_80"),
    (r"NC(\d+)ads_A10", "A10"),
    (r"NV(\d+)ad[ms]*_A10", "A10"),
    (r"ND(\d+)s_H100", "H100"),
    (r"ND(\d+)i?sr_H100", "H100"),
    (r"NC(\d+)s_v3", "V100"),
    (r"NC(\d+)_T4", "T4"),
    (r"NC(\d+)as_T4", "T4"),
    (r"NC(\d+)$", "K80"),
    (r"H100", "H100"),
    (r"V100", "V100"),
    (r"T4", "T4"),
    (r"A100", "A100_80"),
    (r"A10", "A10"),
    (r"L4", "L4"),
    (r"L40", "L40S"),
]


class DatabricksInventoryCollector:
    """Live Databricks inventory collector.

    Connects to a Databricks workspace via REST API to fetch node types,
    runtimes, and cluster policies. Requires valid credentials.
    """

    def __init__(self, credentials: DatabricksCredentials) -> None:
        self._credentials = credentials
        self._host = credentials.host.rstrip("/")

    def collect(
        self,
        *,
        progress_fn: Callable[[str], None] | None = None,
    ) -> DatabricksInventoryCollection:
        """Collect full workspace inventory from the live Databricks API."""

        notes: list[str] = []

        if progress_fn:
            progress_fn("Fetching node types...")
        node_types = self._fetch_node_types()
        notes.append(f"Fetched {len(node_types)} node types")

        if progress_fn:
            progress_fn("Fetching runtime versions...")
        dbr_versions = self._fetch_dbr_versions()
        notes.append(f"Fetched {len(dbr_versions)} runtime versions")

        if progress_fn:
            progress_fn("Fetching cluster policies...")
        policies = self._fetch_policies()
        notes.append(f"Fetched {len(policies)} cluster policies")

        snapshot = WorkspaceInventorySnapshot(
            workspace_url=self._host,
            cloud=Cloud.AZURE,
            region=None,
            compute=node_types,
            runtimes=dbr_versions,
            policies=policies,
        )
        return DatabricksInventoryCollection(snapshot=snapshot, notes=notes)

    def collect_snapshot(self) -> WorkspaceInventorySnapshot:
        """Convenience: return only the snapshot."""
        return self.collect().snapshot

    def _api_get(self, path: str, *, timeout: float = 30.0) -> dict[str, Any]:
        """Make an authenticated GET request to the Databricks REST API."""

        url = f"{self._host}{path}"
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._credentials.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise DatabricksAPIError(
                    "Authentication failed. Check your Databricks token with 'dbx-model-planner auth login'."
                ) from exc
            elif exc.code == 403:
                raise DatabricksAPIError(
                    f"Forbidden: insufficient permissions for {path}"
                ) from exc
            raise DatabricksAPIError(f"API error {exc.code} on {path}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise DatabricksAPIError(
                f"Network error connecting to {self._host}: {exc.reason}"
            ) from exc

    def _fetch_node_types(self) -> list[WorkspaceComputeProfile]:
        data = self._api_get("/api/2.0/clusters/list-node-types")

        node_types: list[WorkspaceComputeProfile] = []
        for item in data.get("node_types", []):
            node_type_id = item.get("node_type_id", "")
            num_gpus = item.get("num_gpus", 0)
            num_cores = item.get("num_cores")

            node_types.append(WorkspaceComputeProfile(
                node_type_id=node_type_id,
                cloud=Cloud.AZURE,
                gpu_family=_extract_gpu_family(node_type_id),
                gpu_count=num_gpus,
                gpu_memory_gb=_extract_gpu_memory(node_type_id, num_gpus),
                vcpu_count=num_cores,
                memory_gb=item.get("memory_mb", 0) / 1024 if item.get("memory_mb") else None,
                dbu_per_hour=None,  # Set later from Azure pricing page data
                availability_source=EstimateSource.DISCOVERED,
            ))
        return node_types

    def _fetch_dbr_versions(self) -> list[RuntimeProfile]:
        data = self._api_get("/api/2.0/clusters/spark-versions")

        runtimes: list[RuntimeProfile] = []
        for item in data.get("versions", []):
            key = item.get("key", "")
            name = item.get("name", "")
            name_lower = name.lower()
            key_lower = key.lower()

            runtimes.append(RuntimeProfile(
                runtime_id=key,
                dbr_version=name,
                ml_runtime="ml" in name_lower or "-ml-" in key_lower,
                gpu_enabled="gpu" in name_lower or "gpu" in key_lower,
                photon_supported="photon" in name_lower,
            ))
        return runtimes

    def _fetch_policies(self) -> list[WorkspacePolicyProfile]:
        data = self._api_get("/api/2.0/policies/clusters/list")

        policies: list[WorkspacePolicyProfile] = []
        for item in data.get("policies", []):
            definition = item.get("definition", {})
            if isinstance(definition, str):
                try:
                    definition = json.loads(definition)
                except json.JSONDecodeError:
                    definition = {}

            node_type_constraint = definition.get("node_type_id", {})
            allowed_nodes: list[str] = []
            if isinstance(node_type_constraint, dict):
                constraint_type = node_type_constraint.get("type", "")
                if constraint_type == "allowlist":
                    allowed_nodes = [str(v) for v in node_type_constraint.get("values", []) if v]
                elif constraint_type == "fixed":
                    fixed_val = node_type_constraint.get("value")
                    if fixed_val:
                        allowed_nodes = [str(fixed_val)]

            policies.append(WorkspacePolicyProfile(
                policy_id=str(item.get("policy_id", "")),
                policy_name=item.get("name", ""),
                allowed_node_types=allowed_nodes,
            ))
        return policies


def _extract_gpu_family(node_type_id: str) -> str | None:
    """Extract GPU family from an Azure VM node type ID."""
    for pattern, gpu in _GPU_PATTERNS:
        if re.search(pattern, node_type_id, re.IGNORECASE):
            return gpu
    return None


def _extract_gpu_memory(node_type_id: str, num_gpus: int) -> float | None:
    """Estimate per-GPU memory from known GPU families.

    Returns the memory in GB for a single GPU, or None if unknown.
    The caller can multiply by ``num_gpus`` to get total node GPU memory.
    """
    if num_gpus == 0:
        return None
    gpu_family = _extract_gpu_family(node_type_id)
    if gpu_family and gpu_family in _GPU_MEMORY_MAP:
        return _GPU_MEMORY_MAP[gpu_family]
    return None


def enrich_dbu_rates(
    nodes: list[WorkspaceComputeProfile],
    dbu_rates: dict[str, float],
) -> int:
    """Set ``dbu_per_hour`` on each node from a {node_type_id: rate} lookup.

    Returns the number of nodes that were enriched.
    """
    count = 0
    for node in nodes:
        rate = dbu_rates.get(node.node_type_id)
        if rate is not None:
            node.dbu_per_hour = rate
            count += 1
    return count
