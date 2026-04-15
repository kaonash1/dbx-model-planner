from __future__ import annotations

from dataclasses import dataclass, field
import re


_KNOWN_GPU_FAMILIES = (
    "A100",
    "A10",
    "H100",
    "L4",
    "L40S",
    "T4",
    "V100",
)


def normalize_azure_token(value: str | None) -> str | None:
    """Normalize Azure-like labels into a stable lowercase token."""

    if value is None:
        return None
    token = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or None


def normalize_node_type_id(node_type_id: str) -> str:
    """Normalize a Databricks node type into a canonical Azure VM SKU candidate."""

    cleaned = node_type_id.strip().replace("-", "_")
    cleaned = re.sub(r"^standard_", "Standard_", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Standard__+", "Standard_", cleaned)
    return cleaned


def arm_sku_candidates_from_node_type(node_type_id: str) -> list[str]:
    """Return candidate Azure ARM SKU names for a Databricks node type."""

    normalized = normalize_node_type_id(node_type_id)
    base = re.sub(r"^Standard_", "", normalized)
    candidates = [normalized]

    if not normalized.startswith("Standard_"):
        candidates.append(f"Standard_{base}")

    if "_" not in base:
        candidates.append(f"Standard_{base.upper()}")

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def infer_gpu_family(node_type_id: str) -> str | None:
    """Extract an accelerator family token such as A100 or T4 when present."""

    normalized = normalize_node_type_id(node_type_id)
    for family in _KNOWN_GPU_FAMILIES:
        if re.search(rf"(?:^|_){re.escape(family)}(?:_|$)", normalized, flags=re.IGNORECASE):
            return family
    return None


def infer_vm_series(node_type_id: str) -> str | None:
    """Extract the Azure VM series token from a Databricks node type."""

    normalized = normalize_node_type_id(node_type_id)
    base = re.sub(r"^Standard_", "", normalized)
    if not base:
        return None
    return base.split("_", 1)[0]


@dataclass(slots=True)
class AzureSkuMapping:
    """Heuristic Azure SKU mapping for a Databricks node type."""

    node_type_id: str
    normalized_node_type_id: str
    arm_sku_candidates: list[str] = field(default_factory=list)
    vm_series: str | None = None
    vm_family: str | None = None
    gpu_family: str | None = None
    notes: list[str] = field(default_factory=list)


def map_node_type_to_azure_sku(node_type_id: str) -> AzureSkuMapping:
    """Build a structured Azure SKU mapping from a Databricks node type string."""

    normalized = normalize_node_type_id(node_type_id)
    vm_series = infer_vm_series(node_type_id)
    vm_family = normalize_azure_token(re.sub(r"\d.*$", "", vm_series or ""))
    gpu_family = infer_gpu_family(node_type_id)
    notes: list[str] = []

    if gpu_family is None:
        notes.append("No explicit accelerator family token was found in the node type.")

    if normalized != node_type_id:
        notes.append("Node type was normalized before deriving Azure SKU candidates.")

    return AzureSkuMapping(
        node_type_id=node_type_id,
        normalized_node_type_id=normalized,
        arm_sku_candidates=arm_sku_candidates_from_node_type(node_type_id),
        vm_series=vm_series,
        vm_family=vm_family,
        gpu_family=gpu_family,
        notes=notes,
    )
