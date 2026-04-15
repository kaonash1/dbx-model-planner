from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ...domain.profiles import ModelProfile


@dataclass(slots=True)
class HuggingFaceRepoMetadata:
    """Offline snapshot of the Hugging Face repo metadata surface."""

    repository_id: str
    revision: str | None = None
    commit_sha: str | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    tags: list[str] = field(default_factory=list)
    siblings: list[dict[str, Any] | str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    processor: dict[str, Any] = field(default_factory=dict)
    card_data: dict[str, Any] = field(default_factory=dict)
    license_name: str | None = None
    gated: bool = False
    sha: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HuggingFaceRepoMetadata":
        """Create a normalized fixture object from hub-like metadata."""

        repository_id = str(raw.get("repository_id") or raw.get("repo_id") or raw.get("id") or "")
        if not repository_id:
            raise ValueError("repository_id is required")

        return cls(
            repository_id=repository_id,
            revision=_as_optional_str(raw.get("revision")),
            commit_sha=_as_optional_str(raw.get("commit_sha") or raw.get("sha")),
            pipeline_tag=_as_optional_str(raw.get("pipeline_tag") or raw.get("task")),
            library_name=_as_optional_str(raw.get("library_name")),
            tags=_as_str_list(raw.get("tags")),
            siblings=_as_siblings(raw.get("siblings")),
            config=_as_mapping(raw.get("config")),
            tokenizer=_as_mapping(raw.get("tokenizer")),
            processor=_as_mapping(raw.get("processor")),
            card_data=_as_mapping(raw.get("card_data")),
            license_name=_as_optional_str(raw.get("license_name") or raw.get("license")),
            gated=bool(raw.get("gated", False)),
            sha=_as_optional_str(raw.get("sha") or raw.get("commit_sha")),
        )


@dataclass(slots=True)
class HuggingFaceArtifactManifest:
    """Preflight manifest for a Hugging Face repository."""

    repository_id: str
    revision: str | None = None
    commit_sha: str | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    license_name: str | None = None
    gated: bool = False
    weight_files: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    tokenizer_files: list[str] = field(default_factory=list)
    processor_files: list[str] = field(default_factory=list)
    auxiliary_files: list[str] = field(default_factory=list)
    config_keys: list[str] = field(default_factory=list)
    tokenizer_keys: list[str] = field(default_factory=list)
    processor_keys: list[str] = field(default_factory=list)
    has_config: bool = False
    has_tokenizer: bool = False
    has_processor: bool = False
    has_image_processor: bool = False
    file_count: int = 0
    artifact_format: str | None = None
    dependency_hints: list[str] = field(default_factory=list)
    preflight_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HuggingFaceNormalizedModel:
    """Normalized HF model data ready for planner preflight."""

    model_profile: ModelProfile
    artifact_manifest: HuggingFaceArtifactManifest
    preflight_notes: list[str] = field(default_factory=list)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_str_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_siblings(value: Any) -> list[dict[str, Any] | str]:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        siblings: list[dict[str, Any] | str] = []
        for item in value:
            if isinstance(item, Mapping):
                siblings.append(dict(item))
            else:
                siblings.append(str(item))
        return siblings
    if isinstance(value, Mapping):
        return [dict(value)]
    return [str(value)]
