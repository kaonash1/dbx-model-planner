from __future__ import annotations

from typing import Any, Iterable, Mapping

from ...domain.common import EstimateSource, ModelFamily, ModelModality
from ...domain.profiles import ModelArtifactProfile, ModelProfile
from .models import HuggingFaceArtifactManifest, HuggingFaceNormalizedModel, HuggingFaceRepoMetadata

_EMBEDDING_PIPELINES = {
    "feature-extraction",
    "sentence-similarity",
    "text-embedding",
}

_VLM_PIPELINES = {
    "image-text-to-text",
    "visual-question-answering",
    "zero-shot-image-classification",
    "image-to-text",
}

_LLM_PIPELINES = {
    "conversational",
    "text-generation",
    "text2text-generation",
}


def normalize_huggingface_repo_metadata(
    raw_metadata: HuggingFaceRepoMetadata | Mapping[str, Any],
) -> HuggingFaceNormalizedModel:
    """Normalize a HF repo snapshot into planner-friendly model data."""

    metadata = raw_metadata if isinstance(raw_metadata, HuggingFaceRepoMetadata) else HuggingFaceRepoMetadata.from_mapping(raw_metadata)

    family, modality, task, classification_notes = _classify(metadata)
    manifest = _build_manifest(metadata, family, modality, task, classification_notes)
    model_profile = _build_model_profile(metadata, family, modality, task, manifest)

    return HuggingFaceNormalizedModel(
        model_profile=model_profile,
        artifact_manifest=manifest,
        preflight_notes=manifest.preflight_notes,
    )


def _classify(metadata: HuggingFaceRepoMetadata) -> tuple[ModelFamily, ModelModality, str, list[str]]:
    pipeline_tag = (metadata.pipeline_tag or "").strip().lower()
    tags = {tag.strip().lower() for tag in metadata.tags if tag.strip()}
    config = metadata.config
    notes: list[str] = []

    if _looks_like_vlm(pipeline_tag, tags, config, metadata.processor):
        task = metadata.pipeline_tag or _guess_task_from_config(config, fallback="image-text-to-text")
        return ModelFamily.VLM, ModelModality.IMAGE_TEXT, task, ["classified as VLM because processor/config surface is multimodal"]

    if pipeline_tag in _EMBEDDING_PIPELINES or _has_embedding_signals(tags, config):
        task = metadata.pipeline_tag or _guess_task_from_config(config, fallback="feature-extraction")
        return ModelFamily.EMBEDDING, ModelModality.TEXT_EMBEDDING, task, ["classified as embedding model from task/tags"]

    if pipeline_tag in _LLM_PIPELINES or _looks_like_llm(tags, config):
        task = metadata.pipeline_tag or _guess_task_from_config(config, fallback="text-generation")
        return ModelFamily.LLM, ModelModality.TEXT, task, ["classified as text generation model from task/config"]

    task = metadata.pipeline_tag or _guess_task_from_config(config, fallback="unknown")
    notes.append("fell back to OTHER because no family-specific signals were found")
    return ModelFamily.OTHER, ModelModality.TEXT, task, notes


def _looks_like_vlm(pipeline_tag: str, tags: set[str], config: Mapping[str, Any], processor: Mapping[str, Any]) -> bool:
    if pipeline_tag in _VLM_PIPELINES:
        return True
    if any(tag in {"vlm", "vision", "multimodal", "image-text"} for tag in tags):
        return True
    if "vision_config" in config or "vision_model_name_or_path" in config:
        return True
    if any(key in processor for key in ("image_processor_type", "image_processor_class", "feature_extractor_type")):
        return True
    return False


def _has_embedding_signals(tags: set[str], config: Mapping[str, Any]) -> bool:
    if any(tag in {"embedding", "sentence-transformers", "sentence_transformers", "bi-encoder", "reranker"} for tag in tags):
        return True
    if str(config.get("model_type", "")).lower() in {"sentence-transformer", "sentence_transformer"}:
        return True
    if "pooling_mode" in config or "normalize_embeddings" in config:
        return True
    return False


def _looks_like_llm(tags: set[str], config: Mapping[str, Any]) -> bool:
    if any(tag in {"llm", "causal-lm", "language-model", "instruct"} for tag in tags):
        return True
    model_type = str(config.get("model_type", "")).lower()
    if model_type in {"llama", "mistral", "gpt_neox", "gpt2", "falcon", "phi", "qwen2", "gemma"}:
        return True
    if "architectures" in config:
        architectures = _stringify_iterable(config.get("architectures"))
        if any(name.lower().endswith("forcausallm") or name.lower().endswith("lmheadmodel") for name in architectures):
            return True
    return False


def _guess_task_from_config(config: Mapping[str, Any], fallback: str) -> str:
    for key in ("pipeline_tag", "task", "model_task"):
        value = config.get(key)
        if value:
            return str(value)
    return fallback


def _build_manifest(
    metadata: HuggingFaceRepoMetadata,
    family: ModelFamily,
    modality: ModelModality,
    task: str,
    classification_notes: list[str],
) -> HuggingFaceArtifactManifest:
    file_names = _file_names(metadata.siblings)
    config_files = [name for name in file_names if _is_config_file(name)]
    tokenizer_files = [name for name in file_names if _is_tokenizer_file(name)]
    processor_files = [name for name in file_names if _is_processor_file(name)]
    weight_files = [name for name in file_names if _is_weight_file(name)]
    auxiliary_files = [
        name
        for name in file_names
        if name not in config_files and name not in tokenizer_files and name not in processor_files and name not in weight_files
    ]

    config_keys = sorted(metadata.config.keys())
    tokenizer_keys = sorted(metadata.tokenizer.keys())
    processor_keys = sorted(metadata.processor.keys())
    has_processor = bool(metadata.processor) or bool(processor_files)
    has_image_processor = has_processor and any(
        key in metadata.processor for key in ("image_processor_type", "feature_extractor_type", "image_processor_class")
    )

    dependency_hints = _dependency_hints(family, modality, has_processor, has_image_processor)
    notes = list(classification_notes)
    if metadata.gated:
        notes.append("repository is gated")
    if not config_files:
        notes.append("no config file was present in the sibling list")
    if not tokenizer_files:
        notes.append("no tokenizer file was present in the sibling list")
    if family is ModelFamily.VLM and not has_processor:
        notes.append("VLM preflight should include processor metadata")
    if family is ModelFamily.EMBEDDING:
        notes.append("embedding preflight should keep tokenizer and pooling metadata visible")

    artifact_format = _guess_artifact_format(weight_files)

    return HuggingFaceArtifactManifest(
        repository_id=metadata.repository_id,
        revision=metadata.revision,
        commit_sha=metadata.commit_sha or metadata.sha,
        pipeline_tag=metadata.pipeline_tag or task,
        library_name=metadata.library_name,
        license_name=metadata.license_name,
        gated=metadata.gated,
        weight_files=weight_files,
        config_files=config_files,
        tokenizer_files=tokenizer_files,
        processor_files=processor_files,
        auxiliary_files=auxiliary_files,
        config_keys=config_keys,
        tokenizer_keys=tokenizer_keys,
        processor_keys=processor_keys,
        has_config=bool(metadata.config),
        has_tokenizer=bool(metadata.tokenizer),
        has_processor=has_processor,
        has_image_processor=has_image_processor,
        file_count=len(file_names),
        artifact_format=artifact_format,
        dependency_hints=dependency_hints,
        preflight_notes=notes,
    )


def _build_model_profile(
    metadata: HuggingFaceRepoMetadata,
    family: ModelFamily,
    modality: ModelModality,
    task: str,
    manifest: HuggingFaceArtifactManifest,
) -> ModelProfile:
    config = metadata.config
    parameter_count = _as_int(
        config.get("num_parameters")
        or config.get("parameter_count")
        or config.get("n_parameters")
        or config.get("n_params")
    )
    active_parameter_count = _as_int(config.get("active_parameter_count") or config.get("active_parameters"))
    context_length = _as_int(
        config.get("context_length")
        or config.get("max_position_embeddings")
        or config.get("max_seq_len")
        or config.get("seq_length")
    )
    max_batch_size_hint = _as_int(config.get("max_batch_size") or config.get("batch_size"))
    architecture = _first_text(config.get("architectures")) or _as_optional_text(config.get("model_type"))
    dtype_options = _collect_dtypes(config, metadata.tags)
    quantization_options = _collect_quantization_options(config, metadata.tags)
    capabilities = _collect_capabilities(metadata, family, modality)

    artifact_profile = ModelArtifactProfile(
        source="huggingface",
        repository_id=metadata.repository_id,
        revision=metadata.revision or metadata.commit_sha or metadata.sha,
        format=manifest.artifact_format,
        quantization=quantization_options[0] if quantization_options else None,
        artifact_size_gb=_as_float(config.get("artifact_size_gb") or config.get("file_size_gb")),
        license_name=metadata.license_name,
        gated=metadata.gated,
        dependency_hints=manifest.dependency_hints,
        processor_required=family is ModelFamily.VLM or manifest.has_processor,
    )

    return ModelProfile(
        model_id=metadata.repository_id,
        family=family,
        modality=modality,
        source="huggingface",
        task=task,
        parameter_count=parameter_count,
        active_parameter_count=active_parameter_count,
        context_length=context_length,
        max_batch_size_hint=max_batch_size_hint,
        architecture=architecture,
        dtype_options=dtype_options,
        quantization_options=quantization_options,
        capabilities=capabilities,
        artifacts=[artifact_profile],
        metadata_sources=[EstimateSource.DISCOVERED],
    )


def _file_names(siblings: Iterable[dict[str, Any] | str]) -> list[str]:
    names: list[str] = []
    for sibling in siblings:
        if isinstance(sibling, str):
            names.append(sibling)
            continue
        for key in ("rfilename", "filename", "name", "path"):
            value = sibling.get(key)
            if value:
                names.append(str(value))
                break
    return names


def _is_config_file(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith("config.json") or lowered.endswith("generation_config.json") or lowered.endswith("adapter_config.json")


def _is_tokenizer_file(name: str) -> bool:
    lowered = name.lower()
    return any(
        lowered.endswith(suffix)
        for suffix in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "spiece.model",
        )
    )


def _is_processor_file(name: str) -> bool:
    lowered = name.lower()
    return any(
        lowered.endswith(suffix)
        for suffix in (
            "preprocessor_config.json",
            "processor_config.json",
            "image_processor_config.json",
            "chat_template.json",
        )
    )


def _is_weight_file(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in (".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf"))


def _guess_artifact_format(weight_files: list[str]) -> str | None:
    if not weight_files:
        return None
    for suffix in (".safetensors", ".gguf", ".onnx", ".bin", ".pt", ".pth"):
        if any(name.lower().endswith(suffix) for name in weight_files):
            return suffix.lstrip(".")
    return "weights"


def _dependency_hints(
    family: ModelFamily,
    modality: ModelModality,
    has_processor: bool,
    has_image_processor: bool,
) -> list[str]:
    hints = ["transformers"]
    if family is ModelFamily.EMBEDDING:
        hints.append("sentence-transformers")
    if family is ModelFamily.VLM or modality is ModelModality.IMAGE_TEXT:
        hints.extend(["pillow", "torchvision"])
    if family is ModelFamily.VLM or has_processor:
        hints.append("processor-config")
    if has_image_processor:
        hints.append("image-processor")
    return sorted(dict.fromkeys(hints))


def _collect_dtypes(config: Mapping[str, Any], tags: set[str]) -> list[str]:
    dtype_options: set[str] = set()
    for key in ("torch_dtype", "dtype", "model_dtype"):
        value = config.get(key)
        if value:
            dtype_options.add(str(value))
    for tag in tags:
        if tag in {"fp16", "bf16", "int8", "int4", "fp8"}:
            dtype_options.add(tag)
    return sorted(dtype_options)


def _collect_quantization_options(config: Mapping[str, Any], tags: set[str]) -> list[str]:
    options: set[str] = set()
    for key in ("quantization", "quantization_config", "load_in_8bit", "load_in_4bit"):
        value = config.get(key)
        if not value:
            continue
        if isinstance(value, Mapping):
            options.update(str(item) for item in value.keys())
        else:
            options.add(str(value))
    for tag in tags:
        if tag in {"gptq", "awq", "bnb", "bitsandbytes", "int8", "int4"}:
            options.add(tag)
    return sorted(options)


def _collect_capabilities(metadata: HuggingFaceRepoMetadata, family: ModelFamily, modality: ModelModality) -> list[str]:
    capabilities = {metadata.pipeline_tag or "unknown"}
    if family is ModelFamily.LLM:
        capabilities.update({"text-generation", "prompting"})
    elif family is ModelFamily.EMBEDDING:
        capabilities.update({"embedding", "similarity-search"})
    elif family is ModelFamily.VLM:
        capabilities.update({"vision-language", "multimodal"})
    if modality is ModelModality.IMAGE_TEXT:
        capabilities.add("image-input")
    return sorted(capabilities)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if item:
                return str(item)
    return None


def _as_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stringify_iterable(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]
