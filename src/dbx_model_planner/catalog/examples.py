from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..adapters.huggingface import HuggingFaceNormalizedModel, normalize_huggingface_repo_metadata


@dataclass(slots=True, frozen=True)
class ExampleModelEntry:
    key: str
    label: str
    raw_metadata: dict[str, Any]


_EXAMPLE_MODELS: tuple[ExampleModelEntry, ...] = (
    ExampleModelEntry(
        key="llama-3.1-8b-instruct",
        label="Llama 3.1 8B Instruct",
        raw_metadata={
            "repository_id": "meta-llama/Llama-3.1-8B-Instruct",
            "revision": "main",
            "sha": "demo-llama31-8b",
            "pipeline_tag": "text-generation",
            "library_name": "transformers",
            "tags": ["llm", "text-generation", "instruct", "bf16", "int4"],
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "tokenizer.json"},
                {"rfilename": "tokenizer_config.json"},
                {"rfilename": "model-00001-of-00004.safetensors"},
            ],
            "config": {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_parameters": 8_000_000_000,
                "max_position_embeddings": 131072,
                "torch_dtype": "bfloat16",
            },
            "tokenizer": {"model_max_length": 131072, "use_fast": True},
        },
    ),
    ExampleModelEntry(
        key="mistral-7b-instruct",
        label="Mistral 7B Instruct",
        raw_metadata={
            "repository_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "revision": "main",
            "sha": "demo-mistral-7b",
            "pipeline_tag": "text-generation",
            "library_name": "transformers",
            "tags": ["llm", "text-generation", "fp16", "int8"],
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "tokenizer.json"},
                {"rfilename": "model.safetensors"},
            ],
            "config": {
                "architectures": ["MistralForCausalLM"],
                "model_type": "mistral",
                "num_parameters": 7_000_000_000,
                "max_position_embeddings": 32768,
                "torch_dtype": "float16",
            },
            "tokenizer": {"model_max_length": 32768, "use_fast": True},
        },
    ),
    ExampleModelEntry(
        key="bge-large-en-v1.5",
        label="BGE Large EN v1.5",
        raw_metadata={
            "repository_id": "BAAI/bge-large-en-v1.5",
            "revision": "main",
            "sha": "demo-bge-large",
            "pipeline_tag": "feature-extraction",
            "library_name": "sentence-transformers",
            "tags": ["embedding", "sentence-transformers", "sentence-similarity"],
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "tokenizer.json"},
                {"rfilename": "model.safetensors"},
            ],
            "config": {
                "model_type": "sentence-transformer",
                "num_parameters": 335_000_000,
                "pooling_mode": "cls",
                "normalize_embeddings": True,
                "max_seq_len": 512,
            },
            "tokenizer": {"model_max_length": 512, "use_fast": True},
        },
    ),
    ExampleModelEntry(
        key="e5-large-v2",
        label="E5 Large v2",
        raw_metadata={
            "repository_id": "intfloat/e5-large-v2",
            "revision": "main",
            "sha": "demo-e5-large",
            "pipeline_tag": "feature-extraction",
            "library_name": "sentence-transformers",
            "tags": ["embedding", "sentence-transformers"],
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "tokenizer.json"},
                {"rfilename": "model.safetensors"},
            ],
            "config": {
                "model_type": "sentence-transformer",
                "num_parameters": 335_000_000,
                "pooling_mode": "mean",
                "normalize_embeddings": True,
                "max_seq_len": 512,
            },
            "tokenizer": {"model_max_length": 512, "use_fast": True},
        },
    ),
    ExampleModelEntry(
        key="qwen2-vl-2b-instruct",
        label="Qwen2 VL 2B Instruct",
        raw_metadata={
            "repository_id": "Qwen/Qwen2-VL-2B-Instruct",
            "revision": "main",
            "sha": "demo-qwen2-vl-2b",
            "pipeline_tag": "image-text-to-text",
            "library_name": "transformers",
            "tags": ["vlm", "multimodal", "vision", "bf16", "int4"],
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "preprocessor_config.json"},
                {"rfilename": "tokenizer.json"},
                {"rfilename": "model.safetensors"},
            ],
            "config": {
                "architectures": ["Qwen2VLForConditionalGeneration"],
                "model_type": "qwen2_vl",
                "num_parameters": 2_000_000_000,
                "max_position_embeddings": 32768,
                "vision_config": {"image_size": 448},
                "torch_dtype": "bfloat16",
            },
            "tokenizer": {"model_max_length": 32768, "use_fast": True},
            "processor": {"image_processor_type": "Qwen2VLImageProcessor"},
        },
    ),
)


@lru_cache(maxsize=None)
def _normalize_example(key: str) -> HuggingFaceNormalizedModel:
    """Normalize and cache an example model entry by key."""

    for entry in _EXAMPLE_MODELS:
        if entry.key == key:
            return normalize_huggingface_repo_metadata(entry.raw_metadata)
    raise KeyError(f"Unknown example model key: {key}")


def list_example_models() -> list[tuple[str, str, str]]:
    models = []
    for entry in _EXAMPLE_MODELS:
        normalized = _normalize_example(entry.key)
        models.append((entry.key, entry.label, normalized.model_profile.model_id))
    return models


def resolve_example_model(model_ref: str) -> HuggingFaceNormalizedModel:
    needle = model_ref.strip().lower()
    for entry in _EXAMPLE_MODELS:
        normalized = _normalize_example(entry.key)
        if needle in {
            entry.key.lower(),
            entry.label.lower(),
            normalized.model_profile.model_id.lower(),
        }:
            return normalized
    available = ", ".join(entry.key for entry in _EXAMPLE_MODELS)
    raise KeyError(f"Unknown example model '{model_ref}'. Available examples: {available}")
