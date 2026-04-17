"""Curated model catalog and HuggingFace trending discovery."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ...auth import HuggingFaceCredentials
from .models import HF_API_BASE, HF_USER_AGENT


@dataclass(slots=True)
class CatalogEntry:
    """A lightweight model entry for the browse list."""

    model_id: str
    provider: str
    params_label: str
    params_raw: int
    category: str  # "LLM", "Embedding", "VLM", "Code"
    use_case: str
    context_length: int | None = None
    downloads: int | None = None
    gated: bool = False
    discovered: bool = False  # True if fetched from HF trending


# -- Curated catalog -------------------------------------------------------

CURATED_MODELS: list[CatalogEntry] = [
    # --- Meta Llama ---
    CatalogEntry("meta-llama/Llama-3.1-8B-Instruct", "Meta", "8.0B", 8_030_000_000, "LLM", "Chat, instruction following", 131072, gated=True),
    CatalogEntry("meta-llama/Llama-3.1-70B-Instruct", "Meta", "70.6B", 70_554_000_000, "LLM", "Large-scale chat, reasoning", 131072, gated=True),
    CatalogEntry("meta-llama/Llama-3.2-3B", "Meta", "3.2B", 3_210_000_000, "LLM", "Lightweight, edge", 131072, gated=True),
    CatalogEntry("meta-llama/Llama-3.2-11B-Vision-Instruct", "Meta", "11.0B", 10_665_000_000, "VLM", "Multimodal vision + text", 131072, gated=True),
    CatalogEntry("meta-llama/Llama-3.3-70B-Instruct", "Meta", "70.6B", 70_554_000_000, "LLM", "Latest 70B instruct", 131072, gated=True),
    # --- Mistral ---
    CatalogEntry("mistralai/Mistral-7B-Instruct-v0.3", "Mistral", "7.2B", 7_248_000_000, "LLM", "Chat, instruction following", 32768),
    CatalogEntry("mistralai/Mixtral-8x7B-Instruct-v0.1", "Mistral", "46.7B", 46_702_000_000, "LLM", "MoE, 12.9B active params", 32768),
    CatalogEntry("mistralai/Mistral-Small-24B-Instruct-2501", "Mistral", "24.0B", 24_000_000_000, "LLM", "Balanced performance", 32768, gated=True),
    # --- Qwen ---
    CatalogEntry("Qwen/Qwen2.5-7B-Instruct", "Alibaba", "7.6B", 7_616_000_000, "LLM", "Chat, tool use", 131072),
    CatalogEntry("Qwen/Qwen2.5-14B-Instruct", "Alibaba", "14.8B", 14_770_000_000, "LLM", "Mid-size, strong reasoning", 131072),
    CatalogEntry("Qwen/Qwen2.5-32B-Instruct", "Alibaba", "32.5B", 32_510_000_000, "LLM", "Large, high quality", 131072),
    CatalogEntry("Qwen/Qwen2.5-72B-Instruct", "Alibaba", "72.7B", 72_710_000_000, "LLM", "Flagship, GPT-4 class", 131072),
    CatalogEntry("Qwen/Qwen2.5-Coder-32B-Instruct", "Alibaba", "32.5B", 32_510_000_000, "Code", "Code generation", 131072),
    CatalogEntry("Qwen/Qwen2.5-VL-7B-Instruct", "Alibaba", "7.6B", 7_616_000_000, "VLM", "Vision-language 7B", 32768),
    # --- Microsoft Phi ---
    CatalogEntry("microsoft/Phi-3.5-mini-instruct", "Microsoft", "3.8B", 3_821_000_000, "LLM", "Lightweight, long context", 131072),
    CatalogEntry("microsoft/phi-4", "Microsoft", "14.0B", 14_000_000_000, "LLM", "Reasoning, STEM, code", 16384),
    # --- Google Gemma ---
    CatalogEntry("google/gemma-2-9b-it", "Google", "9.2B", 9_241_000_000, "LLM", "Efficient, well-rounded", 8192),
    CatalogEntry("google/gemma-2-27b-it", "Google", "27.2B", 27_227_000_000, "LLM", "Large Gemma, strong", 8192),
    CatalogEntry("google/gemma-3-12b-it", "Google", "12.0B", 12_000_000_000, "VLM", "Multimodal vision + text", 131072),
    # --- DeepSeek ---
    CatalogEntry("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek", "7.6B", 7_616_000_000, "LLM", "Reasoning distilled", 131072),
    CatalogEntry("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "DeepSeek", "32.8B", 32_760_000_000, "LLM", "Reasoning distilled, large", 131072),
    # --- Embeddings ---
    CatalogEntry("BAAI/bge-large-en-v1.5", "BAAI", "335M", 335_000_000, "Embedding", "Text embeddings for RAG", 512),
    CatalogEntry("sentence-transformers/all-MiniLM-L6-v2", "SBERT", "22M", 22_700_000, "Embedding", "Lightweight embeddings", 256),
    CatalogEntry("nomic-ai/nomic-embed-text-v1.5", "Nomic", "137M", 137_000_000, "Embedding", "Long-context embeddings", 8192),
    CatalogEntry("BAAI/bge-m3", "BAAI", "568M", 568_000_000, "Embedding", "Multilingual, multi-granularity", 8192),
    # --- Code ---
    CatalogEntry("bigcode/starcoder2-15b", "BigCode", "15.7B", 15_700_000_000, "Code", "Code generation", 16384),
    CatalogEntry("meta-llama/CodeLlama-34b-Instruct-hf", "Meta", "34.0B", 34_019_000_000, "Code", "Code generation, large", 16384, gated=True),
    # --- Small / edge ---
    CatalogEntry("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Community", "1.1B", 1_100_000_000, "LLM", "Ultra-lightweight", 2048),
    CatalogEntry("HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFace", "1.7B", 1_710_000_000, "LLM", "Small, efficient", 8192),
]

# -- HuggingFace trending discovery ----------------------------------------

_PIPELINE_TAGS = [
    "text-generation",
    "feature-extraction",
    "image-text-to-text",
]

_CATEGORY_MAP = {
    "text-generation": "LLM",
    "text2text-generation": "LLM",
    "conversational": "LLM",
    "feature-extraction": "Embedding",
    "sentence-similarity": "Embedding",
    "image-text-to-text": "VLM",
    "visual-question-answering": "VLM",
}

_PROVIDER_MAP = {
    "meta-llama": "Meta",
    "mistralai": "Mistral",
    "qwen": "Alibaba",
    "microsoft": "Microsoft",
    "google": "Google",
    "deepseek-ai": "DeepSeek",
    "bigcode": "BigCode",
    "baai": "BAAI",
    "nomic-ai": "Nomic",
    "sentence-transformers": "SBERT",
    "tiiuae": "TII",
    "stabilityai": "Stability AI",
    "cohereforai": "Cohere",
    "tinyllama": "Community",
    "huggingfacetb": "HuggingFace",
    "ibm-granite": "IBM",
    "nvidia": "NVIDIA",
    "01-ai": "01.ai",
}

_SKIP_ORGS = {
    "thebloke", "unsloth", "mlx-community", "bartowski",
    "mradermacher", "trl-internal-testing", "openai-community",
}


def _format_params(total: int) -> str:
    if total >= 1_000_000_000:
        val = total / 1_000_000_000
        return f"{val:.1f}B"
    elif total >= 1_000_000:
        val = total / 1_000_000
        return f"{val:.0f}M"
    return f"{total // 1_000}K"


def _extract_provider(repo_id: str) -> str:
    org = repo_id.split("/")[0].lower()
    return _PROVIDER_MAP.get(org, org.title())


def discover_trending_models(
    credentials: HuggingFaceCredentials | None = None,
    limit: int = 30,
    timeout: float = 20.0,
) -> list[CatalogEntry]:
    """Fetch trending models from HuggingFace API.

    Returns CatalogEntry items for models NOT already in the curated list.
    """
    curated_ids = {m.model_id for m in CURATED_MODELS}
    discovered: list[CatalogEntry] = []
    seen: set[str] = set()

    headers: dict[str, str] = {"User-Agent": HF_USER_AGENT}
    if credentials and credentials.token:
        headers["Authorization"] = f"Bearer {credentials.token}"

    for pipeline in _PIPELINE_TAGS:
        url = (
            f"{HF_API_BASE}/models?"
            f"pipeline_tag={pipeline}&"
            f"sort=downloads&direction=-1&"
            f"limit={limit * 3}&"
            f"expand[]=safetensors"
        )
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                models: list[dict[str, Any]] = json.loads(resp.read().decode())
        except Exception:
            continue

        for m in models:
            repo_id = m.get("id", "")
            if not repo_id or "/" not in repo_id:
                continue
            if repo_id in curated_ids or repo_id in seen:
                continue

            org = repo_id.split("/")[0].lower()
            if org in _SKIP_ORGS:
                continue

            downloads = m.get("downloads", 0)
            if downloads < 5000:
                continue

            # Skip GGUF/adapter/merge repos
            tags = set(m.get("tags", []))
            if tags & {"gguf", "adapter", "merge", "lora", "qlora"}:
                continue

            # Extract param count from safetensors
            safetensors = m.get("safetensors", {})
            total_params = safetensors.get("total")
            if not total_params:
                params_by_dtype = safetensors.get("parameters", {})
                if params_by_dtype:
                    total_params = max(params_by_dtype.values())
            if not total_params:
                continue

            seen.add(repo_id)
            category = _CATEGORY_MAP.get(pipeline, "LLM")

            # Infer use case from name
            rid = repo_id.lower()
            if "embed" in rid or "bge" in rid:
                use_case = "Text embeddings"
            elif "coder" in rid or "code" in rid:
                use_case = "Code generation"
                category = "Code"
            elif "vision" in rid or "-vl-" in rid:
                use_case = "Vision-language"
                category = "VLM"
            elif "instruct" in rid or "chat" in rid:
                use_case = "Chat, instruction following"
            else:
                use_case = "General purpose"

            entry = CatalogEntry(
                model_id=repo_id,
                provider=_extract_provider(repo_id),
                params_label=_format_params(total_params),
                params_raw=total_params,
                category=category,
                use_case=use_case,
                downloads=downloads,
                gated=bool(m.get("gated", False)),
                discovered=True,
            )
            discovered.append(entry)
            if len(discovered) >= limit:
                break

        if len(discovered) >= limit:
            break

    return discovered


def get_full_catalog(
    discovered: list[CatalogEntry] | None = None,
) -> list[CatalogEntry]:
    """Return curated models + any discovered models, deduplicated."""
    curated_ids = {m.model_id for m in CURATED_MODELS}
    result = list(CURATED_MODELS)
    if discovered:
        for entry in discovered:
            if entry.model_id not in curated_ids:
                result.append(entry)
    return result
