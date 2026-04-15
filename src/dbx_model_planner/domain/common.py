from __future__ import annotations

from enum import StrEnum


class Cloud(StrEnum):
    AZURE = "azure"


class ModelFamily(StrEnum):
    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    VLM = "vlm"
    AUDIO = "audio"
    OTHER = "other"


class ModelModality(StrEnum):
    TEXT = "text"
    TEXT_EMBEDDING = "text_embedding"
    IMAGE_TEXT = "image_text"
    AUDIO_TEXT = "audio_text"
    MULTIMODAL = "multimodal"


class HostingMode(StrEnum):
    FOUNDATION_API = "foundation_api"
    EXTERNAL_MODEL = "external_model"
    CUSTOM_SERVING = "custom_serving"
    CLASSIC_COMPUTE = "classic_compute"
    BATCH_COMPUTE = "batch_compute"


class FitLevel(StrEnum):
    SAFE = "safe"
    BORDERLINE = "borderline"
    UNLIKELY = "unlikely"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EstimateSource(StrEnum):
    DISCOVERED = "discovered"
    INFERRED = "inferred"
    USER_PROVIDED = "user_provided"
