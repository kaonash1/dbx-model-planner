from __future__ import annotations

import unittest

from dbx_model_planner.adapters.huggingface import normalize_huggingface_repo_metadata
from dbx_model_planner.domain.common import ModelFamily, ModelModality


def llm_fixture() -> dict[str, object]:
    return {
        "repository_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "main",
        "sha": "8a1b2c3d",
        "pipeline_tag": "text-generation",
        "library_name": "transformers",
        "tags": ["llm", "text-generation", "fp16"],
        "siblings": [
            {"rfilename": "config.json"},
            {"rfilename": "tokenizer.json"},
            {"rfilename": "tokenizer_config.json"},
            {"rfilename": "model.safetensors"},
        ],
        "config": {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_parameters": 8000000000,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16",
        },
        "tokenizer": {"model_max_length": 8192, "use_fast": True},
    }


def embedding_fixture() -> dict[str, object]:
    return {
        "repository_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "refs/pr/12",
        "commit_sha": "abc12345",
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
            "pooling_mode": "mean",
            "normalize_embeddings": True,
            "max_seq_len": 256,
        },
        "tokenizer": {"model_max_length": 256, "use_fast": True},
    }


def vlm_fixture() -> dict[str, object]:
    return {
        "repository_id": "openai/clip-vit-base-patch32",
        "revision": "v2",
        "commit_sha": "deadbeef",
        "pipeline_tag": "image-text-to-text",
        "library_name": "transformers",
        "tags": ["vlm", "multimodal", "vision"],
        "siblings": [
            {"rfilename": "config.json"},
            {"rfilename": "preprocessor_config.json"},
            {"rfilename": "processor_config.json"},
            {"rfilename": "tokenizer.json"},
            {"rfilename": "model.safetensors"},
        ],
        "config": {
            "architectures": ["VisionEncoderDecoderModel"],
            "vision_config": {"image_size": 224, "patch_size": 32},
            "max_position_embeddings": 77,
        },
        "tokenizer": {"model_max_length": 77, "use_fast": True},
        "processor": {"image_processor_type": "CLIPImageProcessor"},
    }


class HuggingFaceAdapterTest(unittest.TestCase):
    def test_normalizes_llm_fixture(self) -> None:
        normalized = normalize_huggingface_repo_metadata(llm_fixture())

        self.assertEqual(normalized.model_profile.family, ModelFamily.LLM)
        self.assertEqual(normalized.model_profile.modality, ModelModality.TEXT)
        self.assertEqual(normalized.model_profile.task, "text-generation")
        self.assertEqual(normalized.artifact_manifest.commit_sha, "8a1b2c3d")
        self.assertTrue(normalized.artifact_manifest.has_tokenizer)
        self.assertIn("transformers", normalized.artifact_manifest.dependency_hints)
        self.assertEqual(normalized.model_profile.artifacts[0].processor_required, False)

    def test_normalizes_embedding_fixture(self) -> None:
        normalized = normalize_huggingface_repo_metadata(embedding_fixture())

        self.assertEqual(normalized.model_profile.family, ModelFamily.EMBEDDING)
        self.assertEqual(normalized.model_profile.modality, ModelModality.TEXT_EMBEDDING)
        self.assertIn("sentence-transformers", normalized.artifact_manifest.dependency_hints)
        self.assertIn("embedding", normalized.model_profile.capabilities)
        self.assertEqual(normalized.model_profile.context_length, 256)

    def test_normalizes_vlm_fixture_with_processor_metadata(self) -> None:
        normalized = normalize_huggingface_repo_metadata(vlm_fixture())

        self.assertEqual(normalized.model_profile.family, ModelFamily.VLM)
        self.assertEqual(normalized.model_profile.modality, ModelModality.IMAGE_TEXT)
        self.assertTrue(normalized.artifact_manifest.has_processor)
        self.assertTrue(normalized.artifact_manifest.has_image_processor)
        self.assertTrue(normalized.model_profile.artifacts[0].processor_required)
        self.assertIn("pillow", normalized.artifact_manifest.dependency_hints)
        self.assertIn("processor-config", normalized.artifact_manifest.dependency_hints)

    def test_requires_repository_id(self) -> None:
        with self.assertRaises(ValueError):
            normalize_huggingface_repo_metadata({})


if __name__ == "__main__":
    unittest.main()
