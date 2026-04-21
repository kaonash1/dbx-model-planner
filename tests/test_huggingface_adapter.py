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


class SafetensorsFallbackTest(unittest.TestCase):
    """Test that parameter_count falls back to safetensors data."""

    def test_safetensors_total_used_when_config_has_no_params(self) -> None:
        """When config lacks num_parameters but safetensors.total is present."""
        data = {
            "repository_id": "test/safetensors-total",
            "pipeline_tag": "text-generation",
            "tags": ["llm"],
            "siblings": [{"rfilename": "model.safetensors"}],
            "config": {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                # No num_parameters here
                "max_position_embeddings": 4096,
            },
            "safetensors": {
                "total": 7000000000,
                "parameters": {"F16": 7000000000},
            },
        }
        normalized = normalize_huggingface_repo_metadata(data)
        self.assertEqual(normalized.model_profile.parameter_count, 7000000000)

    def test_safetensors_parameters_sum_when_total_missing(self) -> None:
        """When safetensors has no 'total' but has per-dtype parameters dict."""
        data = {
            "repository_id": "test/safetensors-sum",
            "pipeline_tag": "text-generation",
            "tags": ["llm"],
            "siblings": [{"rfilename": "model.safetensors"}],
            "config": {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "max_position_embeddings": 4096,
            },
            "safetensors": {
                "parameters": {"F16": 3000000000, "BF16": 4000000000},
            },
        }
        normalized = normalize_huggingface_repo_metadata(data)
        self.assertEqual(normalized.model_profile.parameter_count, 7000000000)

    def test_config_num_parameters_takes_precedence(self) -> None:
        """Config-level num_parameters should be used even if safetensors exists."""
        data = {
            "repository_id": "test/config-precedence",
            "pipeline_tag": "text-generation",
            "tags": ["llm"],
            "siblings": [{"rfilename": "model.safetensors"}],
            "config": {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_parameters": 8000000000,
                "max_position_embeddings": 4096,
            },
            "safetensors": {
                "total": 9999999999,  # This should NOT be used
            },
        }
        normalized = normalize_huggingface_repo_metadata(data)
        self.assertEqual(normalized.model_profile.parameter_count, 8000000000)

    def test_no_param_count_when_both_missing(self) -> None:
        """When neither config nor safetensors has parameter info."""
        data = {
            "repository_id": "test/no-params",
            "pipeline_tag": "text-generation",
            "tags": ["llm"],
            "siblings": [{"rfilename": "model.safetensors"}],
            "config": {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "max_position_embeddings": 4096,
            },
        }
        normalized = normalize_huggingface_repo_metadata(data)
        self.assertIsNone(normalized.model_profile.parameter_count)


class InferredQuantizationTest(unittest.TestCase):
    """Test that quantization options are inferred when not explicitly tagged."""

    def test_bf16_model_gets_bf16_fp16_int8_int4(self) -> None:
        """BFloat16 model should get bf16, fp16, int8, int4 options."""
        data = {
            "repository_id": "test/bf16-model",
            "pipeline_tag": "text-generation",
            "tags": ["llm"],
            "siblings": [{"rfilename": "model.safetensors"}],
            "config": {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "max_position_embeddings": 4096,
                "torch_dtype": "bfloat16",
            },
            "safetensors": {"total": 7000000000},
        }
        normalized = normalize_huggingface_repo_metadata(data)
        self.assertEqual(
            normalized.model_profile.quantization_options,
            ["bf16", "fp16", "int8", "int4"],
        )

    def test_fp16_model_gets_fp16_int8_int4(self) -> None:
        """FP16 model without bf16 should get fp16, int8, int4."""
        data = {
            "repository_id": "test/fp16-model",
            "pipeline_tag": "text-generation",
            "tags": ["llm"],
            "siblings": [{"rfilename": "model.safetensors"}],
            "config": {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "max_position_embeddings": 4096,
                "torch_dtype": "float16",
            },
            "safetensors": {"total": 7000000000},
        }
        normalized = normalize_huggingface_repo_metadata(data)
        self.assertEqual(
            normalized.model_profile.quantization_options,
            ["fp16", "int8", "int4"],
        )

    def test_explicit_quant_tags_preserved(self) -> None:
        """When explicit quant tags exist, they should be kept (not overridden)."""
        data = {
            "repository_id": "test/explicit-quant",
            "pipeline_tag": "text-generation",
            "tags": ["llm", "gptq", "int4"],
            "siblings": [{"rfilename": "model.safetensors"}],
            "config": {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_parameters": 7000000000,
                "max_position_embeddings": 4096,
            },
        }
        normalized = normalize_huggingface_repo_metadata(data)
        # Explicit tags found, so the inferred defaults should NOT be used
        self.assertIn("gptq", normalized.model_profile.quantization_options)
        self.assertIn("int4", normalized.model_profile.quantization_options)


class VlmVisionConfigExtractionTest(unittest.TestCase):
    """Test VLM-specific vision_config and text_config parsing."""

    def test_extracts_vision_params_from_vision_config(self) -> None:
        """vision_parameter_count should be estimated from vision_config."""
        data = {
            "repository_id": "test/vlm-with-vision-config",
            "pipeline_tag": "image-text-to-text",
            "tags": ["vlm", "multimodal"],
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "preprocessor_config.json"},
                {"rfilename": "model.safetensors"},
            ],
            "config": {
                "architectures": ["LlavaForConditionalGeneration"],
                "vision_config": {
                    "hidden_size": 1024,
                    "intermediate_size": 4096,
                    "num_hidden_layers": 24,
                },
                "text_config": {
                    "num_hidden_layers": 32,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "hidden_size": 4096,
                },
                "num_parameters": 8000000000,
                "max_position_embeddings": 8192,
            },
            "processor": {"image_processor_type": "CLIPImageProcessor"},
        }
        normalized = normalize_huggingface_repo_metadata(data)
        profile = normalized.model_profile

        self.assertEqual(profile.family, ModelFamily.VLM)
        self.assertIsNotNone(profile.vision_parameter_count)
        self.assertGreater(profile.vision_parameter_count, 0)

        # Architecture details should come from text_config, not top-level
        self.assertEqual(profile.num_hidden_layers, 32)
        self.assertEqual(profile.num_kv_heads, 8)
        self.assertEqual(profile.head_dim, 128)  # 4096 / 32

    def test_text_config_overrides_top_level_for_vlm(self) -> None:
        """For VLMs, architecture details from text_config take precedence."""
        data = {
            "repository_id": "test/vlm-nested-config",
            "pipeline_tag": "image-text-to-text",
            "tags": ["vlm"],
            "siblings": [{"rfilename": "model.safetensors"}],
            "config": {
                "architectures": ["LlavaForConditionalGeneration"],
                "num_hidden_layers": 24,  # Vision layers at top level
                "vision_config": {"hidden_size": 1024, "num_hidden_layers": 24},
                "text_config": {
                    "num_hidden_layers": 32,  # Text layers in sub-config
                    "num_attention_heads": 32,
                    "hidden_size": 4096,
                },
            },
            "processor": {"image_processor_type": "CLIPImageProcessor"},
        }
        normalized = normalize_huggingface_repo_metadata(data)
        # Should use text_config.num_hidden_layers (32), not top-level (24)
        self.assertEqual(normalized.model_profile.num_hidden_layers, 32)

    def test_no_vision_config_returns_none(self) -> None:
        """VLM without vision_config should have no vision_parameter_count."""
        data = vlm_fixture()
        # The fixture has a minimal vision_config without enough info to estimate
        # (no hidden_size or num_hidden_layers for estimation)
        data["config"] = {  # type: ignore[index]
            "architectures": ["VisionEncoderDecoderModel"],
            "max_position_embeddings": 77,
        }
        normalized = normalize_huggingface_repo_metadata(data)
        self.assertIsNone(normalized.model_profile.vision_parameter_count)

    def test_llm_config_key_used_for_arch(self) -> None:
        """Some VLMs use 'llm_config' instead of 'text_config'."""
        data = {
            "repository_id": "test/vlm-llm-config",
            "pipeline_tag": "image-text-to-text",
            "tags": ["vlm"],
            "siblings": [{"rfilename": "model.safetensors"}],
            "config": {
                "architectures": ["InternVLChatModel"],
                "vision_config": {"hidden_size": 1024, "num_hidden_layers": 24},
                "llm_config": {
                    "num_hidden_layers": 40,
                    "num_attention_heads": 40,
                    "num_key_value_heads": 8,
                    "hidden_size": 5120,
                },
            },
            "processor": {"image_processor_type": "CLIPImageProcessor"},
        }
        normalized = normalize_huggingface_repo_metadata(data)
        # Should pick up from llm_config
        self.assertEqual(normalized.model_profile.num_hidden_layers, 40)
        self.assertEqual(normalized.model_profile.num_kv_heads, 8)
        self.assertEqual(normalized.model_profile.head_dim, 128)  # 5120 / 40


if __name__ == "__main__":
    unittest.main()
