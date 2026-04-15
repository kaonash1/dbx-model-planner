# HF Hub Prestage Reference

## Recommended sequence

1. `repo_id` and revision preflight
2. Auth and gating check
3. Metadata pull
4. File inventory
5. Config and processor inspection
6. Dry-run download
7. Selective revision-pinned download
8. Cache verification and manifest output

## Important functions and commands

Python:

- `HfApi().model_info(...)`
- `HfApi().repo_info(...)`
- `HfApi().list_repo_files(...)`
- `HfApi().get_paths_info(...)`
- `HfApi().list_repo_tree(..., expand=True)`
- `HfApi().auth_check(...)`
- `hf_hub_download(...)`
- `snapshot_download(...)`

CLI:

- `hf auth login`
- `hf auth whoami`
- `hf download`
- `hf download --dry-run`
- `hf cache ls`
- `hf cache verify`

Transformers:

- `AutoConfig.from_pretrained(...)`
- `AutoProcessor.from_pretrained(...)`

## Operational guardrails

- `model_info(..., expand=...)` cannot be combined with `files_metadata` or `securityStatus`.
- `hf_hub_download` returns cache paths; do not mutate them directly.
- Use `local_dir` for staged, mutable artifacts.
- Pin full commit hashes when planning deterministic staging.
- Gated access often requires browser-side approval before scripts can proceed.
- `login()` accepts personal tokens, not org tokens.
- `list_repo_tree(expand=True)` is useful for file metadata and security-scan metadata, but it is not a full supply-chain attestation.
- For VLMs, inspect processor and preprocessing assets, not just `config.json` and weights.

## Primary sources

- https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api
- https://huggingface.co/docs/huggingface_hub/en/guides/download
- https://huggingface.co/docs/huggingface_hub/en/guides/cli
- https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication
- https://huggingface.co/docs/hub/en/models-gated
- https://huggingface.co/docs/hub/model-cards
- https://huggingface.co/docs/transformers/en/processors
- https://huggingface.co/docs/transformers/en/main_classes/configuration
- https://huggingface.co/docs/transformers/model_doc/auto
- https://huggingface.co/docs/safetensors/en/metadata_parsing
