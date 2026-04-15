---
name: hf-hub-prestage-planner
description: Use when inspecting a Hugging Face model repository before staging or controlled download, especially for repo and revision preflight, auth and gating checks, metadata inspection, file inventory, VLM processor discovery, safetensors-aware planning, selective downloads, and staged artifact verification.
---

# HF Hub Prestage Planner

Use this skill when the task is about understanding a Hugging Face model repo before downloading or staging it.

This is not a generic "download the model" skill. Its purpose is pre-staging inspection and controlled acquisition.

## Outputs

Produce one of these:

- a repo preflight summary with revision, SHA, gating, and artifact facts
- a file and size manifest for staged download planning
- a selective download plan with pinned revision and filters
- a list of missing processor, tokenizer, or config assets

## Core workflows

1. Resolve `repo_id`, revision, and commit SHA.
2. Check auth, privacy, and gating before download.
3. Inspect metadata with the Hub API.
4. Inventory files and sizes before acquisition.
5. Confirm tokenizer, config, and processor assets, especially for VLMs.
6. Prefer safetensors-aware planning over opaque checkpoint handling.
7. Use selective, revision-pinned downloads when acquisition is necessary.
8. Generate a staging manifest for later Databricks deployment steps.

## APIs and commands to know

Python:

- `HfApi`
- `model_info`
- `repo_info`
- `list_repo_files`
- `file_exists`
- `get_paths_info`
- `list_repo_tree`
- `auth_check`
- `hf_hub_download`
- `snapshot_download`
- `hf_hub_url`

CLI:

- `hf auth login`
- `hf auth whoami`
- `hf download`
- `hf download --dry-run`
- `hf cache ls`
- `hf cache verify`
- `hf env`

Transformers:

- `AutoConfig.from_pretrained`
- `AutoProcessor.from_pretrained`

## Guardrails

- Do not default to full model downloads when metadata inspection is enough.
- Do not stage mutable artifacts out of cache paths without using `local_dir`.
- Do not assume VLM readiness from weights alone; processor files matter.
- Do not skip revision pinning when planning a deployment artifact path.
- Do not assume gated access can be resolved programmatically before the user has access.
- Do not use this skill to decide Databricks hosting mode or Azure cost; it is a repo-inspection skill.

## When to read references

- Read `references/metadata-and-downloads.md` when you need the exact Hub functions, CLI commands, or source links.
