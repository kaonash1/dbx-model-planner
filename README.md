# dbx-model-planner

`dbx-model-planner` is a workspace-aware planner for running open and hosted models on Azure Databricks.

The project is intentionally broader than LLMs. The target scope includes:

- text generation models
- embedding models
- rerankers
- vision-language models

The core question is the same across all of them:

- what model do I have
- what compute do I have
- what runtime and configuration do I need
- what will it cost
- what is the shortest safe path to running it on Databricks

## How it works

The planner connects **live** to your Databricks workspace and HuggingFace to:

1. Fetch available node types, runtimes, and cluster policies from your workspace
2. Fetch model metadata (parameter count, architecture, context length) from HuggingFace
3. Estimate GPU memory requirements and rank candidate compute profiles
4. Generate deployment hints with Unity Catalog volume paths and cluster config

Credentials are stored securely in the system keyring (Windows Credential Manager, macOS Keychain, or libsecret on Linux).

## Getting started

### First-time setup

```bash
uv run python -m dbx_model_planner auth login
```

This will prompt for:
- **Databricks workspace URL** (e.g., `https://adb-1234567890123456.7.azuredatabricks.net`)
- **Databricks API token** (generate at: Workspace > Settings > Developer > Access tokens)
- **HuggingFace token** (optional, needed for gated models like Llama, Mistral)

Credentials are validated on entry and stored in the system keyring.

### Interactive terminal planner

```bash
uv run python -m dbx_model_planner app
```

The `app` command syncs live inventory from Databricks, then presents an interactive menu:
1. Show workspace inventory (node types, runtimes, policies)
2. Model -> compute fit (enter a HuggingFace model ID, get ranked compute candidates)
3. Deployment hint (UC volume path, cluster config suggestion)

### CLI commands

```bash
# Auth
uv run python -m dbx_model_planner auth login      # Configure credentials
uv run python -m dbx_model_planner auth logout     # Remove stored credentials
uv run python -m dbx_model_planner auth status     # Show credential status

# Inventory
uv run python -m dbx_model_planner inventory sync  # Sync and cache workspace inventory

# Model planning
uv run python -m dbx_model_planner model fit meta-llama/Llama-3.1-8B-Instruct
uv run python -m dbx_model_planner model fit mistralai/Mistral-7B-Instruct-v0.3 --batch
uv run python -m dbx_model_planner model fit meta-llama/Llama-3.1-8B-Instruct --azure-pricing

# Pricing
uv run python -m dbx_model_planner price estimate Standard_NC6s_v3 --vm-rate 3.25 --dbu-rate 0.75
uv run python -m dbx_model_planner price estimate Standard_NC6s_v3 --azure-pricing

# Deployment hints
uv run python -m dbx_model_planner deploy plan meta-llama/Llama-3.1-8B-Instruct
```

Most commands support `--json` for machine-readable output.

## Scope

The first version is a planner and advisor, not a full orchestrator.

It should answer:

- which workspace GPU computes are available right now
- which DBR versions and policies constrain the choice
- whether a given model is likely to fit on a given compute
- which precision or quantization is realistic
- what the estimated cost is after company pricing rules
- what the recommended deployment path is

It should not initially promise:

- perfect memory prediction
- automatic support for every quantization stack
- one-click production deployment for every model family

## Why the scope can include LLMs, embeddings, and VLMs

These families differ in serving patterns, but they can still share the same planning model:

- model profile: task, modality, parameter count, artifact size, context constraints
- runtime profile: framework, dtype, quantization, dependencies
- compute profile: GPU count, GPU memory, CPU memory, local disk, DBR compatibility
- cost profile: VM rate, DBU rate, company adjustments

The planning engine then applies family-specific heuristics where needed.

## Docs

- [MVP Spec](./docs/mvp-spec.md)
- [Decision Framework](./docs/decision-framework.md)
- [llmfit Analysis](./docs/llmfit-analysis.md)
- [Domain Model](./docs/domain-model.md)
- [Development Backlog](./docs/development-backlog.md)
- [Architecture](./docs/architecture.md)
- [Agent Handoff](./docs/agent-handoff.md)

## Local development

```bash
uv run python -m dbx_model_planner --help
uv run python -m unittest discover -s tests -p 'test_*.py'
```

If the default user data directory is not writable, the CLI falls back to a temp directory for local snapshots. You can also pass `--data-dir`.
