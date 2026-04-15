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

## MVP focus

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

## Current offline CLI

The current implementation runs without Databricks or Hugging Face credentials by using bundled fixtures and example models.

From the project root:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m dbx_model_planner --help
UV_CACHE_DIR=.uv-cache uv run python -m dbx_model_planner app
UV_CACHE_DIR=.uv-cache uv run python -m dbx_model_planner inventory sync
UV_CACHE_DIR=.uv-cache uv run python -m dbx_model_planner model examples
UV_CACHE_DIR=.uv-cache uv run python -m dbx_model_planner model fit mistral-7b-instruct
UV_CACHE_DIR=.uv-cache uv run python -m dbx_model_planner compute fit Standard_NC6s_v3
UV_CACHE_DIR=.uv-cache uv run python -m dbx_model_planner price estimate Standard_NC6s_v3 --vm-rate 3.25 --dbu-rate 0.75
UV_CACHE_DIR=.uv-cache uv run python -m dbx_model_planner deploy plan qwen2-vl-2b-instruct
```

Most commands also support `--json`.

The `app` command is a small terminal app/wizard for exploring the current offline planner. The other commands remain scriptable.

The CLI uses Typer for command structure and Rich for terminal output. The planner itself stays framework-independent.

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
UV_CACHE_DIR=.uv-cache uv run python -m dbx_model_planner --help
UV_CACHE_DIR=.uv-cache uv run python -m unittest discover -s tests -p 'test_*.py'
```

If the default user data directory is not writable, the CLI falls back to `/tmp/dbx-model-planner` for local snapshots. You can also pass `--data-dir`.
