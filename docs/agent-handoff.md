# Agent Handoff

## Current State

A fully live-connected Azure Databricks model planner with a Rich-based TUI.
No mock data. Credentials in system keyring. Connects to real Databricks workspace
and HuggingFace API. The TUI is inspired by llmfit's ratatui TUI, using
`Rich Live(screen=True)` for full-screen rendering with keyboard navigation.

All V1 foundation work is complete. 80 tests passing. No mock data anywhere.

## Run Locally

```bash
# Tests (all mocked, no live API calls)
uv run pytest tests/

# TUI (default, requires credentials in keyring)
uv run python -m dbx_model_planner app

# Classic text-based fallback
uv run python -m dbx_model_planner app --classic

# Auth management
uv run python -m dbx_model_planner auth login
uv run python -m dbx_model_planner auth status
uv run python -m dbx_model_planner auth logout

# CLI commands (all require live credentials)
uv run python -m dbx_model_planner inventory sync
uv run python -m dbx_model_planner model fit meta-llama/Llama-3.1-8B-Instruct
uv run python -m dbx_model_planner price estimate Standard_NC6s_v3 --vm-rate 3.25 --dbu-rate 0.75
uv run python -m dbx_model_planner deploy plan mistralai/Mistral-7B-Instruct-v0.3
```

Use `--json` on planner commands when another tool or agent needs structured output.

## Credentials

Stored via system keyring (Windows Credential Manager / macOS Keychain / libsecret).

- **Databricks**: host + token, validated against `/api/2.0/current-user`
- **HuggingFace**: token, used for gated model access

Never stored in files, environment variables, or config. The `auth login` wizard
uses `getpass` for masked token input.

## Main Files

### TUI
- `src/dbx_model_planner/tui/app.py` — main loop, key handlers, browse/discover/fetch, history I/O
- `src/dbx_model_planner/tui/state.py` — TuiState, View enum, InputMode, SortColumn, scroll helpers
- `src/dbx_model_planner/tui/views.py` — all rendering (tables, sidebars, loading, header, footer)
- `src/dbx_model_planner/tui/keys.py` — keyboard input (PageUp/PageDown/Home/End, raw mode)

### HuggingFace adapter
- `src/dbx_model_planner/adapters/huggingface/catalog.py` — CatalogEntry, 29 curated models, trending discovery
- `src/dbx_model_planner/adapters/huggingface/normalizer.py` — fetch + normalize HF repo metadata
- `src/dbx_model_planner/adapters/huggingface/models.py` — HuggingFaceRepoMetadata, shared constants

### Auth
- `src/dbx_model_planner/auth/credentials.py` — dataclasses with masked repr
- `src/dbx_model_planner/auth/keyring.py` — save/load/delete/list/exists
- `src/dbx_model_planner/auth/wizard.py` — interactive auth wizard

### Core domain / engines
- `src/dbx_model_planner/domain/` — FitLevel, Cloud, ModelFamily, HostingMode, profiles, recommendations
- `src/dbx_model_planner/engines/fit.py` — memory estimation, fit assessment, candidate ranking
- `src/dbx_model_planner/planners/recommendations.py` — compute recommendation assembly
- `src/dbx_model_planner/planners/deployment.py` — deployment hint generation

### Collectors
- `src/dbx_model_planner/collectors/databricks/inventory.py` — live Databricks API collector

### CLI / Config
- `src/dbx_model_planner/cli.py` — Typer CLI entry point
- `src/dbx_model_planner/terminal_app.py` — classic text-based fallback
- `src/dbx_model_planner/config.py` — AppConfig, TOML + env var overrides

## Design Constraints

- Engines return structured data; rendering is separate.
- Discovered facts are separated from inferred values.
- Conservative heuristics over fake precision.
- No mock data anywhere in the app (tests use mocks).
- Credentials stored via system keyring, never in files.
- Rich Live for TUI (not Textual). Minimal/monochrome theme (cyan accent).

## Live Workspace Facts

- 319 node types (19 GPU, 300 CPU)
- 55 runtimes (31 ML)
- 10 cluster policies
- GPU families: K80, V100, T4, A10, A100_40, A100_80, H100
- GPU memory resolved via `_GPU_MEMORY_MAP` lookup (not from API)

## What Is Next

See `docs/development-backlog.md` for the full V2 feature backlog (F1-F12).

Recommended next features:

1. **F2** (fit filter cycling) — small, fast, high usability value
2. **F1** (plan mode / inverse planning) — the big differentiator
3. **F3** (live Azure pricing) — completes the cost story

## Guardrails

- Keep core planner outputs structured and presentation-neutral.
- Do not add deployment execution, UC registration, or automatic quantization.
- Keep model fit estimates conservative and explicit about assumptions.
- No mock data or fixtures in production code.
