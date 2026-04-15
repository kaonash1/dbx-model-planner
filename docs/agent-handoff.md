# Agent Handoff

## Current State

The project is a lean, `llmfit`-style planner for Azure Databricks:

1. Load Databricks compute/runtime/policy inventory.
2. Normalize model metadata from Hugging Face-like metadata.
3. Estimate model-to-compute fit and compute-to-model fit.
4. Estimate cost with VM rate, configurable DBU rate, 37% discount, and 19% VAT by default.
5. Generate a simple deployment hint only.

It is not a deployment orchestrator yet. Keep it that way until the live inventory and HF metadata paths are validated.

## Run Locally

From `/home/luka/Workspace/dbx-model-planner`:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=src python3 -m dbx_model_planner inventory sync
PYTHONPATH=src python3 -m dbx_model_planner model examples
PYTHONPATH=src python3 -m dbx_model_planner model fit mistral-7b-instruct
PYTHONPATH=src python3 -m dbx_model_planner compute fit Standard_NC6s_v3
PYTHONPATH=src python3 -m dbx_model_planner price estimate Standard_NC6s_v3 --vm-rate 3.25 --dbu-rate 0.75
PYTHONPATH=src python3 -m dbx_model_planner deploy plan qwen2-vl-2b-instruct
```

Use `--json` on planner commands when another tool or agent needs structured output.

## Main Files

- `src/dbx_model_planner/cli.py`: thin stdlib CLI.
- `src/dbx_model_planner/config.py`: local TOML config plus environment overrides.
- `src/dbx_model_planner/collectors/databricks/inventory.py`: offline Databricks inventory collector shape.
- `src/dbx_model_planner/adapters/huggingface/normalizer.py`: HF metadata to `ModelProfile`.
- `src/dbx_model_planner/adapters/azure`: Azure SKU mapping and Retail Prices API helpers.
- `src/dbx_model_planner/engines/fit.py`: conservative fit heuristics.
- `src/dbx_model_planner/planners/recommendations.py`: model-first and compute-first planner assembly.
- `src/dbx_model_planner/planners/deployment.py`: non-executing deployment hint.
- `src/dbx_model_planner/catalog/examples.py`: bundled offline model examples.

## Project Skills

Use the repo-local skills when continuing live integrations:

- `.codex/skills/databricks-cli-planner/SKILL.md`
- `.codex/skills/hf-hub-prestage-planner/SKILL.md`
- `.codex/skills/azure-dbx-capacity-pricing/SKILL.md`

## Credentials Tomorrow

Do not put secrets in config files or fixtures.

Expected environment variables:

```bash
export DATABRICKS_HOST="https://..."
export DATABRICKS_TOKEN="..."
export HF_TOKEN="..."
```

Optional planner config overrides:

```bash
export DBX_MODEL_PLANNER_DATABRICKS_DBU_HOURLY_RATE="0.75"
export DBX_MODEL_PLANNER_PRICING_DISCOUNT_RATE="0.37"
export DBX_MODEL_PLANNER_PRICING_VAT_RATE="0.19"
export DBX_MODEL_PLANNER_PRICING_CURRENCY_CODE="EUR"
```

## Next Implementation Tickets

1. Databricks live inventory integration:
   Implement live mode behind `DatabricksInventoryCollector` without changing planner contracts. Start with node types, DBR versions, cluster policies, and instance pools.

2. Hugging Face live metadata integration:
   Add a live adapter that fetches repo metadata, config, tokenizer, processor metadata, siblings, gating status, and revision/commit SHA. Return the existing `HuggingFaceNormalizedModel`.

3. CLI wiring for live mode:
   Add flags such as `--live` or `--source live` rather than replacing fixture mode. Offline mode must remain default until live behavior is stable.

4. Real validation:
   Run one real Databricks inventory sync and one real model fit. Save any mismatches as fixtures before changing heuristics.

## Guardrails

- Keep core planner outputs structured and presentation-neutral.
- Do not add deployment execution, UC registration, or automatic quantization yet.
- Do not make the CLI depend on rich/TUI libraries.
- Keep model fit estimates conservative and explicit about assumptions.
- Prefer adding small fixtures for new real-world cases before changing heuristics.
