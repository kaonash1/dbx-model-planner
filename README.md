# dbx-model-planner

[![CI](https://github.com/kaonash1/dbx-model-planner/actions/workflows/ci.yml/badge.svg)](https://github.com/kaonash1/dbx-model-planner/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-green)](./LICENSE)

A workspace-aware planner for sizing and costing open models on Azure Databricks. Connects live to your Databricks workspace and HuggingFace, estimates GPU memory requirements, and helps you pick the right compute — with real Azure pricing.

## Features

- **Live workspace inventory** — fetches available node types, GPU specs, and cluster policies directly from your Databricks workspace
- **Model fit analysis** — estimates VRAM requirements for any HuggingFace model using architecture-aware KV cache sizing, and ranks compute by fit level (safe / borderline / unlikely)
- **Azure pricing** — real VM prices from the Azure Retail Prices API, combined with DBU rates, enterprise discounts, and VAT
- **What-if analysis** — explore how quantization and context length affect fit and cost across all GPU nodes
- **Interactive TUI** — full-screen terminal interface with keyboard navigation, model browsing, and a pricing setup wizard

## Getting started

### Prerequisites

- Python 3.11+
- A Databricks workspace with API access
- A HuggingFace account (optional, needed for gated models)

### Install

```bash
pip install dbx-model-planner
```

Or from source:

```bash
git clone https://github.com/kaonash1/dbx-model-planner.git
cd dbx-model-planner
pip install .
```

### Run

```bash
dbx-model-planner
```

On first launch, you'll be prompted for your Databricks workspace URL, API token, and (optionally) a HuggingFace token. Credentials are stored in the system keyring — never written to files.

Use `j`/`k` to navigate, `m` to fit a model, `w` for what-if analysis, `$` for pricing setup, and `q` to quit.

### Credential management

```bash
dbx-model-planner auth login    # Configure credentials
dbx-model-planner auth logout   # Remove stored credentials
dbx-model-planner auth status   # Show credential status
```

### Configuration

Default pricing assumes 37.5% enterprise discount, 19% VAT, and All-Purpose Compute DBU rates. Press `$` in the TUI to change these, or set them in a config file:

```bash
dbx-model-planner --config-path config.toml
```

Press `t` to toggle between All-Purpose and Jobs Compute workload types.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

## License

This project is licensed under the [GNU General Public License v3.0](./LICENSE).

## Acknowledgments

The estimation engine draws on ideas from [llmfit](https://github.com/AlexsJones/llmfit) by Alex Jones — particularly quantization-aware memory modeling, KV cache formulas, and TUI design patterns.

Built with assistance from Claude by [Anthropic](https://anthropic.com).
