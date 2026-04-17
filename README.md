# dbx-model-planner

A workspace-aware planner for sizing and costing open models on Azure Databricks. Connects live to your Databricks workspace and HuggingFace, estimates GPU memory requirements, and helps you pick the right compute — with real Azure pricing.

## Features

- **Live workspace inventory** — fetches available node types, GPU specs, runtimes, and cluster policies directly from your Databricks workspace
- **Model fit analysis** — estimates VRAM requirements for any HuggingFace model and ranks workspace compute by fit level (safe / borderline / unlikely)
- **Architecture-aware estimation** — uses model metadata (KV heads, head dimension, layer count) for precise KV cache sizing, with quantization and context length support
- **Azure pricing integration** — fetches VM prices from the Azure Retail Prices API, applies DBU rates, enterprise discounts, and VAT for realistic $/hr estimates
- **What-if analysis** — explore how quantization (fp16, int8, int4, etc.) and context length affect fit and cost across all GPU nodes
- **Interactive TUI** — full-screen terminal interface built with Rich, with keyboard navigation, model browsing, and a pricing setup wizard
- **Supports LLMs, embeddings, and VLMs** — the planning engine handles text generation, embedding, vision-language, and code models with family-specific heuristics

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

### Credential management

```bash
dbx-model-planner auth login    # Configure credentials
dbx-model-planner auth logout   # Remove stored credentials
dbx-model-planner auth status   # Show credential status
```

## How estimation works

The fit engine estimates total GPU memory as three components:

```
total = model_weights + kv_cache + runtime_overhead
```

**Model weights** are computed from parameter count and quantization. Each quantization level has a bytes-per-parameter (BPP) value — for example, fp16 = 2.0 bytes, Q8_0 = 1.05, Q4_0 = 0.55.

**KV cache** is estimated using model architecture metadata from HuggingFace (number of KV heads, head dimension, layer count) when available. For GQA models like Llama 3, this produces accurate estimates that are much lower than naive parameter-scaled heuristics. When architecture metadata is missing, a parameter-scaled fallback is used.

**Runtime overhead** is a conservative fixed estimate that accounts for Databricks serving framework overhead (1.5 GB for LLMs, 2.0 GB for VLMs, 0.6 GB for embeddings).

A model "fits" a node when the total estimate leaves at least 15% headroom on the node's total GPU memory (per-GPU memory multiplied by GPU count for multi-GPU nodes).

**Pricing** combines the Azure VM list price with Databricks DBU charges, then applies enterprise discount and VAT to the total:

```
hourly_cost = (vm_price + dbu_count * dbu_rate) * (1 - discount) * (1 + vat)
```

## Development

```bash
pip install -e .
pytest tests/ -q
```

## Docs

- [Architecture](./docs/architecture.md)
- [Decision Framework](./docs/decision-framework.md)
- [Domain Model](./docs/domain-model.md)
- [llmfit Analysis](./docs/llmfit-analysis.md)

## License

This project is licensed under the [GNU General Public License v3.0](./LICENSE).

## Acknowledgments

The estimation engine draws on ideas from [llmfit](https://github.com/AlexsJones/llmfit) by Alex Jones — particularly quantization-aware memory modeling, KV cache formulas, and TUI design patterns. See [llmfit Analysis](./docs/llmfit-analysis.md) for a detailed comparison.

Built with assistance from Claude by [Anthropic](https://anthropic.com).
