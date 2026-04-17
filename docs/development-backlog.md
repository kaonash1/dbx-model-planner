# Development Backlog

## Product boundary

A planner for Azure Databricks that answers:

1. What Databricks compute and runtimes do I have?
2. Given a model, what compute can run it?
3. Given a compute, what models are realistic?
4. What does it cost under our internal pricing formula?
5. What is the simplest next-step deployment hint?
6. What hosting mode should I use?
7. What hardware would I need for a specific model configuration?

It should not try to be a full platform orchestrator.

## Current state

All V1 foundation tickets (L1-L14) are complete.
The project is fully live-connected with a Rich-based TUI.

### What is built and working

- **Auth**: System keyring credential storage (Databricks + HuggingFace tokens)
- **Live inventory**: Databricks API collector (node types, runtimes, policies)
- **Live model metadata**: HuggingFace API adapter with normalization
- **Fit engine**: Memory estimation, model-to-compute and compute-to-model fit
- **Cost engine**: VM + DBU + company discount + VAT calculation
- **Recommendation planner**: Ranked candidates with fit levels and notes
- **Deployment hints**: UC volume path, runtime, node type suggestions
- **TUI**: Full-screen Rich Live app with:
  - Inventory view with split-pane node detail (GPU/CPU nodes, runtimes, policies)
  - Model browser (29 curated models + HF trending discovery via `D` key)
  - Model fit view with split-pane candidate detail
  - What-if view (`w` key) with quant/context selectors
  - Sort column cycling (`s`) with direction toggle (`S`)
  - Search (`/`) across all views
  - CPU node toggle (`c` / `Tab`)
  - Quick fit from node (`f`), model ID input (`m`), browse (`b`)
  - Deployment hint generation (`d`)
  - Pricing setup wizard (`$` key) -- 4-step: region, discount, VAT, DBU rate
  - Workload type toggle (`t` key) -- All-Purpose Compute / Jobs Compute
  - Fit filter cycling (`f` in fit view)
  - PageUp/PageDown, Home/End, vim-style navigation
  - Model history persistence (last 10)
  - Loading spinner with threaded HF API fetch
  - Minimal/monochrome theme (cyan accent, grey borders)
- **CLI**: Typer-based with auth login/logout/status, `--classic` fallback
- **Tests**: 163 tests, all mocked (no live API calls), passing

### Live workspace facts (from real Databricks workspace)

- 319 node types (19 GPU, 300 CPU)
- 55 runtimes (31 ML)
- 10 cluster policies
- GPU families: K80, V100, T4, A10, A100_40, A100_80, H100

## Design constraints

- Engines return structured data; rendering is separate.
- Discovered facts are separated from inferred values.
- Conservative heuristics over fake precision.
- No mock data anywhere in the app.
- Credentials stored via system keyring, never in files.

---

## V2 feature backlog

Features are organized by tier (impact and alignment with product goals).
Status: `planned`, `in-progress`, `completed`.

### Tier 1 -- High impact, core product value

#### F1: Plan mode (inverse planning)

- Status: `completed`
- Priority: high
- Inspiration: llmfit `p` key / `plan.rs`
- Goal:
  Answer "What hardware do I need for this model?" instead of
  "Does this model fit on this hardware?"
- Scope:
  - New `View.PLAN` in TUI, accessible via `p` key from model fit view
  - Editable fields: quantization (fp16/bf16/int8/int4), context length
    (preset cycle), GPU count (1/2/4/8) -- Tab cycles, h/l adjusts
  - Output: minimum VRAM, recommended node type, feasible run paths
  - Quantization comparison table with upgrade deltas and feasible node counts
  - Run paths table: all quant+node combos that fit, sorted safe-first
  - Split-pane layout: parameters+tables on left, recommendation+detail on right
  - 21 unit tests for plan engine
- Why it matters:
  - Directly answers capacity planning questions
  - llmfit's most distinctive feature
  - Maps to decision-framework dimension 3 (compute fit)

#### F2: Fit filter cycling

- Status: `completed`
- Priority: high
- Inspiration: llmfit `f` key
- Goal:
  Quick filter on model fit results by fit level.
- Scope:
  - `f` key in model fit view cycles: All -> Safe -> Borderline -> Unlikely
  - Filter indicator in footer and table title
  - Persists until changed or view is exited
  - Empty state shown when no candidates match current filter
- Why it matters:
  - When you have 19 GPU nodes, quickly narrowing to "safe only" is essential
  - Small effort, big usability win

#### F3: Live Azure pricing

- Status: `completed`
- Priority: high
- Goal:
  Fetch real VM prices from the Azure Retail Prices API and
  flow them through the cost engine into the TUI.
- Scope:
  - Azure Retail Prices API client (`adapters/azure/pricing.py`) -- existed
  - SKU mapping (`adapters/azure/sku.py`) -- existed
  - Cost engine (`engines/cost.py`) -- existed
  - Price cache with TTL (`adapters/azure/price_cache.py`) -- new
  - Config: `azure_region`, `price_cache_ttl_seconds`, `auto_fetch_pricing`
  - TUI state: `vm_pricing`, `pricing_loaded`, `pricing_region`
  - Background price fetch on TUI startup (auto if enabled)
  - `$` key to manually refresh prices from any view
  - $/hr column in inventory table and fit table
  - Pricing section in node sidebar (inventory view)
  - Cost passed to `recommend_compute_for_model()` for candidate enrichment
  - Cost displayed in candidate sidebar (fit view) -- already existed
  - File-based cache persistence (JSON in XDG data home)
  - 18 tests for price cache, serialization, bulk fetch
- Why it matters:
  - Cost is one of the main reasons to build this planner
  - Completes the cost story end-to-end
  - Maps to decision-framework dimension 7 (cost)

#### F4: Hosting mode recommendations

- Status: `skipped`
- Priority: high (deferred -- premature without data sources)
- Goal:
  Recommend whether a model should use Foundation Model APIs,
  external model endpoints, custom serving on GPU, or batch compute.
- Scope:
  - Hosting mode selector engine (`engines/hosting.py`)
  - Decision logic based on: model availability in FMAPI catalog,
    parameter count, workload shape (online vs batch), cost comparison
  - Show hosting mode recommendation in model fit view header
  - Explain why a mode was chosen or rejected
- Why skipped:
  - FMAPI catalog would be a stale hardcoded list (no API to query it)
  - External Models isn't applicable for HuggingFace models
  - Batch/Classic modes need workload inputs we don't collect
  - Revisit when better data sources are available
- Why it matters:
  - This is the #1 differentiator from llmfit
  - Maps to decision-framework dimension 1 (hosting mode)
  - Before picking a GPU SKU, users need to know if self-hosting
    is even the right approach

### Tier 2 -- Good UX improvements

#### F5: Model comparison view

- Status: `planned`
- Priority: medium
- Inspiration: llmfit `m`/`c` keys, visual mode
- Goal:
  Mark 2-3 models and compare them side by side.
- Scope:
  - `m` key in browse view marks a model for comparison
  - `c` key opens comparison view (table: rows=attributes, cols=models)
  - Attributes: params, family, memory estimate, best fit node, cost
  - Highlight best values per attribute
- Why it matters:
  - Common workflow: "should I use Llama-3.1-8B or Mistral-7B?"
  - Reduces back-and-forth between individual fit analyses

#### F6: Hardware simulation / what-if

- Status: `planned`
- Priority: medium
- Inspiration: llmfit `S` key / simulation popup
- Goal:
  Override GPU memory for a node to simulate different hardware.
- Scope:
  - `W` key opens what-if popup in inventory view
  - Override fields: GPU memory, GPU count, vCPU, RAM
  - All fit calculations re-run against simulated specs
  - `SIM` indicator in header when active
  - Reset to real hardware with a key
- Why it matters:
  - Useful for capacity planning conversations
  - "What if we upgraded to A100s?"

#### F7: Multi-dimensional scoring

- Status: `planned`
- Priority: medium
- Inspiration: llmfit quality/speed/fit/context scores (0-100)
- Goal:
  Add composite scoring to model fit candidates.
- Scope:
  - Score dimensions: fit (memory headroom), cost efficiency,
    runtime compatibility, GPU utilization
  - Weighted composite score (0-100)
  - Score column in fit table, breakdown in sidebar
- Why it matters:
  - Helps rank candidates beyond just "safe/borderline/unlikely"
  - Makes recommendations more nuanced

#### F8: Speed estimation

- Status: `planned`
- Priority: medium
- Inspiration: llmfit bandwidth-based tok/s estimation
- Goal:
  Estimate token generation speed per candidate.
- Scope:
  - GPU memory bandwidth lookup table (by GPU family)
  - Formula: `(bandwidth_GB_s / model_size_GB) * efficiency_factor`
  - Show tok/s estimate in fit table and candidate sidebar
  - Note that estimates are approximate
- Why it matters:
  - Adds credibility to recommendations
  - Helps distinguish between "fits but slow" and "fits and fast"

### Tier 3 -- Infrastructure and polish

#### F9: Docs cleanup

- Status: `planned`
- Priority: medium
- Goal:
  Update all docs to reflect the current reality.
- Scope:
  - Update or remove stale agent-handoff.md
  - Update architecture.md with actual package layout
  - Update mvp-spec.md to include TUI
  - Keep decision-framework.md and domain-model.md (still valid)
  - Keep llmfit-analysis.md (still relevant reference)

#### F10: REST API mode

- Status: `planned`
- Priority: low
- Inspiration: llmfit `serve` command
- Goal:
  `dbx-model-planner serve` starts an HTTP API.
- Scope:
  - Endpoints: /health, /inventory, /fit/{model_id}, /plan/{model_id}
  - JSON responses using existing domain objects
  - Useful for integration with other tools or dashboards

#### F11: Export / report generation

- Status: `planned`
- Priority: low
- Goal:
  Generate a markdown or JSON report of a planning session.
- Scope:
  - Export current model fit results as markdown table
  - Include inventory summary, fit candidates, cost, deployment hint
  - Save to file or clipboard

#### F12: Workspace region detection

- Status: `planned`
- Priority: low
- Goal:
  Detect the workspace region from the Databricks API or URL.
- Scope:
  - Infer region from workspace URL or `/api/2.0/workspace/get-status`
  - Populate `region` field on inventory snapshot
  - Use region for Azure pricing API queries

---

## Completed V1 tickets (historical)

All original L1-L14 tickets from the initial build sprint are complete.
They covered: config, storage, inventory collector, HF adapter, pricing enricher,
fit engine, cost engine, recommendation planner, deployment hints, CLI, and tests.

The implementation diverged from the original plan in these ways:
- Credentials use system keyring instead of environment variables
- No mock mode or fixtures in production code (tests use mocks)
- TUI was built with Rich Live instead of being out of scope
- Model browser with curated catalog was added
- No `--live` flag; everything is live by default

## Suggested feature order

### Next sprint (pick 2-3)

Recommended starting point:

1. **F2** (fit filter) -- small, high-value, fast to build
2. **F1** (plan mode) -- the big differentiator feature
3. **F3** (live Azure pricing) -- completes the cost story

### After that

4. **F4** (hosting mode) -- requires research into FMAPI catalog
5. **F5** (model comparison) -- good UX improvement
6. **F6** (hardware simulation) -- capacity planning tool

### Later

7. **F7** (scoring), **F8** (speed estimation)
8. **F9** (docs), **F10** (REST API), **F11** (export), **F12** (region)
