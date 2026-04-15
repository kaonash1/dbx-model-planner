# Development Backlog

## Product boundary

The correct MVP is close to the shape of `llmfit`, but for Azure Databricks instead of a local machine.

It should answer:

1. what Databricks compute and DBRs do I have
2. given a model, what compute can run it
3. given a compute, what models are realistic
4. what does it cost under our internal pricing formula
5. what is the simplest next-step deployment hint

It should not try to be a full platform orchestrator.

## V1 scope

### In scope

- Databricks inventory:
  node types, DBRs, policies, pools
- model metadata normalization from Hugging Face
- model-to-compute fit
- compute-to-model fit
- Azure VM pricing plus configurable DBU inputs
- company discount and VAT calculation
- simple deployment hint:
  UC volume path suggestion, cluster hint, starter script skeleton
- CLI plus JSON output
- offline fixtures and tests

### Out of scope

- full serving endpoint management
- hosted vs external vs custom-serving strategy engine
- automatic artifact upload
- automatic quantization execution
- Unity Catalog registration automation
- high-fidelity throughput benchmarking
- rich TUI or interactive UX
- advanced observability and billing analytics

## Design constraints

- Keep the output layer swappable.
- Engines must return structured data first.
- CLI rendering must stay thin.
- Separate discovered facts from inferred values.
- Prefer conservative heuristics over complicated fake precision.

## Ticket statuses

- `ready-now`: can start without Databricks or HF credentials
- `blocked-by-keys`: should wait until tomorrow

## Lean tickets

### L1: Config And Local Runtime Context

- Status: `completed`
- Goal:
  Add local config loading for:
  discount, VAT, DBU inputs, preferred regions, blocked SKUs, profile names.
- Deliverables:
  - config loader
  - local config template
  - runtime context object
- Acceptance criteria:
  - config loads from file plus env overrides
  - secrets are referenced, not embedded

### L2: Local Snapshot Store

- Status: `completed`
- Goal:
  Persist inventory snapshots and normalized model metadata locally.
- Deliverables:
  - SQLite-backed storage
  - schema bootstrap
  - save/load helpers
- Acceptance criteria:
  - inventory and model snapshots round-trip cleanly

### L3: Databricks Inventory Collector Skeleton

- Status: `completed`
- Goal:
  Build the collector shape for:
  node types, DBR versions, policies, pools.
- Deliverables:
  - collector interface
  - mock mode
  - fixture-driven parsing
- Acceptance criteria:
  - collector produces `WorkspaceInventorySnapshot`
  - works offline with fixtures

### L4: Databricks Live Inventory Integration

- Status: `blocked-by-keys`
- Goal:
  Wire the real Databricks CLI or API calls for:
  `list-node-types`, `spark-versions`, policies, pools.
- Deliverables:
  - profile-aware execution
  - parsing and normalization
- Acceptance criteria:
  - real workspace inventory succeeds tomorrow with credentials

### L5: Hugging Face Model Adapter Skeleton

- Status: `completed`
- Goal:
  Normalize HF repo metadata into `ModelProfile`.
- Deliverables:
  - adapter interface
  - mocked repo fixtures
  - artifact manifest shape
  - VLM processor-aware checks
- Acceptance criteria:
  - can build `ModelProfile` from fixture data for LLM, embedding, and VLM examples

### L6: Hugging Face Live Metadata Integration

- Status: `blocked-by-keys`
- Goal:
  Wire live HF Hub inspection for public and gated repos.
- Deliverables:
  - repo preflight
  - auth and gating checks
  - revision pinning
  - selective dry-run planning
- Acceptance criteria:
  - public repos work
  - gated/private failures are explicit

### L7: Azure SKU And Pricing Enricher

- Status: `completed`
- Goal:
  Add Azure-side SKU facts and VM pricing inputs.
- Deliverables:
  - node-type to Azure-SKU mapping helpers
  - region and restriction normalization
  - Retail Prices API fetcher
  - discount and VAT application
- Acceptance criteria:
  - returns VM list price, discounted price, VAT-adjusted price
  - DBU stays configurable input

### L8: Fit Engine V1

- Status: `completed`
- Goal:
  Implement conservative heuristics for:
  model -> compute fit and compute -> model fit.
- Deliverables:
  - memory estimate rules
  - fit labels:
    `safe`, `borderline`, `unlikely`
  - family-aware branches for:
    `llm`, `embedding`, `vlm`
- Acceptance criteria:
  - structured candidate assessments
  - assumptions and notes included
  - embeddings and VLMs are not forced through LLM-only logic

### L9: Cost Engine V1

- Status: `completed`
- Goal:
  Compute hourly cost from:
  Azure VM price + configured DBU + company adjustments.
- Deliverables:
  - cost composition logic
  - output fields for raw, discounted, VAT-adjusted values
- Acceptance criteria:
  - pricing output is separate from fit output
  - no hidden formula assumptions

### L10: Recommendation Planner

- Status: `completed`
- Goal:
  Assemble the main answer from inventory, model metadata, fit, and cost.
- Deliverables:
  - ranked candidate list
  - top recommendation
  - assumptions and blockers
- Acceptance criteria:
  - presentation-neutral structured output

### L11: Deploy Hint Generator

- Status: `completed`
- Goal:
  Generate a minimal next-step deployment hint.
- Deliverables:
  - UC volume path suggestion
  - cluster hint
  - dependency notes
  - starter script skeleton
- Acceptance criteria:
  - no real execution
  - no orchestration logic

### L12: CLI Surface V1

- Status: `completed`
- Goal:
  Replace placeholder commands with a lean CLI surface:

  - `inventory sync`
  - `model fit`
  - `compute fit`
  - `cost estimate`
  - `deploy hint`

- Deliverables:
  - command handlers
  - text renderer
  - JSON output mode
- Acceptance criteria:
  - CLI calls planners, not raw integrations
  - renderers are isolated from engines

### L13: Offline Fixtures And End-To-End Tests

- Status: `completed`
- Goal:
  Build enough local fixtures to validate the core planner without credentials.
- Deliverables:
  - Databricks inventory fixtures
  - HF repo fixtures
  - Azure pricing fixtures
  - end-to-end planner tests
- Acceptance criteria:
  - core recommendation flow works offline

### L14: Tomorrow Morning Validation

- Status: `blocked-by-keys`
- Goal:
  Run the planner against your real Databricks workspace and one or more real HF models.
- Deliverables:
  - validated inventory snapshot
  - validated model fit output
  - validated cost estimate path
  - fix list if anything breaks
- Acceptance criteria:
  - one real workspace run completes
  - one real model recommendation run completes

## Suggested order for tonight

### Wave 1

- `L1`
- `L2`
- `L3`
- `L5`
- `L7`

### Wave 2

- `L8`
- `L9`
- `L10`

### Wave 3

- `L11`
- `L12`
- `L13`

### Tomorrow

- `L4`
- `L6`
- `L14`

## Suggested subagent ownership

- `Worker A`
  `L1`, `L2`, `L13`
- `Worker B`
  `L3`, later `L4`
- `Worker C`
  `L5`, later `L6`
- `Worker D`
  `L7`
- `Worker E`
  `L8`, `L9`, `L10`, `L11`, `L12`

## UX recommendation

For V1, optimize for `model-first`, with `compute-first` as an equal but slightly simpler secondary path.

That means the text UX should default to:

1. recommendation summary
2. candidate computes
3. cost summary
4. deployment hint

For `compute fit`, return:

1. realistic model ranges
2. example models
3. major limitations

## Review questions

1. Is this lean ticket set now aligned with your original idea?
2. Do you want me to execute this exact backlog tonight after your approval?
