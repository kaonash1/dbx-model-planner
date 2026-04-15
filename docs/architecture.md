# Architecture

## Positioning

The system should be designed as a planner with optional execution adapters later.

That means:

- the recommendation engine is the center
- data collectors feed it
- deployment actions stay behind explicit commands

## High-level components

### 1. Workspace inventory collector

Collects live Databricks facts:

- node types
- runtimes
- pools
- cluster policies
- optional billing and usage metadata

Primary sources:

- Databricks SDK and REST APIs
- Databricks CLI as a convenience path where useful
- Databricks SQL against system tables

### 2. Model catalog adapter

Normalizes external model metadata into an internal profile:

- Hugging Face metadata
- optional internal allowlists or curated catalogs
- later: Databricks-hosted models and internal UC models

### 3. Fit engine

The decision layer that maps model profiles onto compute profiles.

Responsibilities:

- estimate memory fit
- estimate runtime suitability
- classify risk level
- rank candidate compute options

The fit engine must remain family-aware:

- LLMs are sensitive to context length, KV cache, and quantization
- embedding models are often more throughput and batching oriented
- VLMs add multimodal encoders and processor dependencies

### 4. Cost engine

Builds an estimated cost view from:

- Azure VM pricing
- Databricks DBU pricing
- company discount
- VAT

It should separate:

- list cost
- internal discounted cost
- VAT-adjusted reporting cost

### 5. Plan generator

Produces user-facing plans such as:

- recommended cluster config
- deployment steps
- UC volume path
- starter inference code

### 6. Persistence layer

Stores synced and normalized data locally for fast queries.

Suggested first choice:

- SQLite for simplicity

Possible later move:

- DuckDB if analytical joins become more important

## Suggested package layout

```text
src/dbx_model_planner/
  cli.py
  config.py
  domain/
    common.py
    profiles.py
    recommendations.py
  collectors/
  adapters/
  engines/
  planners/
  storage/
```

## Concrete domain model

The first implementation should center around five primary contracts:

- `ModelProfile`
- `WorkloadProfile`
- `WorkspaceComputeProfile`
- `RuntimeProfile`
- `HostingRecommendation`

Supporting contracts:

- `ModelArtifactProfile`
- `WorkspacePolicyProfile`
- `WorkspaceInventorySnapshot`
- `CandidateCompute`
- `CostProfile`
- `DeploymentTarget`

This split matters because the planner must separate:

- sourced facts about models
- sourced facts about workspace compute and runtimes
- workload assumptions supplied by the user
- recommendation output produced by the engine

## Package responsibilities

### `domain`

Pure dataclasses and enums. No API calls, no pricing lookups, no I/O.

### `collectors`

Databricks, Azure, and billing inventory sync code.

### `adapters`

Normalizers that translate Hugging Face, Databricks, or internal data into domain objects.

### `engines`

Core decision logic:

- fit engine
- cost engine
- hosting mode selector

### `planners`

User-facing assembly logic that turns engine output into:

- ranked recommendations
- deployment plans
- starter scripts

### `storage`

Persistence for normalized inventory snapshots and cached model metadata.

## Config strategy

Use explicit local configuration for organization-specific rules:

- discount percentage
- VAT percentage
- preferred regions
- blocked GPU families
- cluster policy overrides
- approved runtimes

This keeps company-specific logic out of the generic engine.

## Delivery phases

### Phase 1

- project scaffold
- config model
- inventory sync contract
- model profile contract
- CLI command surface

### Phase 2

- Databricks inventory sync
- HF metadata adapter
- first fit heuristics for LLMs, embeddings, and VLMs
- local cache

### Phase 3

- pricing integration
- ranked recommendations
- text deployment plans

### Phase 4

- optional artifact staging to UC volumes
- optional quantization helper workflows
- optional Unity Catalog registration helper

## Key design constraints

- never hardcode workspace availability when it can be discovered live
- keep recommendations explainable
- keep estimates transparent
- treat deployment execution as opt-in
- avoid coupling the planner to a single model family
