# llmfit Analysis

## Summary

`llmfit` is a strong reference for one specific problem:

- "Given this machine, which local models and quantizations are likely to run well?"

It is not solving the same problem as `dbx-model-planner`, but it is close enough in its fit engine that it is worth studying carefully.

The most important conclusion is:

- `llmfit` is a node-local hardware fit and runtime recommendation engine
- `dbx-model-planner` needs to be a workspace-aware cloud deployment planner

That means we should reuse some of its abstractions and estimation ideas, but not its product boundary.

## What llmfit is

At a product level, `llmfit` is:

- a Rust workspace with a reusable core library
- a TUI-first user experience with CLI and REST API surfaces
- a hardware detection and recommendation engine
- a local runtime/provider integration layer

The repo is structured as a multi-crate workspace with:

- `llmfit-core`
- `llmfit-tui`
- `llmfit-desktop`

The root workspace file confirms this split.

## What it does well

### 1. Clear central abstraction

The core abstraction is:

- detect hardware
- normalize model metadata
- estimate memory fit
- estimate speed
- rank models by weighted score

That is the right shape for a recommendation engine.

### 2. Strong fit-engine design

The core engine in `llmfit-core/src/fit.rs` is more serious than a naive "params divided by VRAM" calculator.

It includes:

- run modes: `Gpu`, `MoeOffload`, `CpuOffload`, `CpuOnly`, `TensorParallel`
- fit levels: `Perfect`, `Good`, `Marginal`, `TooTight`
- score components: quality, speed, fit, context
- runtime selection: `llama.cpp`, `MLX`, `vLLM`

This is useful because it separates:

- "does it fit"
- "how will it run"
- "how good is this choice"

That separation is worth copying conceptually.

### 3. Better-than-average memory modeling

The model layer in `llmfit-core/src/models.rs` goes beyond a static parameter-size table.

It models:

- quantization bytes-per-parameter
- KV cache size
- explicit KV cache quantization options
- fallback and precise KV cache formulas
- hybrid attention layouts
- tensor-parallel compatibility
- pre-quantized formats like AWQ and GPTQ

This is one of the strongest parts of the repo.

For our project, the key lesson is that fit needs to be:

- model-family-aware
- runtime-aware
- quantization-aware
- context-aware

### 4. Practical MoE treatment

The scraper and fit engine make a deliberate distinction between:

- total parameter count
- active parameter count

That matters for models like Mixtral, DeepSeek, Qwen MoE, and similar architectures.

This is exactly the kind of thing many simplistic planners miss.

### 5. Explainable planning mode

`llmfit-core/src/plan.rs` includes an inverse mode:

- instead of "what fits my hardware?"
- it also answers "what hardware do I need for this model and target TPS?"

That is directly relevant to what we want.

The useful product lesson is:

- our planner should work in both directions:
- model -> compute
- compute -> model

### 6. Good API surface

The project has:

- TUI
- CLI
- HTTP API

The REST layer is framed as a node-local scheduling endpoint. That is a good architectural move because it separates the fit engine from the presentation layer.

For us, the analogous split would be:

- collectors
- planning engine
- CLI
- later, maybe service or UI

## What llmfit does not solve

This is the most important section.

### 1. It is not workspace-aware

`llmfit` reasons about a machine, not a governed cloud estate.

It does not model:

- Databricks workspaces
- cluster policies
- allowed node types
- DBR compatibility
- region restrictions
- Unity Catalog requirements
- serving-mode constraints

That is the core gap between `llmfit` and our project.

### 2. It is not cost-aware

There is no real pricing or cost engine.

It does not model:

- Azure VM cost
- Databricks DBU cost
- company discount rules
- VAT
- observed billing from Databricks usage tables

For our use case, this is not optional. It is one of the main reasons to build the planner in the first place.

### 3. Its provider layer is local-runtime-centric

The provider layer is designed around:

- Ollama
- llama.cpp
- MLX
- LM Studio
- Docker Model Runner

That is useful for laptops and local servers, but it is the wrong operational center for Databricks.

For us, the equivalent integrations are more like:

- Databricks compute inventory
- Databricks Model Serving modes
- Unity Catalog volumes
- Unity Catalog model registration
- Hugging Face staging and artifact prep

### 4. It uses a curated catalog, not live infrastructure facts

The model database is scraped from Hugging Face and embedded at build time.

That works for a local CLI, but for Databricks planning we need live facts for:

- available node types
- allowed policies
- available runtimes
- region support
- actual price tables

Static model metadata is fine.
Static infrastructure metadata is not.

### 5. It is still LLM-first even when broader

The repo has support for:

- embedding models
- multimodal and vision models

But its reasoning center is still token-generation-centric.

This shows up in:

- token/s as the dominant speed metric
- fit logic optimized around autoregressive inference
- local runtime assumptions that are chat-LLM heavy

For us, embeddings and VLMs should not be treated as edge cases.

We need first-class support for:

- embeddings
- rerankers
- VLMs
- possibly audio models later

### 6. Some important classifications are heuristic

The code infers use cases and capabilities partly from:

- model names
- tags
- string matching

That is pragmatic and often necessary, but it creates risk:

- misclassification of tool-use capability
- weak inference for VLM specifics
- weak inference for operational packaging needs

We can still use heuristics, but our planner should clearly distinguish:

- sourced facts
- inferred attributes

## The key reusable ideas

We should strongly consider reusing these design ideas in our own implementation.

### Reuse 1: Fit vocabulary

The separation between:

- fit level
- run mode
- score components

is good and should survive into `dbx-model-planner`.

### Reuse 2: Inverse planning

The "plan required hardware for this model" concept is directly aligned with our vision.

### Reuse 3: Quantization-aware memory estimation

Their handling of:

- weight quantization
- KV cache
- KV cache quantization
- MoE active parameter handling

is exactly the sort of foundation we need.

### Reuse 4: Explainable notes

`llmfit` emits notes such as:

- context capped for estimation
- best quantization for hardware
- runtime comparison notes

This is a strong UX pattern. Recommendations should explain themselves.

### Reuse 5: API-first fit engine

The fact that the same engine powers:

- TUI
- CLI
- REST API

is a good architectural pattern.

## The key things we should not copy

### 1. Do not make local runtime providers the center

For our project, local runtime mapping is secondary or optional.

The center must be:

- hosting mode selection
- Databricks workspace inventory
- compute/runtime constraints
- cost and governance

### 2. Do not bake infrastructure truth into the binary

Databricks inventory and pricing must be synced live.

### 3. Do not assume token/s is the dominant metric across all families

For embeddings and rerankers, throughput and batch shape matter more than token generation.

For VLMs, processor overhead and image pipeline dependencies matter too.

### 4. Do not collapse cloud planning into single-node fit

A Databricks recommendation needs to answer:

- can I create this compute here
- is it policy-allowed
- is the runtime compatible
- should I use custom serving, external model, Foundation Model APIs, or batch compute

`llmfit` does not need those questions. We do.

## Recommended adaptation for dbx-model-planner

The best adaptation is not to imitate the whole repo.

It is to import the conceptual core:

1. `model profile`
2. `workload profile`
3. `compute profile`
4. `runtime profile`
5. `cost profile`
6. `hosting mode recommendation`

Then add a `fit engine` that is inspired by `llmfit`, but cloud-aware.

### Proposed mapping

`llmfit` concepts:

- `SystemSpecs`
- `LlmModel`
- `ModelFit`
- `PlanEstimate`

`dbx-model-planner` equivalents:

- `WorkspaceComputeProfile`
- `ModelProfile`
- `ModelHostingRecommendation`
- `DeploymentPlan`

### New dimensions we must add

- hosting mode
- Databricks runtime compatibility
- cluster policy eligibility
- region support
- Unity Catalog pathing and identity constraints
- estimated and observed cost

## Bottom line

`llmfit` is a very good reference for:

- fit heuristics
- quantization-aware model estimation
- inverse planning
- explainable recommendation UX

It is not a substitute for what we are building.

The overlap is in the estimation engine.
The difference is in the decision context.

`llmfit` answers:

- "what runs on this machine?"

`dbx-model-planner` needs to answer:

- "what should I host on this Databricks estate, how should I host it, what will it cost, and what is the shortest safe path to deployment?"

## References

- Repo: https://github.com/AlexsJones/llmfit
- Workspace file: https://github.com/AlexsJones/llmfit/blob/main/Cargo.toml
- README: https://github.com/AlexsJones/llmfit/blob/main/README.md
- Core fit engine: https://github.com/AlexsJones/llmfit/blob/main/llmfit-core/src/fit.rs
- Model layer: https://github.com/AlexsJones/llmfit/blob/main/llmfit-core/src/models.rs
- Hardware layer: https://github.com/AlexsJones/llmfit/blob/main/llmfit-core/src/hardware.rs
- Planning mode: https://github.com/AlexsJones/llmfit/blob/main/llmfit-core/src/plan.rs
- Provider layer: https://github.com/AlexsJones/llmfit/blob/main/llmfit-core/src/providers.rs
- HF scraper: https://github.com/AlexsJones/llmfit/blob/main/scripts/scrape_hf_models.py
- API guide: https://github.com/AlexsJones/llmfit/blob/main/API.md
