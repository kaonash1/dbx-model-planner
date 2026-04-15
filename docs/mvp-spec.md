# MVP Spec

## Product statement

Build a planner that answers:

- "Which models can I realistically run on the Databricks compute available to me?"
- "Which Databricks compute should I choose for this model and workload?"
- "What would it cost under our internal pricing formula?"
- "What is the smallest safe deployment path from Hugging Face to Databricks?"

## Supported model families

The MVP should explicitly support three families from day one:

1. LLMs
2. embedding models
3. VLMs

It may also cover rerankers because they fit the same pattern as embedding and encoder models.

## Non-goals for v1

- no promise of exact throughput prediction
- no promise of exact memory prediction
- no universal one-click quantization
- no full production endpoint lifecycle management
- no support for every inference engine in the first release

## User stories

1. As a Senior AI Solutions Engineer, I want to choose between Databricks-hosted, external, and self-hosted GPU deployment paths based on model fit, region support, security constraints, runtime support, and cost.
2. As a Databricks user, I want to list my available GPU compute types, runtimes, and policy limits.
3. As a practitioner, I want to input a model and see which compute types can run it.
4. As a practitioner, I want to input a compute type and see candidate models across LLM, embedding, and VLM families.
5. As a cost-conscious user, I want a cost estimate with company discount and VAT rules.
6. As a deployment-oriented user, I want a suggested path to stage artifacts in a UC volume and generate a starter run script.

## Core entities

### Workspace compute profile

- cloud: Azure
- node type
- GPU model
- GPU count
- GPU memory
- CPU count
- RAM
- local disk
- availability
- cluster policy constraints
- supported DBR versions

### Runtime profile

- DBR version
- ML runtime or standard runtime
- CUDA compatibility if relevant
- supported libraries and engines

### Model profile

- source: Hugging Face, Databricks hosted, internal registry
- family: llm, embedding, vlm, reranker
- task
- modality: text, image-text, text-embedding
- parameter count if known
- artifact size if known
- context length if applicable
- architecture hints
- dtype options
- quantization compatibility hints
- license and gating notes

### Cost profile

- Azure VM retail rate
- Databricks DBU list price
- effective rate after company discount
- VAT-adjusted rate
- estimated monthly run cost

## Core outputs

### `model fit`

For a model, return:

- compatible compute options
- recommended compute option
- precision and quantization options
- risk level: safe, borderline, unlikely
- estimated hourly cost
- deployment notes

### `compute fit`

For a compute type, return:

- realistic model ranges by family
- example models
- memory headroom guidance
- expected tradeoffs

### `deploy plan`

Return:

- recommended UC volume path
- recommended cluster shape
- package and dependency notes
- starter script or notebook skeleton
- optional next step for Unity Catalog model registration

## Planning heuristics

The planner will begin with transparent heuristics rather than pretending exactness.

Examples:

- parameter memory estimate by dtype or quantization
- extra memory headroom for runtime overhead
- additional allowance for KV cache on decoder models
- family-specific multipliers for VLM encoders and multimodal pipelines
- embeddings treated as throughput-sensitive rather than purely memory-sensitive

Every estimate should be labeled as an estimate.

## Decision dimensions

The MVP should be organized around the questions a senior engineer actually needs answered:

1. hosting mode
2. workspace and region eligibility
3. compute and runtime fit
4. model artifact and dependency fit
5. expected latency, throughput, and cold-start behavior
6. governance and access path
7. cost and observability

See the decision framework document for the research-backed breakdown of these dimensions.

## Data sources

- Databricks CLI and APIs for node types, runtimes, clusters, policies, pools
- Databricks system tables for compute metadata and billing
- Azure Retail Prices API for VM price lookup
- Hugging Face Hub metadata for public model details
- local company configuration for discount, VAT, overrides, and blocked SKUs

## MVP boundaries

The first usable release is reached when:

1. workspace inventory sync works
2. model-family-aware fit recommendations work
3. pricing estimates work with local company rules
4. deployment plan generation works as text output

Artifact transfer, quantization execution, and registry automation can follow after that.
