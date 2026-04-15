# Domain Model

## Goal

The domain model should give the planner enough structure to answer:

- what model is being considered
- what workload shape is intended
- what Databricks compute and runtimes exist
- what policies or governance constraints apply
- what recommendation the planner produced

## Primary objects

### `ModelProfile`

Represents a normalized model across LLMs, embeddings, rerankers, and VLMs.

Important fields:

- `family`
- `modality`
- `task`
- `parameter_count`
- `active_parameter_count`
- `context_length`
- `dtype_options`
- `quantization_options`
- `capabilities`
- `artifacts`

### `WorkloadProfile`

Represents what the user is trying to optimize for.

Important fields:

- `online`
- `expected_qps`
- `target_latency_ms`
- `target_concurrency`
- `prompt_tokens`
- `completion_tokens`
- `input_sequence_length`
- `scale_to_zero_tolerated`

### `WorkspaceComputeProfile`

Represents a discovered compute option in Azure Databricks.

Important fields:

- `node_type_id`
- `region`
- `gpu_family`
- `gpu_count`
- `gpu_memory_gb`
- `vcpu_count`
- `memory_gb`
- `runtime_ids`
- `supported_hosting_modes`
- `policy_ids`

### `RuntimeProfile`

Represents a Databricks runtime or serving runtime constraint surface.

Important fields:

- `runtime_id`
- `dbr_version`
- `ml_runtime`
- `gpu_enabled`
- `photon_supported`
- `cuda_version`
- `supported_engines`

### `HostingRecommendation`

Represents the planner output.

Important fields:

- `hosting_mode`
- `summary`
- `candidates`
- `deployment_target`
- `blocking_issues`
- `assumptions`

## Secondary objects

### `ModelArtifactProfile`

Tracks model packaging facts such as:

- source repo
- format
- quantization
- artifact size
- license
- gated access
- dependency hints
- processor requirements

### `WorkspacePolicyProfile`

Tracks policy constraints:

- allowed or blocked node types
- allowed runtimes
- required tags

### `CostProfile`

Separates price layers:

- VM hourly rate
- DBU hourly rate
- estimated total
- discounted total
- VAT-adjusted total

### `CandidateCompute`

Represents one evaluated compute option with:

- fit level
- risk level
- quantization recommendation
- memory estimate
- notes
- optional cost

### `DeploymentTarget`

Represents the UC destination:

- catalog
- schema
- volume
- volume path

## Why this split is important

This model prevents the planner from collapsing everything into one table of guesses.

It forces the system to distinguish:

- discovered facts
- inferred facts
- user inputs
- final recommendations

That distinction is important for trust and explainability.
