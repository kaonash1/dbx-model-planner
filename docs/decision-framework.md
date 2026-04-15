# Decision Framework

## Persona statement

As a Senior AI Solutions Engineer, I need to determine the safest and most cost-effective way to host a model on Databricks by answering:

- should this workload use Databricks-hosted foundation models, external models, or self-hosted custom serving on GPU
- if it should be self-hosted, which Azure Databricks GPU compute and runtime can support it
- what constraints exist around region, networking, access control, governance, and packaging
- what latency, throughput, and cold-start tradeoffs to expect
- what the real cost is after Azure price, Databricks DBU usage, internal discount rules, and VAT

This should work for:

- LLMs
- embedding models
- rerankers
- VLMs and other multimodal models

## Research-backed decision areas

### 1. Hosting mode

Before picking a GPU SKU, the tool should first answer whether the model should be:

- queried through Databricks-hosted Foundation Model APIs
- exposed as an external model endpoint governed by Databricks
- packaged as a custom model and served on Databricks-managed CPU or GPU
- run on classic GPU compute for batch or notebook inference instead of online serving

Why this matters:

- Databricks already offers multiple hosting modes with different operational tradeoffs.
- Foundation Model APIs provide pay-per-token and provisioned throughput modes.
- External models can centralize governance for models hosted outside Databricks.
- Custom models require MLflow packaging, registration, and endpoint configuration.

Planner implication:

- `hosting mode` must be the first recommendation step, not a detail at the end.

## 2. Workspace, region, and security eligibility

The tool should surface:

- whether the workspace region supports the target serving capability
- whether the control plane and feature region combination supports model serving
- whether private connectivity requirements rule out certain options
- whether compliance or networking requirements narrow the viable endpoint types

Why this matters:

- Model Serving is region-gated.
- Private connectivity support differs by endpoint type.
- Endpoint creator identity is tied to Unity Catalog access for the deployment.

Planner implication:

- the tool must reject impossible paths early
- the tool should warn when the correct answer is "this should be deployed by a service principal, not by an individual user"

## 3. Compute and runtime fit

The tool should capture:

- GPU family, count, and VRAM
- CPU and system RAM
- DBR version compatibility
- ML runtime requirement
- Photon support or lack of support
- whether the workspace policies allow the chosen compute

Why this matters:

- Azure Databricks GPU compute requires GPU-capable runtimes and Photon is not supported on GPU instance types.
- Instance families materially change what is runnable: H100 94 GB, A100 80 GB, A10 24 GB, T4 16 GB.
- Workspace reality is not just cloud catalog reality. Policies and regional availability can rule out compute even when Azure supports it.

Planner implication:

- inventory must combine live workspace discovery with static capability metadata
- the recommendation engine must reason in terms of available VRAM and policy-filtered node types

## 4. Model artifact, processor, and dependency fit

The tool should normalize the following for every model family:

- model family and task
- parameter count and artifact size when available
- context length or max sequence length
- tokenizer or processor requirements
- image processor requirements for VLMs and vision models
- framework and dependency expectations
- dtype and quantization compatibility hints
- licensing or gated-access constraints

Why this matters:

- Unity Catalog model registration depends on MLflow packaging and signatures.
- VLMs require more than weights: they often require processor config and image preprocessing metadata.
- A model can fit in VRAM but still fail operationally because the required dependencies or processors are missing.

Planner implication:

- the model adapter cannot stop at parameter count
- VLM support must include processor metadata, not only model config

## 5. Performance and serving envelope

The tool should ask for or infer:

- online or batch workload
- latency target
- expected QPS
- concurrency target
- context length or input size assumptions
- scale-to-zero tolerance
- cold-start tolerance

Why this matters:

- Databricks serving guidance ties compute scale-out to roughly `QPS x model runtime`.
- Scale to zero is explicitly not recommended for production because capacity is not guaranteed and cold-start latency is added.
- Custom model serving has concrete payload, execution duration, and concurrency limits.
- Embedding and reranker workloads are often throughput-bound rather than purely memory-bound.

Planner implication:

- the recommendation should ask for workload shape, not just model name
- `fits in memory` is insufficient as a final answer

## 6. Governance, storage, and deployment path

The tool should surface:

- whether artifacts belong in a managed or external UC volume
- whether the target compute is Unity Catalog-enabled
- whether the model needs to be registered in Unity Catalog
- whether the endpoint identity has access to the required UC resources

Why this matters:

- Volumes are governed storage for non-tabular artifacts and use `/Volumes/<catalog>/<schema>/<volume>/...` paths.
- Volumes require Unity Catalog-enabled compute and DBR 13.3 LTS or above.
- For custom model serving, the endpoint creator identity is used for Unity Catalog resource access and cannot be changed later.

Planner implication:

- deployment plans must include storage and identity checks, not just code snippets

## 7. Cost and observability

The tool should separate:

- estimated infrastructure cost
- Databricks usage cost
- internal discounted cost
- VAT-adjusted reporting cost
- observed cost from usage telemetry

The tool should also expose:

- model serving usage SKUs when using serverless serving
- historical compute utilization when sizing GPU clusters
- observed billing joined to cluster and endpoint metadata

Why this matters:

- Azure VM cost and Databricks DBU cost are separate signals.
- Databricks system tables provide both usage and list pricing joins.
- Model serving costs can be tracked through specific serving-related SKUs in `system.billing.usage`.

Planner implication:

- the planner needs both an `estimate` mode and an `observed` mode

## Facts the product should ingest directly

The following should come from source systems whenever possible:

- available node types
- available Spark and DBR versions
- cluster policies
- Unity Catalog and system table access
- Azure region and retail price data
- Databricks usage and price tables
- Hugging Face model metadata

## Inferences the planner should make

The following are not directly given by the platforms and are the real product value:

- whether a model is a safe, borderline, or poor fit for a given GPU shape
- whether a hosted or external model path is preferable to self-hosting
- what quantization level is likely required
- how much headroom should be reserved for runtime overhead, KV cache, batching, and multimodal processors
- which option is cheapest for a given workload shape

These should always be marked as estimates or recommendations, not presented as ground truth.

## What the MVP must know to be credible

For each recommendation, the tool should be able to explain:

1. why this hosting mode was chosen
2. why this compute was selected or rejected
3. what runtime or package constraint matters
4. what cost components were included
5. which assumptions are user inputs versus discovered facts

If it cannot explain those five points, the recommendation is too weak.

## Sources

- Azure Databricks GPU-enabled compute: https://learn.microsoft.com/en-us/azure/databricks/compute/gpu
- Azure Databricks Model Serving limits and regions: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-limits
- Azure Databricks system tables overview: https://learn.microsoft.com/en-us/azure/databricks/admin/system-tables/
- Azure Databricks compute system tables: https://learn.microsoft.com/en-us/azure/databricks/admin/system-tables/compute
- Azure Databricks billing and pricing system tables: https://learn.microsoft.com/en-us/azure/databricks/admin/usage/system-tables
- Azure Databricks model serving cost monitoring: https://learn.microsoft.com/en-us/azure/databricks/admin/system-tables/model-serving-cost
- Azure Retail Prices API: https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices
- Databricks custom models overview: https://docs.databricks.com/aws/en/machine-learning/model-serving/custom-models
- Databricks custom model serving endpoints: https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints
- Databricks Foundation Model APIs overview: https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis
- Databricks supported foundation models: https://docs.databricks.com/aws/en/machine-learning/model-serving/foundation-model-overview
- Databricks Unity Catalog volumes: https://docs.databricks.com/aws/en/volumes
- Databricks CLI clusters commands: https://docs.databricks.com/aws/en/dev-tools/cli/reference/clusters-commands
- Hugging Face Hub API model metadata: https://huggingface.co/docs/huggingface_hub/main/package_reference/hf_api
- Hugging Face configuration docs: https://huggingface.co/docs/transformers/main/en/main_classes/configuration
- Hugging Face image processor docs: https://huggingface.co/docs/transformers/main/image_processors
