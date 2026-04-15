# Databricks CLI Planner Reference

## Recommended sequence

1. Identity and workspace probe
2. Workspace and storage inventory
3. Compute and runtime inventory
4. UC model inventory
5. Serving inventory
6. Billing and observability mapping
7. Change-plan output

## Important command families

- `databricks current-user me`
- `databricks auth profiles`
- `databricks workspace list`
- `databricks repos list`
- `databricks fs ls dbfs:/Volumes/...`
- `databricks clusters list`
- `databricks clusters list-node-types`
- `databricks clusters spark-versions`
- `databricks cluster-policies list`
- `databricks instance-pools list`
- `databricks volumes list`
- `databricks registered-models list`
- `databricks model-versions list`
- `databricks serving-endpoints list`
- `databricks serving-endpoints get`
- `databricks serving-endpoints logs`
- `databricks serving-endpoints build-logs`
- `databricks account billable-usage download`

## Operational guardrails

- Databricks CLI `0.205+` is Public Preview. Prefer planning and discovery first.
- `workspace` inventories `/Workspace`; UC volume files are accessed through `fs` and `dbfs:/Volumes/...`.
- `clusters list` is time-bounded; terminated visibility is limited.
- Jobs-created clusters are not editable in the same way as interactive clusters.
- Pool instance type is immutable.
- UC models require UC-enabled workspaces and UC-capable compute.
- New UC model versions require signatures.
- UC models use aliases instead of stages.
- Serving endpoint modes are materially different:
  custom, Foundation Model APIs pay-per-token, provisioned throughput, external models.
- Legacy inference tables are being retired; plan only AI Gateway-enabled tracking.
- `system.billing.usage` requires Unity Catalog system-table access.

## Primary sources

- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/commands
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/reference/workspace-commands
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/reference/fs-commands
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/reference/clusters-commands
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/reference/cluster-policies-commands
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/reference/instance-pools-commands
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/reference/volumes-commands
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/reference/registered-models-commands
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/reference/model-versions-commands
- https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/
- https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/
- https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/manage-serving-endpoints
- https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-limits
- https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/overview-serving-endpoints
- https://learn.microsoft.com/en-us/azure/databricks/admin/system-tables/billing
