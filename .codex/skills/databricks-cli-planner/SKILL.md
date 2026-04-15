---
name: databricks-cli-planner
description: Use when inspecting or planning changes in an Azure Databricks workspace via the Databricks CLI and official APIs, especially for identity checks, workspace inventory, compute and runtime discovery, cluster policies, instance pools, Unity Catalog volumes and models, serving endpoints, and billing-aware change plans.
---

# Databricks CLI Planner

Use this skill when the task depends on live Azure Databricks workspace facts.

This is a read-first planning skill. Prefer inspection, inventory, and plan generation before mutation.

## Outputs

Produce one of these:

- a concise workspace inventory summary
- a command-level inspection plan
- a change plan with prerequisites, blockers, and the exact command families to use
- a list of facts still missing from the workspace or identity context

## Core workflows

1. Confirm identity, profile, workspace URL, and UC context.
2. Inventory workspace objects, repos, DBFS, and UC volume paths.
3. Inventory compute options:
   node types, DBR versions, zones, clusters, policies, pools.
4. Inventory UC model objects:
   registered models, model versions, aliases, permissions, workspace bindings.
5. Inventory serving state:
   endpoint types, served entities, config, logs, AI Gateway settings, metrics.
6. Map compute or serving choices to cost and observability surfaces.
7. Emit a concrete change plan with blockers and required commands.

## Command families to know

- `databricks current-user me`
- `databricks auth ...`
- `databricks workspace ...`
- `databricks repos ...`
- `databricks fs ...`
- `databricks clusters ...`
- `databricks cluster-policies ...`
- `databricks instance-pools ...`
- `databricks volumes ...`
- `databricks registered-models ...`
- `databricks model-versions ...`
- `databricks workspace-bindings ...`
- `databricks serving-endpoints ...`
- `databricks account billable-usage download`

## Guardrails

- Do not assume one command group exposes the whole workspace.
- Do not treat UC volumes and `/Workspace` paths as the same storage surface.
- Do not assume CLI parity across older CLI versions.
- Do not recommend mutations before checking policy, runtime, identity, and UC prerequisites.
- Do not plan around legacy inference tables. Prefer AI Gateway-enabled inference tracking as of April 15, 2026.
- Do not use this skill for Azure VM pricing or quota checks alone; use the Azure skill for that.

## When to read references

- Read `references/commands-and-guardrails.md` when you need the exact command/API families, date-sensitive guardrails, or source links.
