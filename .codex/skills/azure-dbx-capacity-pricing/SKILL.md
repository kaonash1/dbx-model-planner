---
name: azure-dbx-capacity-pricing
description: Use when Azure facts are needed for Databricks GPU planning, especially Azure VM SKU and region availability, quota and capacity constraints, Retail Prices API inputs, and Azure-side guardrails for Databricks GPU recommendations.
---

# Azure DBX Capacity Pricing

Use this skill when the task needs Azure facts that sit outside Databricks itself.

This skill is for Azure-side viability and pricing, not for generic ARM resource management.

## Outputs

Produce one of these:

- an Azure-side viability check for a candidate GPU SKU and region
- an Azure VM price input summary for later DBU combination
- a list of quota, restriction, zone, or capacity blockers
- a mapping from Databricks node types to Azure SKU facts

## Core workflows

1. Map Databricks node types to Azure VM SKUs.
2. Check Databricks-supported GPU families before looking at the broader Azure GPU catalog.
3. Check region support, SKU restrictions, zones, and quota family.
4. Distinguish quota from real capacity.
5. Pull Azure Retail Prices API data for the VM portion of hourly cost.
6. Combine Azure VM price with Databricks DBU inputs supplied elsewhere.
7. Surface Azure-side blockers in planner output.

## APIs and data to know

- Azure Retail Prices API
- Azure Compute `ResourceSkus - List`
- Azure Quota Service API
- Azure Databricks compute system tables
- Azure Databricks GPU compute documentation

## Guardrails

- Do not assume Azure Retail Prices equals customer invoice pricing.
- Do not assume regional quota means regional capacity.
- Do not assume the full Azure GPU catalog is usable from Azure Databricks.
- Do not recommend retired Databricks GPU generations such as `NC_v3`.
- Do not use flexible node types as a GPU fallback strategy.
- Do not use this skill to inspect Unity Catalog, serving endpoints, or Databricks workspace objects; use the Databricks skill for that.

## When to read references

- Read `references/apis-and-sku-notes.md` when you need concrete fields, Azure-side limitations, or source links.
