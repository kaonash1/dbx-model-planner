# Azure DBX Capacity Pricing Reference

## Recommended sequence

1. Start from the Databricks node type or target GPU family.
2. Map it to Azure SKU identity and region.
3. Check Databricks-supported GPU subset first.
4. Query Azure SKU restrictions and zones.
5. Query quota state separately.
6. Query Azure Retail Prices API for VM cost inputs.
7. Combine with DBU rates supplied by config or curated references.

## Important data points

Retail Prices API:

- `serviceName`
- `armRegionName`
- `armSkuName`
- `priceType`
- `meterName`
- `retailPrice`
- `unitOfMeasure`
- `effectiveStartDate`
- `reservationTerm`
- `savingsPlan`
- `NextPageLink`

Resource SKUs:

- `name`
- `family`
- `resourceType`
- `locations`
- `locationInfo.zones`
- `capabilities`
- `restrictions`
- `reasonCode`

Databricks system tables:

- `system.compute.node_types`
- `system.compute.clusters`
- `system.compute.node_timeline`

## Operational guardrails

- Retail Prices API is for Microsoft retail prices, not contract-effective prices.
- Retail Prices API is paginated.
- Azure GPU availability is subscription- and region-specific.
- Quota and capacity are separate checks.
- Capacity reservation does not support all GPU series.
- Databricks GPU compute requires ML runtimes and Photon must be off.
- Databricks flexible node types do not apply to GPU clusters.
- As of April 15, 2026, Databricks-supported Azure GPU families should be treated as:
  `NCads_H100_v5`, `NC_A100_v4`, `NDasrA100_v4`, `NVadsA10_v5`, `NCasT4_v3`
- Treat `NC_v3` as retired or historical, not as a recommended target.
- There is no documented Microsoft public API equivalent to Azure Retail Prices for classic Databricks DBU list prices; keep DBU prices configurable.

## Primary sources

- https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices
- https://learn.microsoft.com/en-us/rest/api/compute/resourceskus/list
- https://learn.microsoft.com/en-us/rest/api/quota/
- https://learn.microsoft.com/en-us/azure/virtual-machines/capacity-reservation-overview
- https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview
- https://learn.microsoft.com/en-us/azure/databricks/compute/gpu
- https://learn.microsoft.com/en-us/azure/databricks/resources/supported-regions
- https://learn.microsoft.com/en-us/azure/databricks/admin/system-tables/compute
- https://learn.microsoft.com/en-us/azure/databricks/compute/flexible-node-types
- https://azure.microsoft.com/en-us/pricing/details/databricks/
