"""Debug script: inspect SKU mapping and Azure pricing URL for sample GPU node types."""

from dbx_model_planner.adapters.azure.sku import (
    arm_sku_candidates_from_node_type,
    normalize_node_type_id,
    map_node_type_to_azure_sku,
)
from dbx_model_planner.adapters.azure.pricing import (
    AzureRetailPriceQuery,
    build_azure_retail_prices_url,
)

SAMPLES = [
    "Standard_NC24ads_A100_v4",
    "Standard_ND96asr_v4",
    "Standard_NC6s_v3",
    "Standard_NC4as_T4_v3",
]

print("=" * 80)
print("SKU MAPPING RESULTS")
print("=" * 80)

all_candidates = []
for nid in SAMPLES:
    norm = normalize_node_type_id(nid)
    candidates = arm_sku_candidates_from_node_type(nid)
    mapping = map_node_type_to_azure_sku(nid)
    all_candidates.extend(candidates)

    print(f"\nNode type:   {nid}")
    print(f"  Normalized:      {norm}")
    print(f"  ARM candidates:  {candidates}")
    print(f"  VM series:       {mapping.vm_series}")
    print(f"  GPU family:      {mapping.gpu_family}")
    if mapping.notes:
        print(f"  Notes:           {mapping.notes}")

print("\n" + "=" * 80)
print("AZURE RETAIL PRICES URL (westeurope, all candidates)")
print("=" * 80)

query = AzureRetailPriceQuery(
    arm_region_name="westeurope",
    arm_sku_names=all_candidates,
)
url = build_azure_retail_prices_url(query)
print(f"\n{url}\n")
print(f"URL length: {len(url)} chars")
