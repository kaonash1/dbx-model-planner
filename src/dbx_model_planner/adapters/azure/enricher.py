from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ...domain.profiles import CostProfile, WorkspaceComputeProfile
from ...engines.cost import build_cost_profile
from .pricing import (
    AzureRetailPriceQuery,
    AzureRetailPriceRecord,
    fetch_azure_retail_prices,
    normalize_azure_region,
    normalize_azure_restrictions,
    select_azure_retail_price,
)
from .sku import AzureSkuMapping, map_node_type_to_azure_sku


@dataclass(slots=True)
class AzurePricingInputs:
    """External cost inputs used by the Azure enricher."""

    dbu_hourly_rate: float | None = None
    discount_rate: float = 0.0
    vat_rate: float = 0.0
    currency_code: str = "USD"


@dataclass(slots=True)
class AzurePricingEstimate:
    """Structured pricing result for a selected Azure VM SKU."""

    cost: CostProfile
    vm_retail_price_record: AzureRetailPriceRecord | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AzureComputeEnrichment:
    """Structured Azure enrichment for a Databricks compute profile."""

    compute: WorkspaceComputeProfile
    sku_mapping: AzureSkuMapping
    normalized_region: str | None = None
    normalized_restrictions: list[str] = field(default_factory=list)
    price_candidates: list[AzureRetailPriceRecord] = field(default_factory=list)
    selected_price: AzureRetailPriceRecord | None = None
    pricing: AzurePricingEstimate | None = None
    notes: list[str] = field(default_factory=list)


def build_azure_cost_profile(
    *,
    vm_hourly_rate: float | None,
    dbu_hourly_rate: float | None,
    discount_rate: float,
    vat_rate: float,
    currency_code: str,
) -> CostProfile:
    """Compose a cost profile without discovering DBU pricing.

    Delegates to the canonical ``build_cost_profile`` in the cost engine so
    the formula is never duplicated.
    """

    return build_cost_profile(
        vm_hourly_rate=vm_hourly_rate,
        dbu_hourly_rate=dbu_hourly_rate,
        discount_rate=discount_rate,
        vat_rate=vat_rate,
        currency_code=currency_code,
    )


def enrich_azure_compute(
    compute: WorkspaceComputeProfile,
    *,
    dbu_hourly_rate: float | None = None,
    discount_rate: float = 0.0,
    vat_rate: float = 0.0,
    currency_code: str = "USD",
    price_query: AzureRetailPriceQuery | None = None,
    opener: Callable[[str, float], object] | None = None,
    timeout: float = 30.0,
    extra_restrictions: object | None = None,
) -> AzureComputeEnrichment:
    """Enrich a workspace compute profile with Azure SKU and pricing facts."""

    sku_mapping = map_node_type_to_azure_sku(compute.node_type_id)
    normalized_region = normalize_azure_region(compute.region)
    normalized_restrictions = normalize_azure_restrictions(extra_restrictions)
    notes: list[str] = []

    if normalized_region is None and compute.region:
        notes.append("Compute region could not be normalized to an ARM region name.")

    query = price_query or AzureRetailPriceQuery(
        arm_region_name=normalized_region,
        arm_sku_names=sku_mapping.arm_sku_candidates,
        currency_code=currency_code,
    )
    price_candidates = fetch_azure_retail_prices(query, opener=opener, timeout=timeout)
    selected_price = select_azure_retail_price(
        price_candidates,
        arm_region_name=normalized_region,
        arm_sku_names=sku_mapping.arm_sku_candidates,
    )

    pricing = None
    if selected_price is not None:
        cost_profile = build_azure_cost_profile(
            vm_hourly_rate=selected_price.unit_price,
            dbu_hourly_rate=dbu_hourly_rate,
            discount_rate=discount_rate,
            vat_rate=vat_rate,
            currency_code=selected_price.currency_code or currency_code,
        )
        pricing_notes = []
        if dbu_hourly_rate is None:
            pricing_notes.append("DBU hourly rate was not supplied, so only the VM retail price is represented.")
        pricing = AzurePricingEstimate(
            cost=cost_profile,
            vm_retail_price_record=selected_price,
            notes=pricing_notes,
        )
    else:
        notes.append("No Azure Retail Prices record matched the node type and region.")

    return AzureComputeEnrichment(
        compute=compute,
        sku_mapping=sku_mapping,
        normalized_region=normalized_region,
        normalized_restrictions=normalized_restrictions,
        price_candidates=price_candidates,
        selected_price=selected_price,
        pricing=pricing,
        notes=notes,
    )
