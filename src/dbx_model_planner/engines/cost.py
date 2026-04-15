from __future__ import annotations

from dbx_model_planner.config import AppConfig
from dbx_model_planner.domain import CostProfile, EstimateSource


def build_cost_profile(
    *,
    vm_hourly_rate: float | None,
    dbu_hourly_rate: float | None,
    discount_rate: float,
    vat_rate: float,
    currency_code: str,
    pricing_reference: str | None = None,
    source: EstimateSource = EstimateSource.INFERRED,
) -> CostProfile:
    """Build a CostProfile from explicit rate parameters.

    This is the single source of truth for the cost composition formula.
    Both the Azure enricher and the config-aware helper delegate here.
    """

    if vm_hourly_rate is None and dbu_hourly_rate is None:
        return CostProfile(currency_code=currency_code)

    estimated_hourly_rate = round((vm_hourly_rate or 0.0) + (dbu_hourly_rate or 0.0), 4)
    discounted_hourly_rate = round(estimated_hourly_rate * (1.0 - discount_rate), 4)
    vat_adjusted_hourly_rate = round(discounted_hourly_rate * (1.0 + vat_rate), 4)

    return CostProfile(
        currency_code=currency_code,
        vm_hourly_rate=round(vm_hourly_rate, 4) if vm_hourly_rate is not None else None,
        dbu_hourly_rate=round(dbu_hourly_rate, 4) if dbu_hourly_rate is not None else None,
        estimated_hourly_rate=estimated_hourly_rate,
        discounted_hourly_rate=discounted_hourly_rate,
        vat_adjusted_hourly_rate=vat_adjusted_hourly_rate,
        pricing_reference=pricing_reference,
        source=source,
    )


def compose_cost_profile(
    config: AppConfig,
    vm_hourly_rate: float | None,
    dbu_hourly_rate: float | None,
    pricing_reference: str | None = None,
) -> CostProfile:
    return build_cost_profile(
        vm_hourly_rate=vm_hourly_rate,
        dbu_hourly_rate=dbu_hourly_rate,
        discount_rate=config.pricing.discount_rate,
        vat_rate=config.pricing.vat_rate,
        currency_code=config.pricing.currency_code,
        pricing_reference=pricing_reference,
        source=EstimateSource.INFERRED,
    )
