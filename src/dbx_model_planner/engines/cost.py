from __future__ import annotations

from dbx_model_planner.config import AppConfig
from dbx_model_planner.domain import CostProfile, EstimateSource


def compose_cost_profile(
    config: AppConfig,
    vm_hourly_rate: float | None,
    dbu_hourly_rate: float | None,
    pricing_reference: str | None = None,
) -> CostProfile:
    vm_rate = vm_hourly_rate or 0.0
    dbu_rate = dbu_hourly_rate or 0.0
    estimated_hourly_rate = round(vm_rate + dbu_rate, 4)
    discounted_hourly_rate = round(estimated_hourly_rate * (1.0 - config.pricing.discount_rate), 4)
    vat_adjusted_hourly_rate = round(discounted_hourly_rate * (1.0 + config.pricing.vat_rate), 4)

    return CostProfile(
        currency_code=config.pricing.currency_code,
        vm_hourly_rate=round(vm_rate, 4),
        dbu_hourly_rate=round(dbu_rate, 4),
        estimated_hourly_rate=estimated_hourly_rate,
        discounted_hourly_rate=discounted_hourly_rate,
        vat_adjusted_hourly_rate=vat_adjusted_hourly_rate,
        pricing_reference=pricing_reference,
        source=EstimateSource.INFERRED,
    )
