from __future__ import annotations

import io
import json
from pathlib import Path

from dbx_model_planner.adapters.azure import (
    AzureRetailPriceQuery,
    AzureSkuMapping,
    build_azure_cost_profile,
    enrich_azure_compute,
    fetch_azure_retail_prices,
    map_node_type_to_azure_sku,
    normalize_azure_region,
    normalize_azure_restrictions,
    parse_azure_retail_price_item,
    parse_azure_retail_prices_page,
    select_azure_retail_price,
)
from dbx_model_planner.domain.profiles import WorkspaceComputeProfile


FIXTURES = Path(__file__).parent / "fixtures"


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def _fixture_opener(responses: list[dict[str, object]]):
    queue = [json.dumps(payload).encode("utf-8") for payload in responses]

    def opener(url: str, timeout: float) -> _FakeResponse:
        if not queue:
            raise AssertionError(f"unexpected request for {url!r}")
        return _FakeResponse(queue.pop(0))

    return opener


def test_node_type_mapping_preserves_candidate_arm_skus() -> None:
    mapping = map_node_type_to_azure_sku("Standard_NC4as_T4_v3")

    assert isinstance(mapping, AzureSkuMapping)
    assert mapping.normalized_node_type_id == "Standard_NC4as_T4_v3"
    assert mapping.arm_sku_candidates[0] == "Standard_NC4as_T4_v3"
    assert mapping.gpu_family == "T4"
    assert mapping.vm_series == "NC4as"
    assert mapping.vm_family == "nc"


def test_region_and_restriction_normalization() -> None:
    assert normalize_azure_region("EU West") == "westeurope"
    assert normalize_azure_region("West Europe") == "westeurope"
    assert normalize_azure_restrictions(["Consumption", {"type": "Reservation"}, "Primary Meter Region"]) == [
        "consumption",
        "reservation",
        "primary_meter_region",
    ]


def test_parse_and_page_fetcher_paginates() -> None:
    page1 = json.loads((FIXTURES / "retail_prices_page_1.json").read_text())
    page2 = json.loads((FIXTURES / "retail_prices_page_2.json").read_text())

    parsed = parse_azure_retail_prices_page(page1)
    assert parsed.count == 1
    assert parsed.next_page_link and parsed.next_page_link.endswith("$skip=1000")
    assert parsed.items[0].normalized_region == "westeurope"
    assert parsed.items[0].normalized_restrictions == ["consumption", "primary_meter_region"]

    records = fetch_azure_retail_prices(
        AzureRetailPriceQuery(
            arm_region_name="westeurope",
            arm_sku_names=("Standard_NC4as_T4_v3",),
        ),
        opener=_fixture_opener([page1, page2]),
    )

    assert len(records) == 2
    assert records[0].arm_sku_name == "Standard_NC4as_T4_v3"
    assert records[1].normalized_price_type == "devtestconsumption"


def test_select_price_prefers_region_and_sku_match() -> None:
    page1 = json.loads((FIXTURES / "retail_prices_page_1.json").read_text())
    page2 = json.loads((FIXTURES / "retail_prices_page_2.json").read_text())
    prices = [
        parse_azure_retail_price_item(item)
        for item in page1["Items"] + page2["Items"]
    ]

    selected = select_azure_retail_price(
        prices,
        arm_region_name="West Europe",
        arm_sku_names=("Standard_NC4as_T4_v3",),
    )

    assert selected is not None
    assert selected.retail_price == 3.25
    assert selected.normalized_region == "westeurope"


def test_enrichment_composes_vm_cost_and_external_dbu() -> None:
    page1 = json.loads((FIXTURES / "retail_prices_page_1.json").read_text())
    page2 = json.loads((FIXTURES / "retail_prices_page_2.json").read_text())
    compute = WorkspaceComputeProfile(
        node_type_id="Standard_NC4as_T4_v3",
        region="West Europe",
    )

    enrichment = enrich_azure_compute(
        compute,
        dbu_hourly_rate=0.75,
        discount_rate=0.1,
        vat_rate=0.2,
        opener=_fixture_opener([page1, page2]),
    )

    assert enrichment.selected_price is not None
    assert enrichment.pricing is not None
    assert enrichment.pricing.cost.vm_hourly_rate == 3.25
    assert enrichment.pricing.cost.dbu_hourly_rate == 0.75
    assert enrichment.pricing.cost.estimated_hourly_rate == 4.0
    assert round(enrichment.pricing.cost.discounted_hourly_rate or 0.0, 3) == 3.6
    assert round(enrichment.pricing.cost.vat_adjusted_hourly_rate or 0.0, 3) == 4.32


def test_build_cost_profile_without_dbu_input_keeps_it_external() -> None:
    cost = build_azure_cost_profile(
        vm_hourly_rate=3.25,
        dbu_hourly_rate=None,
        discount_rate=0.1,
        vat_rate=0.2,
        currency_code="USD",
    )

    assert cost.dbu_hourly_rate is None
    assert cost.estimated_hourly_rate == 3.25
    assert round(cost.vat_adjusted_hourly_rate or 0.0, 3) == 3.51
