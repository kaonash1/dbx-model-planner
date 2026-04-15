from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import re
from typing import Any
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen


AZURE_RETAIL_PRICES_ENDPOINT = "https://prices.azure.com/api/retail/prices"
DEFAULT_SERVICE_NAME = "Virtual Machines"
DEFAULT_PRICE_TYPE = "Consumption"

_LOCATION_TO_REGION = {
    "australiaeast": "australiaeast",
    "australiasoutheast": "australiasoutheast",
    "brazilsouth": "brazilsouth",
    "canadacentral": "canadacentral",
    "canadaeast": "canadaeast",
    "centralindia": "centralindia",
    "centralus": "centralus",
    "eastasia": "eastasia",
    "eastus": "eastus",
    "eastus2": "eastus2",
    "francecentral": "francecentral",
    "francesouth": "francesouth",
    "germanycentral": "germanywestcentral",
    "germanywestcentral": "germanywestcentral",
    "indiacentral": "centralindia",
    "indiacentral": "centralindia",
    "indiasouth": "southindia",
    "japaneast": "japaneast",
    "japanwest": "japanwest",
    "koreacentral": "koreacentral",
    "koreasouth": "koreasouth",
    "northcentralus": "northcentralus",
    "northeurope": "northeurope",
    "norwayeast": "norwayeast",
    "norwaywest": "norwaywest",
    "southafricanorth": "southafricanorth",
    "southafricawest": "southafricawest",
    "southcentralus": "southcentralus",
    "southeastasia": "southeastasia",
    "southindia": "southindia",
    "swedencentral": "swedencentral",
    "switzerlandnorth": "switzerlandnorth",
    "switzerlandwest": "switzerlandwest",
    "uaecentral": "uaecentral",
    "uaenorth": "uaenorth",
    "uksouth": "uksouth",
    "ukwest": "ukwest",
    "westcentralus": "westcentralus",
    "westeurope": "westeurope",
    "westindia": "westindia",
    "westus": "westus",
    "westus2": "westus2",
    "westus3": "westus3",
    "euwest": "westeurope",
    "euwest": "westeurope",
    "eunorth": "northeurope",
    "northeurope": "northeurope",
    "west europe": "westeurope",
    "eu west": "westeurope",
    "europe west": "westeurope",
    "eu north": "northeurope",
    "us east": "eastus",
    "us east 2": "eastus2",
    "us west": "westus",
    "us west 2": "westus2",
    "us west 3": "westus3",
    "us central": "centralus",
    "us south central": "southcentralus",
    "uk south": "uksouth",
    "uk west": "ukwest",
    "france central": "francecentral",
    "germany west central": "germanywestcentral",
    "japan east": "japaneast",
    "japan west": "japanwest",
    "canada central": "canadacentral",
    "canada east": "canadaeast",
    "india south": "southindia",
}


def normalize_azure_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or None


def normalize_azure_region(value: str | None) -> str | None:
    """Normalize Azure region and location labels into ARM region names."""

    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None

    token = normalize_azure_token(stripped)
    if token is None:
        return None

    if token in _LOCATION_TO_REGION:
        return _LOCATION_TO_REGION[token]

    compact = token.replace("_", "")
    if compact in _LOCATION_TO_REGION:
        return _LOCATION_TO_REGION[compact]

    return token


def normalize_azure_restrictions(value: object | None) -> list[str]:
    """Flatten arbitrary restriction payloads into normalized machine tokens."""

    if value is None:
        return []

    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[|,;/]+", value) if part.strip()]
        normalized = [normalize_azure_token(part) or part.lower() for part in parts]
        seen: set[str] = set()
        ordered: list[str] = []
        for item in normalized:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    if isinstance(value, Mapping):
        ordered: list[str] = []
        for key in ("reasonCode", "code", "restrictionType", "type", "description", "name"):
            raw = value.get(key)
            if raw is None:
                continue
            ordered.extend(normalize_azure_restrictions(raw))
        if ordered:
            seen: set[str] = set()
            unique: list[str] = []
            for item in ordered:
                if item not in seen:
                    seen.add(item)
                    unique.append(item)
            return unique
        return [normalize_azure_token(json.dumps(dict(value), sort_keys=True)) or "restriction"]

    if isinstance(value, Iterable):
        flattened: list[str] = []
        for item in value:
            flattened.extend(normalize_azure_restrictions(item))
        seen: set[str] = set()
        ordered: list[str] = []
        for item in flattened:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    token = normalize_azure_token(str(value))
    return [token] if token is not None else []


@dataclass(slots=True)
class AzureRetailPriceSavingsPlan:
    term: str
    unit_price: float | None = None
    retail_price: float | None = None


@dataclass(slots=True)
class AzureRetailPriceRecord:
    currency_code: str
    retail_price: float
    unit_price: float
    arm_region_name: str | None = None
    location: str | None = None
    effective_start_date: str | None = None
    meter_id: str | None = None
    meter_name: str | None = None
    product_id: str | None = None
    sku_id: str | None = None
    product_name: str | None = None
    sku_name: str | None = None
    service_name: str | None = None
    service_id: str | None = None
    service_family: str | None = None
    unit_of_measure: str | None = None
    meter_type: str | None = None
    is_primary_meter_region: bool | None = None
    arm_sku_name: str | None = None
    reservation_term: str | None = None
    savings_plan: list[AzureRetailPriceSavingsPlan] = field(default_factory=list)
    normalized_region: str | None = None
    normalized_location: str | None = None
    normalized_price_type: str | None = None
    normalized_restrictions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AzureRetailPricesPage:
    items: list[AzureRetailPriceRecord] = field(default_factory=list)
    next_page_link: str | None = None
    count: int | None = None
    billing_currency: str | None = None


@dataclass(slots=True)
class AzureRetailPriceQuery:
    arm_region_name: str | None = None
    arm_sku_names: Sequence[str] = field(default_factory=tuple)
    currency_code: str = "USD"
    service_name: str = DEFAULT_SERVICE_NAME
    service_family: str = "Compute"
    price_type: str | None = DEFAULT_PRICE_TYPE
    meter_region_primary_only: bool = True
    api_version: str | None = "2023-01-01-preview"
    extra_filters: Sequence[str] = field(default_factory=tuple)


def build_azure_retail_prices_url(query: AzureRetailPriceQuery) -> str:
    """Build an Azure Retail Prices API URL."""

    params: list[tuple[str, str]] = []
    if query.api_version:
        params.append(("api-version", query.api_version))
    if query.currency_code:
        params.append(("currencyCode", f"'{query.currency_code}'"))
    if query.meter_region_primary_only:
        params.append(("meterRegion", "'primary'"))

    filter_parts: list[str] = []
    if query.service_name:
        filter_parts.append(f"serviceName eq '{query.service_name}'")
    if query.service_family:
        filter_parts.append(f"serviceFamily eq '{query.service_family}'")
    if query.price_type:
        filter_parts.append(f"priceType eq '{query.price_type}'")
    if query.arm_region_name:
        filter_parts.append(f"armRegionName eq '{normalize_azure_region(query.arm_region_name)}'")
    if query.arm_sku_names:
        arm_sku_filters = [f"armSkuName eq '{sku}'" for sku in query.arm_sku_names]
        if len(arm_sku_filters) == 1:
            filter_parts.append(arm_sku_filters[0])
        else:
            filter_parts.append("(" + " or ".join(arm_sku_filters) + ")")
    if query.extra_filters:
        filter_parts.extend(query.extra_filters)
    if filter_parts:
        params.append(("$filter", " and ".join(filter_parts)))

    return f"{AZURE_RETAIL_PRICES_ENDPOINT}?{urlencode(params, quote_via=quote)}"


def _coerce_float(value: object | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _coerce_bool(value: object | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def parse_azure_retail_price_item(payload: Mapping[str, Any]) -> AzureRetailPriceRecord:
    """Parse a single Retail Prices API item into a structured record."""

    currency_code = str(payload.get("currencyCode") or payload.get("billingCurrency") or "USD")
    arm_region_name = payload.get("armRegionName")
    location = payload.get("location") or payload.get("Location")
    meter_type = payload.get("type") or payload.get("priceType")
    reservation_term = payload.get("reservationTerm")
    normalized_region = normalize_azure_region(str(arm_region_name)) if arm_region_name else None
    if normalized_region is None:
        normalized_region = normalize_azure_region(str(location)) if location else None
    normalized_location = normalize_azure_token(str(location)) if location else None
    normalized_price_type = normalize_azure_token(str(meter_type)) if meter_type else None

    normalized_restrictions: list[str] = []
    if normalized_price_type:
        normalized_restrictions.append(normalized_price_type)
    primary_region = _coerce_bool(payload.get("isPrimaryMeterRegion"))
    if primary_region:
        normalized_restrictions.append("primary_meter_region")
    if reservation_term:
        normalized_restrictions.append(f"reservation_{normalize_azure_token(str(reservation_term)) or 'term'}")
    if payload.get("savingsPlan"):
        normalized_restrictions.append("savings_plan")
    normalized_restrictions = normalize_azure_restrictions(normalized_restrictions)

    savings_plan: list[AzureRetailPriceSavingsPlan] = []
    for plan in payload.get("savingsPlan") or []:
        if not isinstance(plan, Mapping):
            continue
        savings_plan.append(
            AzureRetailPriceSavingsPlan(
                term=str(plan.get("term") or plan.get("reservationTerm") or ""),
                unit_price=plan.get("unitPrice") if plan.get("unitPrice") is None else _coerce_float(plan.get("unitPrice")),
                retail_price=plan.get("retailPrice") if plan.get("retailPrice") is None else _coerce_float(plan.get("retailPrice")),
            )
        )

    return AzureRetailPriceRecord(
        currency_code=currency_code,
        retail_price=_coerce_float(payload.get("retailPrice")),
        unit_price=_coerce_float(payload.get("unitPrice") or payload.get("retailPrice")),
        arm_region_name=str(arm_region_name) if arm_region_name is not None else None,
        location=str(location) if location is not None else None,
        effective_start_date=str(payload.get("effectiveStartDate")) if payload.get("effectiveStartDate") is not None else None,
        meter_id=str(payload.get("meterId")) if payload.get("meterId") is not None else None,
        meter_name=str(payload.get("meterName")) if payload.get("meterName") is not None else None,
        product_id=str(payload.get("productId") or payload.get("productid")) if payload.get("productId") or payload.get("productid") else None,
        sku_id=str(payload.get("skuId")) if payload.get("skuId") is not None else None,
        product_name=str(payload.get("productName")) if payload.get("productName") is not None else None,
        sku_name=str(payload.get("skuName")) if payload.get("skuName") is not None else None,
        service_name=str(payload.get("serviceName")) if payload.get("serviceName") is not None else None,
        service_id=str(payload.get("serviceId")) if payload.get("serviceId") is not None else None,
        service_family=str(payload.get("serviceFamily")) if payload.get("serviceFamily") is not None else None,
        unit_of_measure=str(payload.get("unitOfMeasure")) if payload.get("unitOfMeasure") is not None else None,
        meter_type=str(meter_type) if meter_type is not None else None,
        is_primary_meter_region=primary_region,
        arm_sku_name=str(payload.get("armSkuName")) if payload.get("armSkuName") is not None else None,
        reservation_term=str(reservation_term) if reservation_term is not None else None,
        savings_plan=savings_plan,
        normalized_region=normalized_region,
        normalized_location=normalized_location,
        normalized_price_type=normalized_price_type,
        normalized_restrictions=normalized_restrictions,
    )


def parse_azure_retail_prices_page(payload: Mapping[str, Any]) -> AzureRetailPricesPage:
    """Parse a Retail Prices API response page."""

    items_payload = payload.get("Items") or payload.get("items") or []
    items: list[AzureRetailPriceRecord] = []
    for item in items_payload:
        if isinstance(item, Mapping):
            items.append(parse_azure_retail_price_item(item))

    count = payload.get("Count")
    if count is None:
        count = payload.get("count")

    billing_currency = payload.get("BillingCurrency") or payload.get("billingCurrency")
    next_page_link = payload.get("NextPageLink") or payload.get("nextPageLink")

    return AzureRetailPricesPage(
        items=items,
        next_page_link=str(next_page_link) if next_page_link is not None else None,
        count=int(count) if count is not None else None,
        billing_currency=str(billing_currency) if billing_currency is not None else None,
    )


def _read_json(url: str, opener: Callable[[str, float], Any] | None, timeout: float) -> Mapping[str, Any]:
    if opener is None:
        request = Request(url, headers={"User-Agent": "dbx-model-planner/0.1"})
        with urlopen(request, timeout=timeout) as response:  # type: ignore[arg-type]
            payload = response.read()
    else:
        response = opener(url, timeout)
        if hasattr(response, "__enter__"):
            with response as opened:
                payload = opened.read()
        else:
            payload = response.read()

    if isinstance(payload, bytes):
        text = payload.decode("utf-8")
    else:
        text = str(payload)
    data = json.loads(text)
    if not isinstance(data, Mapping):
        raise TypeError("Azure Retail Prices API response was not a JSON object.")
    return data


def fetch_azure_retail_prices(
    query: AzureRetailPriceQuery,
    *,
    opener: Callable[[str, float], Any] | None = None,
    timeout: float = 30.0,
) -> list[AzureRetailPriceRecord]:
    """Fetch and paginate Azure Retail Prices API records."""

    url = build_azure_retail_prices_url(query)
    records: list[AzureRetailPriceRecord] = []

    while url:
        page = parse_azure_retail_prices_page(_read_json(url, opener, timeout))
        records.extend(page.items)
        if not page.next_page_link:
            break
        url = urljoin(AZURE_RETAIL_PRICES_ENDPOINT, page.next_page_link)

    return records


def select_azure_retail_price(
    prices: Sequence[AzureRetailPriceRecord],
    *,
    arm_region_name: str | None = None,
    arm_sku_names: Sequence[str] = (),
    price_type: str | None = DEFAULT_PRICE_TYPE,
) -> AzureRetailPriceRecord | None:
    """Pick the best Retail Prices record for a VM SKU and region."""

    normalized_region = normalize_azure_region(arm_region_name)
    sku_candidates = {normalize_azure_token(sku) or sku.lower() for sku in arm_sku_names}
    normalized_price_type = normalize_azure_token(price_type) if price_type else None

    def score(record: AzureRetailPriceRecord) -> tuple[int, float]:
        score_value = 0
        if normalized_price_type and record.normalized_price_type == normalized_price_type:
            score_value += 100
        if normalized_region and record.normalized_region == normalized_region:
            score_value += 50
        if sku_candidates and record.arm_sku_name:
            normalized_sku = normalize_azure_token(record.arm_sku_name) or record.arm_sku_name.lower()
            if normalized_sku in sku_candidates:
                score_value += 25
        if record.is_primary_meter_region:
            score_value += 10
        return score_value, record.retail_price

    candidates = list(prices)
    if normalized_region is not None:
        region_matches = [record for record in candidates if record.normalized_region == normalized_region]
        if region_matches:
            candidates = region_matches

    if sku_candidates:
        sku_matches = []
        for record in candidates:
            normalized_sku = normalize_azure_token(record.arm_sku_name) if record.arm_sku_name else None
            if normalized_sku and normalized_sku in sku_candidates:
                sku_matches.append(record)
        if sku_matches:
            candidates = sku_matches

    if normalized_price_type:
        type_matches = [record for record in candidates if record.normalized_price_type == normalized_price_type]
        if type_matches:
            candidates = type_matches

    if not candidates:
        return None

    return sorted(candidates, key=lambda record: (-score(record)[0], record.retail_price))[0]
