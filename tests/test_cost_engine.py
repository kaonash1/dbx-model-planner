from __future__ import annotations

import unittest

from dbx_model_planner.config import AppConfig
from dbx_model_planner.domain.common import EstimateSource
from dbx_model_planner.engines.cost import build_cost_profile, compose_cost_profile


class BuildCostProfileTests(unittest.TestCase):
    """Test the low-level build_cost_profile function."""

    def test_both_rates_none_returns_empty_profile(self) -> None:
        cost = build_cost_profile(
            vm_hourly_rate=None,
            dbu_hourly_rate=None,
            discount_rate=0.0,
            vat_rate=0.0,
            currency_code="USD",
        )

        self.assertEqual(cost.currency_code, "USD")
        self.assertIsNone(cost.vm_hourly_rate)
        self.assertIsNone(cost.dbu_hourly_rate)
        self.assertIsNone(cost.estimated_hourly_rate)

    def test_vm_only_rate(self) -> None:
        cost = build_cost_profile(
            vm_hourly_rate=3.25,
            dbu_hourly_rate=None,
            discount_rate=0.0,
            vat_rate=0.0,
            currency_code="USD",
        )

        self.assertEqual(cost.vm_hourly_rate, 3.25)
        self.assertIsNone(cost.dbu_hourly_rate)
        self.assertEqual(cost.estimated_hourly_rate, 3.25)
        self.assertEqual(cost.discounted_hourly_rate, 3.25)
        self.assertEqual(cost.vat_adjusted_hourly_rate, 3.25)

    def test_combined_rates(self) -> None:
        cost = build_cost_profile(
            vm_hourly_rate=3.25,
            dbu_hourly_rate=0.75,
            discount_rate=0.0,
            vat_rate=0.0,
            currency_code="EUR",
        )

        self.assertEqual(cost.estimated_hourly_rate, 4.0)
        self.assertEqual(cost.currency_code, "EUR")

    def test_discount_applied(self) -> None:
        cost = build_cost_profile(
            vm_hourly_rate=10.0,
            dbu_hourly_rate=0.0,
            discount_rate=0.1,
            vat_rate=0.0,
            currency_code="USD",
        )

        self.assertEqual(cost.estimated_hourly_rate, 10.0)
        self.assertEqual(cost.discounted_hourly_rate, 9.0)
        self.assertEqual(cost.vat_adjusted_hourly_rate, 9.0)

    def test_vat_applied(self) -> None:
        cost = build_cost_profile(
            vm_hourly_rate=10.0,
            dbu_hourly_rate=0.0,
            discount_rate=0.0,
            vat_rate=0.2,
            currency_code="USD",
        )

        self.assertEqual(cost.estimated_hourly_rate, 10.0)
        self.assertEqual(cost.discounted_hourly_rate, 10.0)
        self.assertEqual(cost.vat_adjusted_hourly_rate, 12.0)

    def test_discount_and_vat_combined(self) -> None:
        cost = build_cost_profile(
            vm_hourly_rate=3.25,
            dbu_hourly_rate=0.75,
            discount_rate=0.1,
            vat_rate=0.2,
            currency_code="USD",
        )

        self.assertEqual(cost.estimated_hourly_rate, 4.0)
        self.assertAlmostEqual(cost.discounted_hourly_rate, 3.6, places=3)
        self.assertAlmostEqual(cost.vat_adjusted_hourly_rate, 4.32, places=3)

    def test_pricing_reference_and_source_passed_through(self) -> None:
        cost = build_cost_profile(
            vm_hourly_rate=1.0,
            dbu_hourly_rate=0.0,
            discount_rate=0.0,
            vat_rate=0.0,
            currency_code="USD",
            pricing_reference="test-ref",
            source=EstimateSource.USER_PROVIDED,
        )

        self.assertEqual(cost.pricing_reference, "test-ref")
        self.assertEqual(cost.source, EstimateSource.USER_PROVIDED)


class ComposeCostProfileTests(unittest.TestCase):
    """Test the config-aware compose_cost_profile wrapper."""

    def test_uses_config_rates(self) -> None:
        config = AppConfig()
        config.pricing.discount_rate = 0.15
        config.pricing.vat_rate = 0.19
        config.pricing.currency_code = "EUR"

        cost = compose_cost_profile(config, vm_hourly_rate=10.0, dbu_hourly_rate=2.0)

        self.assertEqual(cost.currency_code, "EUR")
        self.assertEqual(cost.estimated_hourly_rate, 12.0)
        expected_discounted = round(12.0 * (1.0 - 0.15), 4)
        self.assertEqual(cost.discounted_hourly_rate, expected_discounted)
        expected_vat = round(expected_discounted * (1.0 + 0.19), 4)
        self.assertEqual(cost.vat_adjusted_hourly_rate, expected_vat)
        self.assertEqual(cost.source, EstimateSource.INFERRED)

    def test_compose_with_none_rates(self) -> None:
        config = AppConfig()
        cost = compose_cost_profile(config, vm_hourly_rate=None, dbu_hourly_rate=None)

        self.assertIsNone(cost.estimated_hourly_rate)


if __name__ == "__main__":
    unittest.main()
