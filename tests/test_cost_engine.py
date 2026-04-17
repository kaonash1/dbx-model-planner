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
        # Discount applies to the whole total (VM + DBU)
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
        # VAT applies to the discounted total
        self.assertEqual(cost.vat_adjusted_hourly_rate, 12.0)

    def test_discount_and_vat_combined(self) -> None:
        cost = build_cost_profile(
            vm_hourly_rate=3.25,
            dbu_hourly_rate=0.75,
            discount_rate=0.1,
            vat_rate=0.2,
            currency_code="USD",
        )

        # total = 3.25 + 0.75 = 4.0
        self.assertEqual(cost.estimated_hourly_rate, 4.0)
        # discounted = 4.0 * 0.9 = 3.6
        self.assertAlmostEqual(cost.discounted_hourly_rate, 3.6, places=3)
        # vat = 3.6 * 1.2 = 4.32
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

    def test_dbu_only_no_vm(self) -> None:
        """DBU-only cost (VM=None) should still compute correctly."""
        cost = build_cost_profile(
            vm_hourly_rate=None,
            dbu_hourly_rate=5.50,
            discount_rate=0.375,
            vat_rate=0.19,
            currency_code="USD",
        )

        self.assertIsNone(cost.vm_hourly_rate)
        self.assertEqual(cost.dbu_hourly_rate, 5.50)
        self.assertEqual(cost.estimated_hourly_rate, 5.50)
        # discounted = 5.50 * 0.625 = 3.4375
        self.assertAlmostEqual(cost.discounted_hourly_rate, 3.4375, places=4)
        # vat = 3.4375 * 1.19 = 4.090625
        self.assertAlmostEqual(cost.vat_adjusted_hourly_rate, 4.0906, places=4)

    def test_full_discount_zeroes_cost(self) -> None:
        """100% discount should produce zero discounted and zero VAT-adjusted."""
        cost = build_cost_profile(
            vm_hourly_rate=10.0,
            dbu_hourly_rate=5.0,
            discount_rate=1.0,
            vat_rate=0.19,
            currency_code="USD",
        )

        self.assertEqual(cost.estimated_hourly_rate, 15.0)
        self.assertEqual(cost.discounted_hourly_rate, 0.0)
        self.assertEqual(cost.vat_adjusted_hourly_rate, 0.0)


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
        # Discount and VAT apply to the whole total (VM + DBU)
        # discounted = 12.0 * (1 - 0.15) = 10.2
        expected_discounted = round(12.0 * (1.0 - 0.15), 4)
        self.assertEqual(cost.discounted_hourly_rate, expected_discounted)
        # vat = 10.2 * (1 + 0.19) = 12.138
        expected_vat = round(expected_discounted * (1.0 + 0.19), 4)
        self.assertEqual(cost.vat_adjusted_hourly_rate, expected_vat)
        self.assertEqual(cost.source, EstimateSource.INFERRED)

    def test_compose_with_none_rates(self) -> None:
        config = AppConfig()
        cost = compose_cost_profile(config, vm_hourly_rate=None, dbu_hourly_rate=None)

        self.assertIsNone(cost.estimated_hourly_rate)

    def test_default_config_discount_and_vat(self) -> None:
        """AppConfig defaults must use 37.5% discount and 19% VAT."""
        config = AppConfig()

        self.assertEqual(config.pricing.discount_rate, 0.375)
        self.assertEqual(config.pricing.vat_rate, 0.19)
        self.assertEqual(config.pricing.currency_code, "USD")

        # Verify the defaults produce correct results
        cost = compose_cost_profile(config, vm_hourly_rate=19.108, dbu_hourly_rate=24.20)

        # total = 43.308
        self.assertAlmostEqual(cost.estimated_hourly_rate, 43.308, places=2)
        # discounted = 43.308 * 0.625 = 27.0675
        self.assertAlmostEqual(cost.discounted_hourly_rate, 27.0675, places=2)
        # vat = 27.0675 * 1.19 = 32.2103
        self.assertAlmostEqual(cost.vat_adjusted_hourly_rate, 32.2103, places=2)

    def test_nc96ads_a100_v4_real_world_pricing(self) -> None:
        """Validate the NC96ads A100 v4 pricing against colleague's verified numbers.

        VM price = $19.108/hr (Azure Retail Prices API, eastus)
        DBU count = 44, per-DBU rate = $0.55 → DBU price = $24.20/hr
        Total = $43.308/hr (matches Databricks pricing page $43.30)
        Discount = 37.5% on the whole total
        VAT = 19% on the discounted total
        """
        cost = build_cost_profile(
            vm_hourly_rate=19.108,
            dbu_hourly_rate=24.20,  # 44 DBU × $0.55
            discount_rate=0.375,
            vat_rate=0.19,
            currency_code="USD",
        )

        # Total = VM + DBU = 19.108 + 24.20 = 43.308
        self.assertAlmostEqual(cost.estimated_hourly_rate, 43.308, places=2)
        # Discounted = 43.308 × (1 - 0.375) = 27.0675
        self.assertAlmostEqual(cost.discounted_hourly_rate, 27.0675, places=2)
        # + VAT = 27.0675 × 1.19 = 32.2103
        self.assertAlmostEqual(cost.vat_adjusted_hourly_rate, 32.2103, places=2)


if __name__ == "__main__":
    unittest.main()
