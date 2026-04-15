from __future__ import annotations

import sys
from pathlib import Path
import unittest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dbx_model_planner.collectors.databricks import DatabricksInventoryCollector
from dbx_model_planner.config import AppConfig
from dbx_model_planner.terminal_app import run_terminal_app


class TerminalAppTests(unittest.TestCase):
    def test_terminal_app_inventory_then_quit(self) -> None:
        inputs = iter(["1", "q"])
        outputs: list[str] = []

        exit_code = run_terminal_app(
            config=AppConfig(),
            inventory=DatabricksInventoryCollector().collect_snapshot(),
            input_fn=lambda _: next(inputs),
            output_fn=outputs.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(any("Compute profiles: 2" in output for output in outputs))
        self.assertEqual(outputs[-1], "Bye.")

    def test_terminal_app_model_fit_path(self) -> None:
        inputs = iter(["2", "2", "q"])
        outputs: list[str] = []

        exit_code = run_terminal_app(
            config=AppConfig(),
            inventory=DatabricksInventoryCollector().collect_snapshot(),
            input_fn=lambda _: next(inputs),
            output_fn=outputs.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(any("Recommended compute for mistralai/Mistral-7B-Instruct-v0.3" in output for output in outputs))


if __name__ == "__main__":
    unittest.main()
