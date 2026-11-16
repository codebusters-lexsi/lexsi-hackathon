"""
Unit tests for Gradio application components and UI helper functions.
"""

import unittest
import sys
from pathlib import Path

# Add app and src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import app


class TestAppIntegration(unittest.TestCase):

    def test_presets_exist_and_non_empty(self):
        self.assertGreaterEqual(len(app.SAMPLE_PRESETS), 3)
        for name, text in app.SAMPLE_PRESETS.items():
            self.assertTrue(len(text) > 50, f"Preset '{name}' is too short.")

    def test_simulate_extraction_function(self):
        sample_text = app.SAMPLE_PRESETS["Cessna 172S - Spongy Brake Caliper Squawk (ATA 32)"]
        base_out, base_b, align_out, align_b = app.simulate_extraction(sample_text)

        # Baseline output contains chatter
        self.assertIn("Sure", base_out)
        self.assertIn("Schema Validation Failed", base_b)

        # Aligned output is formatted with markdown json fences
        self.assertIn("```json", align_out)
        clean_json = align_out.split("```json")[-1].split("```")[0].strip()
        self.assertTrue(clean_json.startswith("{"))
        self.assertTrue(clean_json.endswith("}"))

        # Badge confirms strict schema validity
        self.assertIn("100% Valid", align_b)

    def test_calculate_roi_savings_function(self):
        # Test 50,000 logs
        closed, self_h, sav, time_h = app.calculate_roi_savings(50000)
        self.assertEqual(closed, "$171.25")
        self.assertIn("99.8% Savings", sav)
        self.assertEqual(self_h, "$0.35")


if __name__ == "__main__":
    unittest.main()
