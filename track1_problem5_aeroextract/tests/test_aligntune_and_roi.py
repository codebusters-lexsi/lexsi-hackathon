"""
Unit tests for AlignTune training recipe configuration and Enterprise ROI economics.
"""

import unittest
import json
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aligntune_training import compute_roi_financial_model, generate_aligntune_recipe


class TestAlignTuneAndROI(unittest.TestCase):

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent

    def test_recipe_yaml_validity(self):
        recipe_path = self.root_dir / "recipes/aviation_tiny_aya_sft.yaml"
        self.assertTrue(recipe_path.exists(), "Recipe YAML must exist.")

        with open(recipe_path, "r") as f:
            recipe = yaml.safe_load(f)

        self.assertEqual(recipe["model"]["model_name_or_path"], "CohereLabs/tiny-aya-base-32K")
        self.assertEqual(recipe["training_type"], "sft")
        self.assertEqual(recipe["backend"], "trl")
        self.assertEqual(recipe["peft"]["rank"], 16)
        self.assertEqual(recipe["peft"]["alpha"], 32)
        self.assertIn("q_proj", recipe["peft"]["target_modules"])

    def test_roi_calculation_logic(self):
        roi = compute_roi_financial_model()
        self.assertIn("volume_tiers", roi)
        self.assertEqual(len(roi["volume_tiers"]), 3)

        # 10k logs check
        t10k = roi["volume_tiers"][0]
        self.assertEqual(t10k["document_volume"], "10,000")
        self.assertIn("99.0%", t10k["cost_reduction_percent"])
        self.assertEqual(t10k["self_hosted_tiny_aya_cost"], "$0.35")
        self.assertEqual(t10k["closed_source_api_cost"], "$34.25")

        # 1M logs check
        t1m = roi["volume_tiers"][2]
        self.assertEqual(t1m["document_volume"], "1,000,000")
        self.assertIn("99.8%", t1m["cost_reduction_percent"])
        self.assertEqual(t1m["net_savings_dollars"], "$3,418.52")

    def test_roi_file_integrity(self):
        roi_file = self.root_dir / "data/results/roi_economic_analysis.json"
        self.assertTrue(roi_file.exists())
        with open(roi_file, "r") as f:
            data = json.load(f)
        self.assertIn("NVIDIA L4", data["workload_profile"]["target_hardware"])


if __name__ == "__main__":
    unittest.main()
