"""
Unit tests for CuratorKIT pipeline integration, data splits, and Lexsi provenance standard.
"""

import unittest
import json
import hashlib
from pathlib import Path


class TestCuratorKitIntegration(unittest.TestCase):

    def setUp(self):
        self.curated_dir = Path(__file__).resolve().parent.parent / "data/curated"

    def test_curated_files_exist(self):
        train_file = self.curated_dir / "train.jsonl"
        val_file = self.curated_dir / "val.jsonl"
        test_file = self.curated_dir / "test.jsonl"
        manifest_file = self.curated_dir / "manifest.json"
        provenance_file = self.curated_dir / "lexsi_provenance.json"
        checksums_file = self.curated_dir / "checksums.txt"

        for p in [train_file, val_file, test_file, manifest_file, provenance_file, checksums_file]:
            self.assertTrue(p.exists(), f"Missing expected file: {p.name}")

    def test_split_counts(self):
        def count_lines(p):
            with open(p, "r") as f:
                return sum(1 for _ in f)

        self.assertEqual(count_lines(self.curated_dir / "train.jsonl"), 140)
        self.assertEqual(count_lines(self.curated_dir / "val.jsonl"), 30)
        self.assertEqual(count_lines(self.curated_dir / "test.jsonl"), 30)

    def test_lexsi_provenance_format(self):
        prov_file = self.curated_dir / "lexsi_provenance.json"
        with open(prov_file, "r") as f:
            prov = json.load(f)

        self.assertEqual(prov.get("schema"), "lexsi.provenance/1")
        self.assertEqual(prov.get("library"), "curatorkit")
        self.assertEqual(prov.get("method"), "curate")
        self.assertIn("created_at", prov)
        self.assertIn("params", prov)
        self.assertIn("pipeline_config_hash", prov["params"])

    def test_checksum_integrity(self):
        checksums_file = self.curated_dir / "checksums.txt"
        with open(checksums_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            self.assertEqual(len(parts), 2)
            expected_hash, rel_filename = parts
            file_path = self.curated_dir / rel_filename
            self.assertTrue(file_path.exists(), f"File {rel_filename} not found in {self.curated_dir}")

            # Verify actual SHA-256
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                hasher.update(f.read())
            self.assertEqual(hasher.hexdigest(), expected_hash, f"Hash mismatch for {rel_filename}")


if __name__ == "__main__":
    unittest.main()
