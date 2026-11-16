"""
Unit tests for the Aviation Record Extraction Engine.
Tests handling of handwritten OCR noise, margin notes, ATA mapping, and error resilience.
"""

import unittest
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from extractor import extract_structured_record, generate_baseline_unaligned_output
from schema import AviationMaintenanceRecord


class TestExtractorEngine(unittest.TestCase):

    def test_sample_1_cessna_extraction(self):
        text = (
            "LOG ENTRY -- REG: N734SP (Cessna 172S Skyhawk) AFTT: 1420.5 hrs\n"
            "TOTAL TIME: 1420.5 hrs  |  DATE: 2024-03-15\n"
            "SQUAWK: [ATA: 32] LH main landing gear brake spongy during rollout.\n"
            "ACTION: Jacked aircraft. R&R LH brake caliper O-ring seal P/N MS28775-222.\n"
            "PARTS:\n"
            "  - OFF: P/N MS28775-222 S/N N/A  -->  ON: P/N MS28775-222 S/N LOT-94821 (FAA Form 8130-3)\n"
            "Return to service approved. Signed: James R. Miller A&P #AP-3489210IA"
        )
        record = extract_structured_record(text)
        self.assertIsInstance(record, AviationMaintenanceRecord)
        self.assertEqual(record.aircraft.tail_number, "N734SP")
        self.assertEqual(record.maintenance_event.ata_chapter, "32")
        self.assertGreaterEqual(len(record.parts_replaced), 1)
        self.assertEqual(record.parts_replaced[0].serial_number_on, "LOT-94821")
        self.assertIn("Miller", record.airworthiness_release.certifying_technician)

    def test_sample_2_piper_extraction(self):
        text = (
            "A/C: N281PA  TYPE: Piper PA-28-181 Archer III  S/N: 2843819\n"
            "TOTAL TIME: 3840.2 hrs  |  DATE: 2024-05-18\n"
            "DEFECT: [SYS: 24-ELECTRIC] Alternator inop in flight.\n"
            "REMEDY: R&R alternator P/N ALX-9524 S/N 49182. Installed overhauled P/N ALX-9524 S/N 84102.\n"
            "Aircraft returned to service. /s/ Sarah L. Jenkins Cert #IA-4019283"
        )
        record = extract_structured_record(text)
        self.assertEqual(record.aircraft.tail_number, "N281PA")
        self.assertEqual(record.maintenance_event.ata_chapter, "24")
        self.assertGreaterEqual(len(record.parts_replaced), 1)
        self.assertEqual(record.parts_replaced[0].part_number_on, "ALX-9524")
        self.assertEqual(record.parts_replaced[0].serial_number_on, "84102")

    def test_noisy_and_shorthand_robustness(self):
        dirty_text = (
            "dirty grease smudge\n"
            "plane tail: n734sp (turboprop) tt: 550.0 hrs 2024-11-01\n"
            "squawk: ata 28 fuel leak at strainer valve\n"
            "action taken: disassembled valve, flushed line iaw amm ops check sat.\n"
            "inspected by test tech #AP-9999999\n"
            "[margin: stained paper]"
        )
        record = extract_structured_record(dirty_text)
        self.assertIsInstance(record, AviationMaintenanceRecord)
        self.assertEqual(record.aircraft.tail_number, "N734SP")
        self.assertEqual(record.maintenance_event.ata_chapter, "28")
        self.assertEqual(record.log_metadata.total_time_in_service_hours, 550.0)

    def test_baseline_unaligned_behavior(self):
        text = "REG: N734SP AFTT: 1420.5 hrs DATE: 2024-03-15 SQUAWK: spongy brake"
        base_output = generate_baseline_unaligned_output(text)
        # Base model produces conversational chatter
        self.assertTrue(base_output.startswith("Sure! Here is the extracted JSON") or "Sure" in base_output)
        # Base model output fails strict Pydantic validation
        with self.assertRaises(Exception):
            clean_json = base_output.split("```json")[-1].split("```")[0].strip()
            AviationMaintenanceRecord.model_validate_json(clean_json)


if __name__ == "__main__":
    unittest.main()
