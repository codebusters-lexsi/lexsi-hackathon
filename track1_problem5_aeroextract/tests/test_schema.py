"""
Unit tests for Aviation Maintenance Pydantic v2 schemas.
Verifies data contracts, regulatory constraints, and validation boundaries.
"""

import unittest
from pydantic import ValidationError
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schema import (
    AircraftIdentification,
    LogMetadata,
    PartTracked,
    MaintenanceEvent,
    AirworthinessRelease,
    AviationMaintenanceRecord,
    ATA_CHAPTER_MAP,
)


class TestAviationSchemas(unittest.TestCase):

    def setUp(self):
        self.valid_data = {
            "aircraft": {
                "tail_number": "N734SP",
                "aircraft_model": "Cessna 172S Skyhawk",
                "serial_number": "172S-10492"
            },
            "log_metadata": {
                "log_date": "2024-03-15",
                "total_time_in_service_hours": 1420.5,
                "total_cycles_landings": 890
            },
            "maintenance_event": {
                "ata_chapter": "32",
                "event_type": "UNSCHEDULED_DEFECT",
                "pilot_discrepancy": "LH main landing gear brake spongy during rollout.",
                "corrective_action": "R&R LH brake caliper O-ring seal P/N MS28775-222."
            },
            "parts_replaced": [
                {
                    "part_name": "Caliper O-ring",
                    "part_number_off": "MS28775-222",
                    "part_number_on": "MS28775-222",
                    "serial_number_on": "LOT-94821",
                    "tracking_tag_type": "FAA Form 8130-3"
                }
            ],
            "airworthiness_release": {
                "return_to_service": True,
                "certifying_technician": "James R. Miller",
                "certificate_type": "A&P",
                "certificate_number": "AP-3489210IA",
                "certification_statement": "I certify this aircraft has been inspected and approved for return to service."
            }
        }

    def test_valid_record_instantiation(self):
        record = AviationMaintenanceRecord(**self.valid_data)
        self.assertEqual(record.aircraft.tail_number, "N734SP")
        self.assertEqual(record.maintenance_event.ata_chapter, "32")
        self.assertEqual(record.log_metadata.total_time_in_service_hours, 1420.5)
        self.assertTrue(record.airworthiness_release.return_to_service)

    def test_invalid_registration_number(self):
        bad_data = dict(self.valid_data)
        bad_data["aircraft"] = {
            "tail_number": "invalid_tail!",
            "aircraft_model": "Cessna 172",
            "serial_number": "123"
        }
        with self.assertRaises(ValidationError):
            AviationMaintenanceRecord(**bad_data)

    def test_invalid_ata_chapter(self):
        bad_data = dict(self.valid_data)
        bad_data["maintenance_event"] = {
            "ata_chapter": "INVALID_ATA",
            "event_type": "UNSCHEDULED_DEFECT",
            "pilot_discrepancy": "Something broke",
            "corrective_action": "Fixed it"
        }
        with self.assertRaises(ValidationError):
            AviationMaintenanceRecord(**bad_data)

    def test_negative_airframe_time(self):
        bad_data = dict(self.valid_data)
        bad_data["log_metadata"] = {
            "log_date": "2024-03-15",
            "total_time_in_service_hours": -10.5
        }
        with self.assertRaises(ValidationError):
            AviationMaintenanceRecord(**bad_data)

    def test_json_roundtrip(self):
        record = AviationMaintenanceRecord(**self.valid_data)
        json_str = record.model_dump_json()
        restored = AviationMaintenanceRecord.model_validate_json(json_str)
        self.assertEqual(record.model_dump(), restored.model_dump())


if __name__ == "__main__":
    unittest.main()
