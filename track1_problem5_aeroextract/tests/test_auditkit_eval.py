"""
Unit tests for AuditKIT evaluation module and scorecard generation.
"""

import unittest
import json
import sys
from pathlib import Path

# Add AuditKIT and local modules to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "AuditKIT/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import auditkit as ak
from auditkit.sample import Sample
from auditkit.metrics.structured import JsonSchemaValid, FieldF1
from auditkit.metrics.code import IsJson
from schema import AviationMaintenanceRecord
from auditkit_eval import get_aviation_json_schema, run_auditkit_benchmark


class TestAuditKitEval(unittest.TestCase):

    def test_json_schema_valid_metric(self):
        schema = get_aviation_json_schema()
        metric = JsonSchemaValid(schema=schema)

        # 1. Valid JSON sample
        valid_json = AviationMaintenanceRecord(
            aircraft={"tail_number": "N734SP", "aircraft_model": "Cessna 172S", "serial_number": "123"},
            log_metadata={"log_date": "2024-01-01", "total_time_in_service_hours": 100.0},
            maintenance_event={
                "ata_chapter": "32",
                "event_type": "UNSCHEDULED_DEFECT",
                "pilot_discrepancy": "Tire low",
                "corrective_action": "Serviced tire"
            },
            airworthiness_release={
                "return_to_service": True,
                "certifying_technician": "John Doe",
                "certificate_type": "A&P",
                "certificate_number": "AP12345",
                "certification_statement": "Returned to service"
            }
        ).model_dump_json()

        s_valid = Sample(input="test", target=valid_json)
        score_valid = metric.score(s_valid, valid_json)
        self.assertEqual(score_valid.value, 1.0)

        # 2. Invalid schema (missing required fields)
        invalid_json = json.dumps({"aircraft": "just a string"})
        s_invalid = Sample(input="test", target=valid_json)
        score_invalid = metric.score(s_invalid, invalid_json)
        self.assertEqual(score_invalid.value, 0.0)

    def test_field_f1_metric(self):
        metric = FieldF1(strip_fences=True)
        target = json.dumps({"a": 1, "b": "two", "c": [1, 2]})
        pred = json.dumps({"a": 1, "b": "two", "c": [1, 2]})
        sample = Sample(input="test", target=target)
        scores = metric.score(sample, pred)
        f1_score = next(s for s in scores if s.name == "field_f1")
        self.assertEqual(f1_score.value, 1.0)

    def test_is_json_metric(self):
        metric = IsJson()
        s = Sample(input="x")
        self.assertEqual(metric.score(s, '{"key": "value"}').value, 1.0)
        self.assertEqual(metric.score(s, '{key: "value", trailing_comma,}').value, 0.0)

    def test_scorecard_file_validity(self):
        scorecard_path = Path(__file__).resolve().parent.parent / "data/results/auditkit_scorecard.json"
        self.assertTrue(scorecard_path.exists(), "Scorecard JSON file must exist.")
        with open(scorecard_path, "r") as f:
            data = json.load(f)
        self.assertIn("baseline", data)
        self.assertIn("aligned", data)
        self.assertIn("deltas", data)
        self.assertEqual(data["aligned"]["schema_validation_pass_rate"], 100.0)
        self.assertEqual(data["aligned"]["json_syntax_pass_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
