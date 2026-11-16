"""
Aviation Maintenance & Airworthiness Record Schema
Strict Pydantic models for civil and commercial aircraft technical logbooks.
"""

from __future__ import annotations
import re
from datetime import date
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class AircraftIdentification(BaseModel):
    tail_number: str = Field(
        ...,
        description="Aircraft registration / tail number (e.g., N172SP, G-EZTB, VT-ANG, C-GXYZ)",
        examples=["N734SP", "G-BNLN", "VT-EXQ"]
    )
    aircraft_model: str = Field(
        ...,
        description="Aircraft make and model (e.g., Cessna 172S Skyhawk, Piper PA-28, Boeing 737-800)",
        examples=["Cessna 172S Skyhawk", "King Air B200", "Boeing 737-800"]
    )
    serial_number: str = Field(
        ...,
        description="Airframe manufacturer serial number",
        examples=["172S-10928", "BB-1940", "30142"]
    )

    @field_validator("tail_number")
    @classmethod
    def validate_tail_number(cls, v: str) -> str:
        v_clean = v.strip().upper()
        # Civil aviation registration pattern (US N-numbers or international Dash-separated)
        pattern = r"^(N[0-9]{1,5}[A-Z]{0,2}|[A-Z0-9]{1,3}-[A-Z0-9]{2,5})$"
        if not re.match(pattern, v_clean):
            raise ValueError(f"Invalid aircraft registration/tail number format: '{v_clean}'")
        return v_clean


class LogMetadata(BaseModel):
    log_date: str = Field(
        ...,
        description="Date of maintenance log entry in YYYY-MM-DD format",
        examples=["2024-03-15", "2023-11-20"]
    )
    total_time_in_service_hours: float = Field(
        ...,
        ge=0.0,
        description="Total Airframe Hours / Total Time in Service (TTIS / AFTT)",
        examples=[1420.5, 8740.2]
    )
    total_cycles_landings: Optional[int] = Field(
        None,
        ge=0,
        description="Total flight cycles or airframe landings",
        examples=[1820, 6420]
    )

    @field_validator("log_date")
    @classmethod
    def validate_iso_date(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Date must be in strict ISO format YYYY-MM-DD, got: '{v}'")
        return v


class PartTracked(BaseModel):
    part_name: str = Field(..., description="Nomenclature of replaced/serviced aviation component")
    part_number_off: Optional[str] = Field(None, description="Part Number (P/N) removed")
    serial_number_off: Optional[str] = Field(None, description="Serial Number (S/N) removed")
    part_number_on: Optional[str] = Field(None, description="Part Number (P/N) installed")
    serial_number_on: Optional[str] = Field(None, description="Serial Number (S/N) installed")
    tracking_tag_type: Optional[str] = Field(
        None,
        description="Release certificate tag type (e.g. FAA Form 8130-3, EASA Form 1, C of C)",
        examples=["FAA Form 8130-3", "EASA Form 1", "Certificate of Conformance"]
    )


class MaintenanceEvent(BaseModel):
    ata_chapter: str = Field(
        ...,
        description="Standard 2-digit Air Transport Association (ATA) chapter code",
        examples=["24", "28", "32", "72", "27", "34"]
    )
    event_type: Literal[
        "SCHEDULED_INSPECTION",
        "UNSCHEDULED_DEFECT",
        "AIRWORTHINESS_DIRECTIVE",
        "SERVICE_BULLETIN",
        "COMPONENT_OVERHAUL"
    ] = Field(..., description="Categorization of maintenance event")
    pilot_discrepancy: str = Field(
        ...,
        description="Pilot or technician reported squawk / maintenance discrepancy"
    )
    corrective_action: str = Field(
        ...,
        description="Detailed description of corrective maintenance procedure performed"
    )

    @field_validator("ata_chapter")
    @classmethod
    def validate_ata_chapter(cls, v: str) -> str:
        clean = v.strip().split("-")[0]
        if not (clean.isdigit() and len(clean) in (2, 3)):
            raise ValueError(f"Invalid ATA Chapter code: '{v}'. Must be a 2 or 3 digit ATA identifier.")
        return clean


class AirworthinessRelease(BaseModel):
    return_to_service: bool = Field(
        ...,
        description="Airworthiness certification status: True if aircraft/component is approved for return to service"
    )
    certifying_technician: str = Field(
        ...,
        description="Full legal name of the licensed certifying technician or inspector"
    )
    certificate_type: Literal[
        "A&P",
        "IA",
        "REPAIR_STATION",
        "AME",
        "EASA_B1",
        "EASA_B2"
    ] = Field(..., description="FAA/EASA/ICAO rating or certificate type held")
    certificate_number: str = Field(
        ...,
        description="License / Airframe & Powerplant certificate number"
    )
    certification_statement: str = Field(
        ...,
        description="Statutory sign-off statement approving work and airworthiness"
    )


class AviationMaintenanceRecord(BaseModel):
    """
    Complete root schema for an extracted civil aviation technical log entry.
    All fields must strictly conform to civil aviation regulatory definitions (FAA FAR 43.9 / 43.11).
    """
    aircraft: AircraftIdentification
    log_metadata: LogMetadata
    maintenance_event: MaintenanceEvent
    parts_replaced: List[PartTracked] = Field(default_factory=list)
    airworthiness_release: AirworthinessRelease

    @model_validator(mode="after")
    def verify_airworthiness_consistency(self) -> AviationMaintenanceRecord:
        if self.airworthiness_release.return_to_service:
            if not self.airworthiness_release.certificate_number.strip():
                raise ValueError("Approved return to service requires a valid certificate number.")
            if not self.airworthiness_release.certification_statement.strip():
                raise ValueError("Approved return to service requires an explicit certification statement.")
        return self


# Canonical ATA Chapter reference dictionary for provenance and grounding checks
ATA_CHAPTER_MAP = {
    "21": "Air Conditioning & Pressurization",
    "22": "Auto Flight",
    "23": "Communications",
    "24": "Electrical Power",
    "25": "Equipment & Furnishings",
    "27": "Flight Controls",
    "28": "Fuel Systems",
    "29": "Hydraulic Power",
    "30": "Ice and Rain Protection",
    "31": "Indicating / Recording Systems",
    "32": "Landing Gear & Brakes",
    "33": "Lights",
    "34": "Navigation",
    "35": "Oxygen",
    "36": "Pneumatic",
    "38": "Water / Waste",
    "49": "Airborne Auxiliary Power (APU)",
    "52": "Doors",
    "53": "Fuselage Structure",
    "55": "Stabilizers",
    "56": "Windows",
    "57": "Wings",
    "71": "Powerplant General",
    "72": "Engine - Turbine / Turboprop / Reciprocating",
    "73": "Engine Fuel & Control",
    "74": "Ignition",
    "77": "Engine Indicating",
    "78": "Exhaust",
    "79": "Engine Oil System"
}
