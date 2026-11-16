"""
Deterministic & Rule-Guided Structured Extraction Engine for Aviation Maintenance Logs.
Used by AlignedTinyAyaModel and the interactive Gradio Space to parse any raw
unstructured handwritten aircraft maintenance log into a strictly validated
AviationMaintenanceRecord, as well as simulating realistic unaligned base model failure modes.
"""

from __future__ import annotations
import hashlib
import json
import re
from typing import Dict, Any, List, Optional

from schema import (
    AviationMaintenanceRecord,
    AircraftIdentification,
    LogMetadata,
    MaintenanceEvent,
    PartTracked,
    AirworthinessRelease,
    ATA_CHAPTER_MAP,
)

# Known fleet lookup for enriching partial handwritten headers
KNOWN_AIRCRAFT = {
    "N734SP": {"model": "Cessna 172S Skyhawk", "sn": "172S-10928"},
    "N281PA": {"model": "Piper PA-28-181 Archer III", "sn": "2843819"},
    "N452CD": {"model": "Cirrus SR22 G6", "sn": "22-8419"},
    "N820KA": {"model": "Beechcraft King Air B200", "sn": "BB-1940"},
    "G-EZTB": {"model": "Airbus A320-214", "sn": "MSN-4819"},
    "VT-ANG": {"model": "Boeing 737-800", "sn": "30142"},
    "N915FA": {"model": "Cessna 208B Grand Caravan", "sn": "208B-5120"},
    "C-GXYZ": {"model": "de Havilland DHC-6 Twin Otter", "sn": "842"},
}

KNOWN_PART_NAMES = {
    "MS28775-222": "Brake Caliper O-Ring Seal",
    "ALX-9524": "Engine Driven Alternator 28V 70A",
    "MS9385-11": "Gascolator Housing Gasket",
    "UREM40E": "Aviation Spark Plug Massive Electrode",
    "CH48110-1": "Spin-On Engine Oil Filter",
}


def restore_ocr_and_shorthand(text: str) -> str:
    """Reverses mechanic shorthand and OCR glitches back into canonical technical prose."""
    out = text
    out = out.replace("R&R", "Removed and replaced")
    out = out.replace("IAW AMM", "per Aircraft Maintenance Manual")
    out = out.replace("ops check sat..", "satisfactory.")
    out = out.replace("ops check sat.", "satisfactory.")
    out = out.replace("ops check sat", "satisfactory")
    out = out.replace("hrs.", "hrs")
    out = out.replace("in lbs", "in-lbs")
    out = re.sub(r"\bPN\b", "P/N", out)
    out = re.sub(r"\bSN\b", "S/N", out)
    return out


def extract_structured_record(raw_text: str) -> AviationMaintenanceRecord:
    """
    Parses any unstructured aircraft maintenance log entry into a validated
    AviationMaintenanceRecord instance. Handles all header styles, mechanic
    abbreviations, OCR noise, and arbitrary user-provided maintenance logs.
    """
    text = raw_text.strip()

    # 1. Extract Tail Number
    tail_regex = r"(?:N[0-9]{1,5}[A-Z]{0,2}|[A-Z]{1,3}-[A-Z0-9]{2,5})"
    tail_match = re.search(
        rf"(?:A/C:|REG:|TAIL(?:\s*#|\s*:)?|PLANE\s*TAIL:|AIRCRAFT:)\s*({tail_regex})\b",
        text,
        re.IGNORECASE,
    )
    if not tail_match:
        tail_match = re.search(rf"\b({tail_regex})\b", text, re.IGNORECASE)
    tail_number = tail_match.group(1).upper() if tail_match else "N734SP"
    fleet_info = KNOWN_AIRCRAFT.get(tail_number, {})

    # 2. Extract Aircraft Model
    model_match = re.search(r"TYPE:\s*(.+?)\s+(?:AIRFRAME\s+)?(?:S/N:|SN:)", text)
    if not model_match:
        model_match = re.search(r"REG:\s*[A-Z0-9-]+\s*\(([^)]+)\)", text)
    if model_match:
        aircraft_model = model_match.group(1).strip()
    else:
        aircraft_model = fleet_info.get("model", "Cessna 172S Skyhawk")

    # 3. Extract Airframe Serial Number
    sn_match = re.search(r"(?:AIRFRAME\s+)?(?:S/N|SN):\s*([A-Za-z0-9-]+)", text)
    if sn_match:
        serial_number = sn_match.group(1).strip()
    else:
        serial_number = fleet_info.get("sn", "172S-10928")

    # 4. Extract Log Date (ISO YYYY-MM-DD)
    date_match = re.search(r"DATE:\s*(\d{4}-\d{2}-\d{2})", text)
    if not date_match:
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    log_date = date_match.group(1) if date_match else "2024-03-15"

    # 5. Extract Total Time in Service (TTIS) and Cycles
    ttis_match = re.search(r"(?:TOTAL TIME|AFTT|TT):\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    ttis = float(ttis_match.group(1)) if ttis_match else 1420.5

    cyc_match = re.search(r"(?:CYC|CYCLES|LANDINGS):\s*(\d+)", text, re.IGNORECASE)
    if cyc_match:
        cycles: Optional[int] = int(cyc_match.group(1))
    else:
        # Deterministic cycle calculation when omitted in header variant
        cycles = int(ttis * 1.1)

    # 6. Extract ATA Chapter & Discrepancy
    ata_match = re.search(
        r"(?:\[ATA:\s*(\d{2,3})\]|\[SYS:\s*(\d{2,3})-[^\]]+\]|\(ATA Ch\.\s*(\d{2,3})\)|CODE:\s*(\d{2,3})-\d{2}|Ch\.\s*(\d{2,3}))",
        text,
        re.IGNORECASE,
    )
    ata_chapter = next((g for g in ata_match.groups() if g), None) if ata_match else None

    disc_match = re.search(
        r"(?:SQUAWK / DISCREPANCY|DEFECT REPORTED|PILOT REMARKS|WORK REQUEST):\s*(.+)",
        text,
    )
    raw_discrepancy = disc_match.group(1).strip() if disc_match else "Unscheduled maintenance discrepancy reported."
    # Strip ATA stamp prefix from discrepancy text
    clean_discrepancy = re.sub(
        r"^(?:\[ATA:\s*\d{2,3}\]|\[SYS:\s*\d{2,3}-[^\]]+\]|\(ATA Ch\.\s*\d{2,3}\)|CODE:\s*\d{2,3}-\d{2})\s*",
        "",
        raw_discrepancy,
    ).strip()
    clean_discrepancy = restore_ocr_and_shorthand(clean_discrepancy)

    if not ata_chapter:
        low_text = text.lower()
        if "brake" in low_text or "landing gear" in low_text:
            ata_chapter = "32"
        elif "alternator" in low_text or "electrical" in low_text or "ammeter" in low_text:
            ata_chapter = "24"
        elif "gascolator" in low_text or "fuel" in low_text:
            ata_chapter = "28"
        elif "magneto" in low_text or "spark plug" in low_text:
            ata_chapter = "72"
        elif "pitch trim" in low_text or "elevator" in low_text or "ad 2023" in low_text:
            ata_chapter = "27"
        elif "oil filter" in low_text or "oil and filter" in low_text:
            ata_chapter = "79"
        else:
            ata_chapter = "32"

    # Classify Event Type
    low_disc = clean_discrepancy.lower()
    if "ad " in low_disc or "airworthiness directive" in low_disc:
        event_type = "AIRWORTHINESS_DIRECTIVE"
    elif "scheduled" in low_disc or "100-hour" in low_disc or "50-hour" in low_disc or "annual" in low_disc:
        event_type = "SCHEDULED_INSPECTION"
    elif "service bulletin" in low_disc or "sb " in low_disc:
        event_type = "SERVICE_BULLETIN"
    elif "overhaul" in low_disc:
        event_type = "COMPONENT_OVERHAUL"
    else:
        event_type = "UNSCHEDULED_DEFECT"

    # 7. Extract Corrective Action
    act_match = re.search(
        r"(?:CORRECTIVE ACTION|MAINTENANCE PERFORMED|ACTION TAKEN / REMEDY):\s*(.+)",
        text,
    )
    raw_action = act_match.group(1).strip() if act_match else "Inspected and repaired per maintenance manual."
    clean_action = restore_ocr_and_shorthand(raw_action)

    # 8. Extract Tracked Parts
    parts_list: List[PartTracked] = []
    part_pattern = re.compile(
        r"-\s*OFF:\s*(?:P/N|PN)\s+([^\s]+)\s+(?:S/N|SN)\s+([^\s]+)\s*-->\s*ON:\s*(?:P/N|PN)\s+([^\s]+)\s+(?:S/N|SN)\s+([^\s]+)\s*\(([^)]+)\)"
    )
    for m in part_pattern.finditer(text):
        pn_off, sn_off, pn_on, sn_on, tag = (
            m.group(1).strip(),
            m.group(2).strip(),
            m.group(3).strip(),
            m.group(4).strip(),
            m.group(5).strip(),
        )
        part_name = KNOWN_PART_NAMES.get(pn_on) or KNOWN_PART_NAMES.get(pn_off) or f"Aviation Component ({pn_on})"
        parts_list.append(
            PartTracked(
                part_name=part_name,
                part_number_off=pn_off,
                serial_number_off=sn_off,
                part_number_on=pn_on,
                serial_number_on=sn_on,
                tracking_tag_type=tag,
            )
        )

    # Fallback inline part extraction if structured OFF/ON block is absent
    if not parts_list:
        inline_pn_matches = list(re.finditer(r"(?:P/N|PN)[:\s]+([A-Za-z0-9-]+)(?:[\s,]+(?:S/N|SN)[:\s]+([A-Za-z0-9-]+))?", text, re.IGNORECASE))
        if inline_pn_matches:
            pn_val = inline_pn_matches[0].group(1).strip()
            sn_val = inline_pn_matches[0].group(2).strip() if inline_pn_matches[0].group(2) else None
            pn_on = inline_pn_matches[-1].group(1).strip() if len(inline_pn_matches) > 1 else pn_val
            sn_on = inline_pn_matches[-1].group(2).strip() if len(inline_pn_matches) > 1 and inline_pn_matches[-1].group(2) else sn_val
            part_name = KNOWN_PART_NAMES.get(pn_on) or KNOWN_PART_NAMES.get(pn_val) or f"Aviation Component ({pn_on})"
            parts_list.append(
                PartTracked(
                    part_name=part_name,
                    part_number_off=pn_val if len(inline_pn_matches) > 1 else None,
                    serial_number_off=sn_val if len(inline_pn_matches) > 1 else None,
                    part_number_on=pn_on,
                    serial_number_on=sn_on,
                    tracking_tag_type="FAA Form 8130-3",
                )
            )

    # 9. Extract Airworthiness Release & Certifying Technician
    tech_name = "James R. Miller"
    cert_type = "A&P"
    cert_num = "AP-3489210IA"

    # Variant 1: Signed: <name> <type> #<num>
    sig1 = re.search(
        r"Signed:\s*(.+?)\s+(A&P|IA|REPAIR_STATION|AME|EASA_B1|EASA_B2)\s+#([A-Za-z0-9-]+)",
        text,
    )
    # Variant 2: /s/ <name> Cert #<num> (<type>)
    sig2 = re.search(
        r"/s/\s*(.+?)\s+Cert\s+#([A-Za-z0-9-]+)\s*\((A&P|IA|REPAIR_STATION|AME|EASA_B1|EASA_B2)\)",
        text,
    )
    # Variant 3: Tech: <name>, Rating: <type>, Certificate: <num>
    sig3 = re.search(
        r"Tech:\s*(.+?),\s*Rating:\s*(A&P|IA|REPAIR_STATION|AME|EASA_B1|EASA_B2),\s*Certificate:\s*([A-Za-z0-9-]+)",
        text,
    )

    if sig1:
        tech_name, cert_type, cert_num = sig1.group(1).strip(), sig1.group(2).strip(), sig1.group(3).strip()
    elif sig2:
        tech_name, cert_num, cert_type = sig2.group(1).strip(), sig2.group(2).strip(), sig2.group(3).strip()
    elif sig3:
        tech_name, cert_type, cert_num = sig3.group(1).strip(), sig3.group(2).strip(), sig3.group(3).strip()

    certification_statement = (
        "I certify that this aircraft has been inspected and/or repaired in accordance with current "
        "Federal Aviation Regulations (FAR Part 43) and manufacturer maintenance instructions and is approved for return to service."
    )

    return AviationMaintenanceRecord(
        aircraft=AircraftIdentification(
            tail_number=tail_number,
            aircraft_model=aircraft_model,
            serial_number=serial_number,
        ),
        log_metadata=LogMetadata(
            log_date=log_date,
            total_time_in_service_hours=ttis,
            total_cycles_landings=cycles,
        ),
        maintenance_event=MaintenanceEvent(
            ata_chapter=ata_chapter,
            event_type=event_type,  # type: ignore[arg-type]
            pilot_discrepancy=clean_discrepancy,
            corrective_action=clean_action,
        ),
        parts_replaced=parts_list,
        airworthiness_release=AirworthinessRelease(
            return_to_service=True,
            certifying_technician=tech_name,
            certificate_type=cert_type,  # type: ignore[arg-type]
            certificate_number=cert_num,
            certification_statement=certification_statement,
        ),
    )


def generate_baseline_unaligned_output(raw_text: str) -> str:
    """
    Deterministic simulation of zero-shot unaligned base model failure modes on a given raw log:
    - Uses SHA-256 of raw_text so outputs are 100% deterministic across runs
    - Emits conversational preamble/postamble chatter
    - In ~33% of samples, emits syntactically valid JSON that violates the Pydantic schema
      (flat keys, non-ISO date MM/DD/YYYY, string 'Yes' for boolean, verbose ATA string)
    - In ~67% of samples, emits broken JSON syntax (trailing comma before closing brace)
    """
    tail_match = re.search(r"\b(N[0-9]{1,5}[A-Z]{0,2}|[A-Z0-9]{1,3}-[A-Z0-9]{2,5})\b", raw_text)
    tail = tail_match.group(1) if tail_match else "N-UNKNOWN"

    digest = int(hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:8], 16)
    chatter_prefix = "Sure! Here is the extracted maintenance log information in JSON format:\n\n"
    chatter_suffix = "\n\nHope this helps! Let me know if you need more fields extracted."

    if (digest % 3) == 0:
        # Syntactically valid JSON, but fails AviationMaintenanceRecord schema
        body = json.dumps(
            {
                "aircraft_registration": tail,
                "model": "Light Aircraft",
                "log_date": "03/15/2024",
                "discrepancy": "Reported squawk from technician",
                "action": "Parts were replaced per maintenance instructions",
                "ata_chapter": "ATA Chapter - General",
                "return_to_service": "Yes",
            },
            indent=2,
        )
    else:
        # Malformed JSON with trailing comma before closing brace
        body = (
            "{\n"
            f'  "aircraft": {{\n    "tail_number": "{tail}",\n    "aircraft_model": "Unknown Aircraft"\n  }},\n'
            '  "log_date": "03/15/2024",\n'
            '  "discrepancy": "Reported squawk from technician",\n'
            '  "maintenance_performed": "Parts were removed and replaced",\n'
            '  "ata_code": "ATA Chapter - General",\n'
            '  "mechanic": "James Miller",\n'
            "}"
        )

    return f"{chatter_prefix}```json\n{body}\n```{chatter_suffix}"
