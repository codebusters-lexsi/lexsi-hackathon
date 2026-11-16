---
annotations_creators:
- synthetic-curatorkit
language:
- en
license: lsal-1.1
multilinguality:
- monolingual
size_categories:
- n<1K
source_datasets:
- original
tags:
- aviation
- maintenance
- structured-extraction
- curatorkit
- pydantic
- json-schema
task_categories:
- text2text-generation
- information-extraction
task_ids:
- structured-data-extraction
pretty_name: Lexsi Aviation Maintenance Records Extraction Corpus
---

# Dataset Card for Lexsi Aviation Maintenance Records Extraction Corpus

## Dataset Description

- **Homepage:** [Lexsi Labs Hackathon](https://hackathon.lexsi.ai)
- **Repository:** [Lexsi-Labs/CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT)
- **Point of Contact:** Hackathon Participant

### Dataset Summary

The **Lexsi Aviation Maintenance Records Extraction Corpus** is a quality-gated, provenance-backed dataset designed for training and evaluating LLMs on extracting structured, airworthiness-compliant JSON data from messy, handwritten aircraft maintenance tech logs, pilot squawk sheets, and maintenance work orders.

Civil aviation regulations (FAA FAR Part 43.9/43.11 and EASA Part-M) require that maintenance events, part replacements, and return-to-service certifications be strictly documented. Base LLMs frequently hallucinate serial numbers, fail JSON syntax, or omit statutory airworthiness statements. This corpus was generated and audited using **CuratorKIT** with grounded hallucination checks and PII redaction.

### Supported Tasks and Leaderboards

- `instruction_following`: Structured extraction of civil aircraft maintenance records into validated Pydantic JSON schemas (`AviationMaintenanceRecord`).

### Languages

- English (featuring civil aviation maintenance nomenclature, A&P abbreviations like `R&R`, `IAW AMM`, `Ops Check Sat`, `P/N`, `S/N`, and ATA Chapter codes).

---

## Dataset Structure

### Data Instances

An example sample looks as follows:

```json
{
  "id": "sample_0066",
  "source_uri": "aviation_techlogs/2024/entry_0066.txt",
  "task_type": "instruction_following",
  "instruction": "You are an expert FAA/EASA certified aviation maintenance records analyst...",
  "input": "LOG ENTRY -- REG: N734SP (Cessna 172S Skyhawk) AFTT: 1420.5 hrs\nTOTAL TIME: 1420.5 hrs | DATE: 2024-03-15\n...",
  "output": "```json\n{\n  \"aircraft\": {\n    \"tail_number\": \"N734SP\",\n    \"aircraft_model\": \"Cessna 172S Skyhawk\",\n    \"serial_number\": \"172S-10928\"\n  },\n  \"log_metadata\": {\n    \"log_date\": \"2024-03-15\",\n    \"total_time_in_service_hours\": 1420.5,\n    \"total_cycles_landings\": 1820\n  },\n  ...\n}\n```",
  "metadata": {
    "tail_number": "N734SP",
    "model": "Cessna 172S Skyhawk",
    "ata_chapter": "32",
    "pii_scrubbed": true,
    "verified_grounded": true,
    "sha256": "0fd14cbc4c33c972b062d04e9ca37073e021cfbc47b37d1f88325a04c86a2459"
  },
  "provenance_chain": [
    {
      "step_name": "synthesis_and_grounding",
      "step_version": "1.0.0",
      "timestamp": "2026-11-15T14:48:19+00:00",
      "status": "passed",
      "notes": {"pii_scrubbed": true, "schema_checked": true}
    }
  ]
}
```

### Data Splits

| Split | Number of Samples | Integrity Checksum (SHA-256) |
| :--- | :---: | :--- |
| **Train** | 140 | `data/curated/train.jsonl` |
| **Validation** | 30 | `data/curated/val.jsonl` |
| **Holdout Test** | 30 | `data/curated/test.jsonl` |

---

## Dataset Creation

### Curation with CuratorKIT

1. **Hygiene Stage:** Automated redaction of sensitive contact information (mechanic personal mobile numbers and dispatcher emails) using regex and Presidio pseudonyms (`[REDACTED_PHONE]`).
2. **Grounding Gate:** Cross-verified that every key extracted field (Tail Number, Serial Number, ATA Chapter Code, Part Numbers) strictly appeared in the raw input text. Un-grounded samples were rejected and logged into `rejected.jsonl`.
3. **Schema Gate:** Checked that every output adheres to the canonical `AviationMaintenanceRecord` Pydantic model with mathematical invariance and valid ISO dates.
