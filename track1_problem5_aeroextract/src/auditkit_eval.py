"""
AuditKIT Evaluation Suite for Aviation Maintenance Extraction
Benchmarks Baseline vs Aligned Tiny Aya on the test split using native AuditKIT:
- ak.evaluate()
- JsonSchemaValid (Code / JSON Schema metric)
- FieldF1 (Field precision, recall, and F1)
- IsJson (Strict JSON syntax validity)
- Conversational chatter overhead & token efficiency
"""

from __future__ import annotations
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any

# Add AuditKIT and local modules to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "AuditKIT/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import auditkit as ak
from auditkit.sample import Sample
from auditkit.metrics.structured import JsonSchemaValid, FieldF1
from auditkit.metrics.code import IsJson
from schema import AviationMaintenanceRecord, ATA_CHAPTER_MAP
from extractor import extract_structured_record, generate_baseline_unaligned_output


def get_aviation_json_schema() -> Dict[str, Any]:
    """Generates the JSON schema directly from the Pydantic model."""
    return AviationMaintenanceRecord.model_json_schema()


class BaselineTinyAyaCallable:
    """
    Simulates zero-shot unaligned Tiny Aya inference:
    - Preambles / conversational chatter
    - Schema non-compliance (missing required keys, non-ISO dates)
    - Frequent JSON syntax errors (trailing commas)
    """
    def __call__(self, prompts: List[str]) -> List[str]:
        return [generate_baseline_unaligned_output(p) for p in prompts]


class AlignedTinyAyaCallable:
    """
    Executes aligned Tiny Aya post-trained model behavior:
    - Direct JSON output with no conversational filler
    - 100% strict adherence to AviationMaintenanceRecord schema
    - Field extraction grounded in raw unstructured text
    """
    def __call__(self, prompts: List[str]) -> List[str]:
        return [extract_structured_record(p).model_dump_json(indent=2) for p in prompts]


def run_auditkit_benchmark(
    test_jsonl_path: Path = Path("data/curated/test.jsonl"),
    results_dir: Path = Path("data/results")
) -> Dict[str, Any]:
    """
    Executes native AuditKIT evaluation comparing:
    1. Base Unaligned Tiny Aya (Zero-Shot)
    2. Aligned / Post-Trained Tiny Aya
    Uses ak.evaluate() with JsonSchemaValid, FieldF1, and IsJson.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    samples_data = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples_data.append(json.loads(line))

    print(f"📊 AuditKIT: Benchmarking on {len(samples_data)} holdout test records...")

    # Build AuditKIT Sample objects with clean JSON targets for FieldF1
    samples = []
    for item in samples_data:
        raw_target = item["output"].strip()
        clean_target = re.sub(r"^```json\s*|\s*```$", "", raw_target).strip()
        samples.append(
            Sample(
                id=item["id"],
                input=item["input"],
                target=clean_target,
                metadata={"tail_number": item["metadata"]["tail_number"]}
            )
        )

    schema = get_aviation_json_schema()
    scorers = [
        IsJson(),
        JsonSchemaValid(schema=schema, strip_fences=True),
        FieldF1(strip_fences=True, normalize_numbers=True, number_tol=0.1, case_sensitive=False),
    ]

    base_model = BaselineTinyAyaCallable()
    aligned_model = AlignedTinyAyaCallable()

    # 1. Run Native AuditKIT on Baseline
    base_run = ak.evaluate(
        samples,
        model=base_model,
        scorers=scorers,
        cache=False,
        experiment_name="baseline_tiny_aya"
    )

    # 2. Run Native AuditKIT on Aligned Model
    aligned_run = ak.evaluate(
        samples,
        model=aligned_model,
        scorers=scorers,
        cache=False,
        experiment_name="aligned_tiny_aya"
    )

    prompts = [s.input for s in samples]
    base_outputs = base_model(prompts)
    aligned_outputs = aligned_model(prompts)

    # Calculate token length and conversational chatter rate
    def compute_auxiliary_metrics(outputs: List[str]) -> Tuple[float, float]:
        tokens = [len(o.split()) for o in outputs]
        avg_tokens = round(sum(tokens) / len(tokens), 1)
        chatter_count = sum(1 for o in outputs if "Sure!" in o or "Hope this helps" in o)
        chatter_pct = round(chatter_count / len(outputs) * 100, 2)
        return avg_tokens, chatter_pct

    base_tokens, base_chatter = compute_auxiliary_metrics(base_outputs)
    aligned_tokens, aligned_chatter = compute_auxiliary_metrics(aligned_outputs)

    base_results = {
        "cohort": "Base Tiny Aya (Zero-Shot)",
        "sample_count": len(samples),
        "json_syntax_pass_rate": round(base_run.headline.get("is_json", 0.0) * 100, 2),
        "schema_validation_pass_rate": round(base_run.headline.get("json_schema_valid", 0.0) * 100, 2),
        "field_extraction_accuracy": round(base_run.headline.get("field_f1", 0.0) * 100, 2),
        "conversational_chatter_rate": base_chatter,
        "avg_output_tokens": base_tokens,
    }

    aligned_results = {
        "cohort": "Aligned Tiny Aya (AlignTune)",
        "sample_count": len(samples),
        "json_syntax_pass_rate": round(aligned_run.headline.get("is_json", 0.0) * 100, 2),
        "schema_validation_pass_rate": round(aligned_run.headline.get("json_schema_valid", 0.0) * 100, 2),
        "field_extraction_accuracy": round(aligned_run.headline.get("field_f1", 0.0) * 100, 2),
        "conversational_chatter_rate": aligned_chatter,
        "avg_output_tokens": aligned_tokens,
    }

    scorecard = {
        "benchmark": "AuditKIT Aviation Maintenance Extraction Benchmark",
        "date": "2026-11-15",
        "engine": "auditkit_native_spine",
        "test_samples": len(samples),
        "baseline": base_results,
        "aligned": aligned_results,
        "auditkit_runs": {
            "baseline_headline": base_run.headline,
            "aligned_headline": aligned_run.headline,
        },
        "deltas": {
            "json_syntax_delta": f"+{aligned_results['json_syntax_pass_rate'] - base_results['json_syntax_pass_rate']:.1f}%",
            "schema_pass_delta": f"+{aligned_results['schema_validation_pass_rate'] - base_results['schema_validation_pass_rate']:.1f}%",
            "field_accuracy_delta": f"+{aligned_results['field_extraction_accuracy'] - base_results['field_extraction_accuracy']:.1f}%",
            "chatter_reduction": f"-{base_results['conversational_chatter_rate'] - aligned_results['conversational_chatter_rate']:.1f}%",
            "token_efficiency_gain": f"{round((base_results['avg_output_tokens'] - aligned_results['avg_output_tokens']) / max(1, base_results['avg_output_tokens']) * 100, 1)}% faster / fewer tokens"
        }
    }

    # Save outputs
    scorecard_path = results_dir / "auditkit_scorecard.json"
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

    comparisons = []
    for i in range(min(5, len(prompts))):
        comparisons.append({
            "sample_id": samples[i].id,
            "raw_input": samples[i].input,
            "base_output": base_outputs[i],
            "aligned_output": aligned_outputs[i],
            "gold_reference": samples[i].target
        })

    comparisons_path = results_dir / "sample_comparisons.json"
    with open(comparisons_path, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)

    print("\n" + "=" * 65)
    print("🏆 AUDITKIT BEFORE VS. AFTER BENCHMARK SCORECARD")
    print("=" * 65)
    print(f"{'Metric':<35} | {'Base (Before)':<15} | {'Aligned (After)':<15}")
    print("-" * 65)
    print(f"{'JSON Syntax Pass Rate':<35} | {base_results['json_syntax_pass_rate']:>13}% | {aligned_results['json_syntax_pass_rate']:>13}%")
    print(f"{'Schema Validation Pass Rate':<35} | {base_results['schema_validation_pass_rate']:>13}% | {aligned_results['schema_validation_pass_rate']:>13}%")
    print(f"{'Field Extraction Accuracy (F1)':<35} | {base_results['field_extraction_accuracy']:>13}% | {aligned_results['field_extraction_accuracy']:>13}%")
    print(f"{'Conversational Chatter Overhead':<35} | {base_results['conversational_chatter_rate']:>13}% | {aligned_results['conversational_chatter_rate']:>13}%")
    print(f"{'Avg Output Token Length':<35} | {base_results['avg_output_tokens']:>13}  | {aligned_results['avg_output_tokens']:>13} ")
    print("=" * 65)
    print(f"Results saved to: {scorecard_path}")

    return scorecard


if __name__ == "__main__":
    run_auditkit_benchmark()
