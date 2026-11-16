"""
Master Pipeline Orchestrator: Lexsi Hackathon Track 1, Problem Statement 5
Chains CuratorKIT -> AlignTune -> AuditKIT into an integrated, reproducible pipeline.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

# Add project root and local packages to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from curatorkit_pipeline import generate_curated_dataset
from aligntune_training import generate_aligntune_recipe, compute_roi_financial_model
from auditkit_eval import run_auditkit_benchmark


def run_full_pipeline():
    start_time = time.time()
    print("=" * 75)
    print("🚀 LEXSI LABS HACKATHON: TRACK 1, PROBLEM 5 END-TO-END PIPELINE")
    print("   Chaining: CuratorKIT  -->  AlignTune  -->  AuditKIT")
    print("=" * 75)

    # -------------------------------------------------------------
    # STAGE 1: CuratorKIT Data Curation & Provenance Pipeline
    # -------------------------------------------------------------
    print("\n[STAGE 1/4] Running CuratorKIT Data Curation & Quality Gate...")
    samples, manifest = generate_curated_dataset(
        num_samples=200,
        output_dir=Path("data/curated"),
        raw_dir=Path("data/raw")
    )
    print(f"  -> Generated {manifest['total_passed']} quality-gated samples.")
    print(f"  -> Emitted provenance manifest with SHA-256 integrity verification.")

    # -------------------------------------------------------------
    # STAGE 2: AlignTune Configuration & PEFT Recipe Generator
    # -------------------------------------------------------------
    print("\n[STAGE 2/4] Generating AlignTune SFT Training Recipe for Tiny Aya...")
    recipe_path = generate_aligntune_recipe(Path("recipes/aviation_tiny_aya_sft.yaml"))
    print(f"  -> Recipe saved to {recipe_path}")

    # -------------------------------------------------------------
    # STAGE 3: AuditKIT Multi-Metric Evaluation & Diagnostics
    # -------------------------------------------------------------
    print("\n[STAGE 3/4] Running AuditKIT Benchmark (Baseline vs. Aligned)...")
    scorecard = run_auditkit_benchmark(
        test_jsonl_path=Path("data/curated/test.jsonl"),
        results_dir=Path("data/results")
    )

    # -------------------------------------------------------------
    # STAGE 4: AlignTune ROI Financial Economics Analysis
    # -------------------------------------------------------------
    print("\n[STAGE 4/4] Computing AlignTune Enterprise ROI Analysis...")
    roi_model = compute_roi_financial_model()

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 75)
    print(f"✅ PIPELINE COMPLETED SUCCESSFULLY IN {elapsed}s")
    print("   All artifacts and results are saved in data/curated/ and data/results/")
    print("=" * 75)


if __name__ == "__main__":
    run_full_pipeline()
