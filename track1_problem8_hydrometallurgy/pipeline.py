"""
Lexsi AI Safety and Alignment Hackathon - Master Pipeline Orchestrator
Track 1 (Fine-Tuning & Data Curation), Problem 8: Data-Scarce Domains
Domain: Rare Earth Element Extraction Hydrometallurgy (Nd3+/Pr3+ Separation)

Chains:
  1. Synthetic Generation (generate_rare_earth_dataset.py)
  2. CuratorKIT Curation (curate_rare_earth_dataset.py)
  3. Hugging Face DatasetDict Packaging & Hub Sync (hf_dataset_sync.py)
  4. AlignTune Comparative Training: Raw vs Curated (train_aligntune_adapters.py)
  5. AuditKIT Comparative Benchmarking (audit_comparative_models.py)
  6. End-to-End Test Verification (pytest tests/test_pipeline.py)
"""

import sys
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / "venv" / "bin" / "python"
PYTHON_BIN = str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)


def run_cmd(cmd_list):
    print(f"\n[Executing] {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, check=True)
    return res.returncode


def stage_generate(samples: int = 5, provider: str = "cohere"):
    print("=" * 70)
    print("STAGE 1: Asynchronous Synthetic REE Hydrometallurgy Generation")
    print("=" * 70)
    cmd = [PYTHON_BIN, "generate_rare_earth_dataset.py", "--samples", str(samples), "--provider", provider]
    run_cmd(cmd)


def stage_curate(skip_llm_judge: bool = True):
    print("=" * 70)
    print("STAGE 2: Lexsi CuratorKIT Curation & Filtering Pipeline")
    print("=" * 70)
    cmd = [PYTHON_BIN, "curate_rare_earth_dataset.py", "--raw", "raw_dataset.jsonl", "--curated", "curated_dataset.jsonl"]
    if skip_llm_judge:
        cmd.append("--skip-llm-judge")
    run_cmd(cmd)


def stage_sync_hf(repo_id: str = None):
    print("=" * 70)
    print("STAGE 3: Hugging Face DatasetDict Packaging & Sync")
    print("=" * 70)
    cmd = [PYTHON_BIN, "hf_dataset_sync.py"]
    if repo_id:
        cmd.extend(["--repo-id", repo_id])
    run_cmd(cmd)


def stage_train(split: str = "both"):
    print("=" * 70)
    print("STAGE 4: Lexsi AlignTune Post-Training on Cohere Tiny Aya")
    print("=" * 70)
    cmd = [PYTHON_BIN, "train_aligntune_adapters.py", "--split", split]
    run_cmd(cmd)


def stage_audit():
    print("=" * 70)
    print("STAGE 5: Lexsi AuditKIT Empirical Benchmarking")
    print("=" * 70)
    cmd = [PYTHON_BIN, "audit_comparative_models.py"]
    run_cmd(cmd)


def stage_test():
    print("=" * 70)
    print("STAGE 6: Full Verification Test Suite (pytest)")
    print("=" * 70)
    cmd = [PYTHON_BIN, "-m", "pytest", "tests/test_pipeline.py", "-v"]
    run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(description="Lexsi Track 1 Problem 8 Master Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="test",
        choices=["all", "generate", "curate", "sync-hf", "train", "audit", "test"],
        help="Pipeline stage to execute"
    )
    parser.add_argument("--samples", type=int, default=5, help="Samples to generate")
    parser.add_argument("--provider", type=str, default="cohere", help="Generation provider (cohere/groq)")
    parser.add_argument("--split", type=str, default="both", help="Split to train (raw/curated/both)")
    parser.add_argument("--repo-id", type=str, default=None, help="Target Hugging Face repo ID")
    parser.add_argument("--with-llm-judge", action="store_true", help="Enable LLM judge during curation (default: fast heuristic-only)")

    args = parser.parse_args()

    if args.stage == "all":
        stage_generate(samples=args.samples, provider=args.provider)
        stage_curate(skip_llm_judge=not args.with_llm_judge)
        stage_sync_hf(repo_id=args.repo_id)
        stage_train(split=args.split)
        stage_audit()
        stage_test()
    elif args.stage == "generate":
        stage_generate(samples=args.samples, provider=args.provider)
    elif args.stage == "curate":
        stage_curate(skip_llm_judge=not args.with_llm_judge)
    elif args.stage == "sync-hf":
        stage_sync_hf(repo_id=args.repo_id)
    elif args.stage == "train":
        stage_train(split=args.split)
    elif args.stage == "audit":
        stage_audit()
    elif args.stage == "test":
        stage_test()


if __name__ == "__main__":
    main()
