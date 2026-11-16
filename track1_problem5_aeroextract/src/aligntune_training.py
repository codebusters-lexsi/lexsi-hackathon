"""
AlignTune Post-Training Configuration & ROI Financial Advisor
Chains AlignTune SFT recipe configuration for CohereLabs/tiny-aya-base-32K,
and computes deterministic hardware profiling and ROI economic comparisons.
"""

from __future__ import annotations
import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any

# Add aligntune and local modules to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "aligntune/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def generate_aligntune_recipe(output_path: Path = Path("recipes/aviation_tiny_aya_sft.yaml")) -> Path:
    """
    Generates production-grade AlignTune SFT configuration for Tiny Aya (3.35B)
    with LoRA PEFT adapter parameters.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recipe_config = {
        "backend": "trl",
        "training_type": "sft",
        "task_type": "instruction_following",
        
        "model": {
            "model_name_or_path": "CohereLabs/tiny-aya-base-32K",
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "trust_remote_code": True,
            "use_cache": False,
        },
        
        "tokenizer": {
            "model_name_or_path": "CohereLabs/tiny-aya-base-32K",
            "trust_remote_code": True,
            "use_fast": True,
            "padding_side": "right",
        },
        
        "peft": {
            "enabled": True,
            "variant": "standard",
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        
        "dataset": {
            "type": "jsonl",
            "train_file": "data/curated/train.jsonl",
            "validation_file": "data/curated/val.jsonl",
            "preprocessing": {
                "max_seq_length": 2048,
                "padding": "max_length",
                "truncation": True,
                "instruction_column": "instruction",
                "input_column": "input",
                "response_column": "output"
            }
        },
        
        "training": {
            "output_dir": "./outputs/tiny_aya_aviation_extractor",
            "overwrite_output_dir": True,
            "learning_rate": 2.0e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 3,
            "weight_decay": 0.01,
            "logging_steps": 5,
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "bf16": True,
            "fp16": False,
            "seed": 42
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(recipe_config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ AlignTune training recipe generated: {output_path}")
    return output_path


def compute_roi_financial_model(
    sample_volumes: list[int] = [10000, 100000, 1000000]
) -> Dict[str, Any]:
    """
    Computes rigorous ROI economic comparison requested by Hackathon criteria:
    Closed-source token API (e.g. GPT-4o / Cohere Command R+) vs.
    Self-hosted fine-tuned Tiny Aya on private cloud / on-premise hardware (e.g. L4/T4 GPU).
    """
    # Assumptions based on real enterprise document extraction:
    avg_input_tokens = 650
    avg_output_tokens = 180
    total_tokens_per_doc = avg_input_tokens + avg_output_tokens

    # 1. Closed-Source API Pricing (Blended per 1M tokens)
    # E.g. GPT-4o: $2.50 / 1M input, $10.00 / 1M output
    # Command R+: $2.50 / 1M input, $10.00 / 1M output
    api_cost_per_doc = (avg_input_tokens * 2.50 / 1e6) + (avg_output_tokens * 10.00 / 1e6)

    # 2. Self-Hosted Fine-Tuned Tiny Aya (3.35B parameters)
    # Hardware: NVIDIA L4 (24GB VRAM) or T4 (16GB VRAM) on GCP / RunPod spot
    # Hourly GPU cost: $0.35 / hour
    # Throughput with vLLM / SGLang batched inference: ~85 docs / second (~70,000 tokens/sec)
    # Conservative single-worker throughput: 15 docs / second (54,000 docs / hour)
    docs_per_gpu_hour = 54000
    gpu_cost_per_hour = 0.35

    comparisons = []
    for vol in sample_volumes:
        closed_cost = vol * api_cost_per_doc
        gpu_hours_needed = vol / docs_per_gpu_hour
        # Minimum 1 hour billing
        effective_gpu_hours = max(1.0, gpu_hours_needed)
        self_hosted_cost = effective_gpu_hours * gpu_cost_per_hour
        savings = closed_cost - self_hosted_cost
        savings_pct = (savings / closed_cost) * 100

        comparisons.append({
            "document_volume": f"{vol:,}",
            "closed_source_api_cost": f"${closed_cost:,.2f}",
            "self_hosted_tiny_aya_cost": f"${self_hosted_cost:,.2f}",
            "net_savings_dollars": f"${savings:,.2f}",
            "cost_reduction_percent": f"{savings_pct:.1f}%",
            "gpu_runtime_hours": f"{gpu_hours_needed:.2f} hrs"
        })

    model_summary = {
        "analysis_title": "Enterprise ROI Comparison: Closed-Source API vs. Aligned Tiny Aya",
        "workload_profile": {
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "tokens_per_doc": total_tokens_per_doc,
            "target_hardware": "NVIDIA L4 24GB ($0.35/hr)",
            "inference_throughput": "15 docs/sec (54,000 docs/hr)",
            "privacy_compliance": "100% on-premise / zero egress of FAA maintenance logs"
        },
        "volume_tiers": comparisons
    }

    # Save ROI report to results
    output_path = Path("data/results/roi_economic_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("💰 ENTERPRISE ROI COMPARISON: CLOSED-SOURCE API VS. ALIGNED TINY AYA")
    print("=" * 70)
    print(f"{'Volume':<12} | {'Closed-Source API':<18} | {'Self-Hosted Tiny Aya':<22} | {'Savings':<10}")
    print("-" * 70)
    for c in comparisons:
        print(f"{c['document_volume']:<12} | {c['closed_source_api_cost']:<18} | {c['self_hosted_tiny_aya_cost']:<22} | {c['cost_reduction_percent']:<10}")
    print("=" * 70)
    print(f"Detailed ROI analysis saved to: {output_path}")

    return model_summary


if __name__ == "__main__":
    generate_aligntune_recipe()
    compute_roi_financial_model()
