"""
Lexsi AI Safety and Alignment Hackathon - Central Configuration
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CURATED_DATA_DIR = DATA_DIR / "curated"
EXPORTS_DIR = DATA_DIR / "exports"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ADAPTERS_DIR = MODELS_DIR / "adapters"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

for d in [DATA_DIR, RAW_DATA_DIR, CURATED_DATA_DIR, EXPORTS_DIR, 
          MODELS_DIR, CHECKPOINTS_DIR, ADAPTERS_DIR, LOGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    # --- Hugging Face Configuration ---
    hf_username: str = os.getenv("HF_USERNAME", "lexsi-participant")
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    hf_dataset_repo: str = field(default="")
    hf_model_repo: str = field(default="")

    # --- Cohere Labs Base Models ---
    # Options: "CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-base", "CohereLabs/aya-expanse-8b"
    base_model_name: str = os.getenv("BASE_MODEL_NAME", "CohereLabs/tiny-aya-global")
    
    # --- Synthetic Data Generation (OpenAI / Anthropic) ---
    generation_provider: str = os.getenv("GENERATION_PROVIDER", "openai")  # "openai" or "anthropic"
    generator_model: str = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")  # or "claude-3-5-sonnet-20241022"
    num_samples_to_generate: int = int(os.getenv("NUM_SAMPLES", "100"))
    target_safety_domains: List[str] = field(default_factory=lambda: [
        "jailbreak_resistance",
        "hallucination_and_groundedness",
        "deception_and_sycophancy",
        "cybersecurity_and_dual_use",
        "pii_and_privacy_protection",
        "nuanced_refusal_and_helpful_redirection"
    ])

    # --- CuratorKIT Settings ---
    enable_hallucination_gate: bool = True
    enable_reward_gate: bool = True
    enable_adaptive_recovery: bool = True
    hygiene_pii: bool = True
    hygiene_secrets: bool = True
    dedup_strategy: str = "minhash"

    # --- AlignTune Settings (SFT / DPO) ---
    training_algorithm: str = "sft"  # "sft", "dpo", or "grpo"
    backend: str = "trl"             # "trl" or "unsloth"
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    # --- AuditKIT Settings ---
    audit_metrics: List[str] = field(default_factory=lambda: [
        "exact_match",
        "contains",
        "hallucination",
        "security.defcon_grade",
        "safety.refusal_precision",
        "perf.latency_stats"
    ])

    def __post_init__(self):
        slug = self.base_model_name.split("/")[-1].lower()
        if not self.hf_dataset_repo:
            self.hf_dataset_repo = f"{self.hf_username}/lexsi-cohere-alignment-dataset"
        if not self.hf_model_repo:
            self.hf_model_repo = f"{self.hf_username}/{slug}-lexsi-safety-aligned"


# Global singleton instance
config = PipelineConfig()
