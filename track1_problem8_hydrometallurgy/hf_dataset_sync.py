"""
Step 4: Hugging Face DatasetDict Packaging & Hub Sync
Loads raw_dataset.jsonl and curated_dataset.jsonl into a unified DatasetDict:
  - split 'raw': 821 uncurated generated samples
  - split 'curated': 768 CuratorKIT-filtered high-purity samples
Saves locally and pushes directly to Hugging Face Hub.
"""

import os
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

load_dotenv()

RAW_FILE = Path("raw_dataset.jsonl")
CURATED_FILE = Path("curated_dataset.jsonl")
LOCAL_EXPORT_DIR = Path("data/hf_dataset_dict")


def create_dataset_card(raw_count: int, curated_count: int, repo_id: str) -> str:
    """Generates a professional Hugging Face Dataset Card."""
    retention = f"{(curated_count / max(1, raw_count)) * 100:.2f}%"
    card = f"""---
license: apache-2.0
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- alignment
- ai-safety
- curatorkit
- cohere
- tiny-aya
- hydrometallurgy
- rare-earth-elements
- data-scarce-domains
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: raw
    path: raw/*
  - split: curated
    path: curated/*
---

# Rare Earth Element (REE) Hydrometallurgy Alignment Dataset

**Track 1: Fine-Tuning & Data Curation — Problem 8: Data-Scarce Domains**  
*Submitted to the Lexsi AI Safety & Alignment Hackathon (Lexsi Labs × Cohere Labs × Hugging Face)*

## 🔬 Dataset Overview
This dataset addresses synthetic data generation, validation, and filtering in highly specialized, data-scarce technical domains. The selected domain is **Rare Earth Element (REE) Extraction Hydrometallurgy**, specifically focusing on:
- Multi-stage countercurrent solvent extraction parameters (Nd3+ / Pr3+ separation).
- Extractant dimerization equilibria (PC-88A / P507, Cyanex 272, D2EHPA).
- Saponification fractions (30-40%), equilibrium pH control (2.8 - 3.6), and scrub/strip HCl molarity optimization.
- Industrial troubleshooting: phase inversion, interphase crud formation (colloidal silica/iron particulates, CaSO4 scaling), and third-phase microemulsion prevention with isodecanol/TBP modifiers.

## 📊 Dataset Splits

| Split | Description | Samples |
| :--- | :--- | :---: |
| **`raw`** | Raw synthetic generation from Cohere Labs `command-r-plus-08-2024` and Groq | **{raw_count}** |
| **`curated`** | Filtered through **Lexsi CuratorKIT** (MinHash dedup, length bounds, stoichiometry & domain verification) | **{curated_count}** |

## 🧪 Curation Pipeline (Lexsi CuratorKIT)
- **Deduplicator Gate**: 5-gram token shingling with MinHash Jaccard similarity threshold >= 0.82.
- **LengthFilter Gate**: Filtered out degenerate, truncated, or unclosed markdown responses (85 <= words <= 1600).
- **Stoichiometry & Chemistry Verifier**: Trapped false saponification byproducts (hallucinated alcohols) and unrealistic separation factors (beta > 2.0).
- **Retention Rate**: **{retention}**

## 💬 Schema
Conforms to the standard Hugging Face conversational ChatML template:
```json
{{
  "messages": [
    {{"role": "user", "content": "<Technical metallurgical inquiry>"}},
    {{"role": "assistant", "content": "<Chemically and mathematically rigorous engineering response>"}}
  ],
  "metadata": {{
    "subdomain": "Extractant Chemistry & Saponification",
    "extractant": "PC-88A",
    "target_elements": ["Nd", "Pr"]
  }}
}}
```

## 🚀 Model Post-Training
Used with **Lexsi AlignTune** to train two comparative PEFT/LoRA adapters on **Cohere Labs' dense open-weights model** (`CohereLabs/tiny-aya-global`):
1. `tiny-aya-global-ree-raw`: Trained on the `raw` split.
2. `tiny-aya-global-ree-curated`: Trained on the `curated` split.
"""
    return card


def load_jsonl_to_dataset(file_path: Path) -> Dataset:
    """Loads JSONL records into a Hugging Face Dataset."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                records.append({
                    "messages": item.get("messages", []),
                    "subdomain": item.get("metadata", {}).get("subdomain", "General Hydrometallurgy"),
                    "extractant": item.get("metadata", {}).get("extractant", "PC-88A"),
                    "generator": item.get("generator", "cohere-command-r-plus")
                })
    return Dataset.from_list(records)


def package_and_push_dataset(
    repo_id: Optional[str] = None,
    private: bool = False
) -> DatasetDict:
    """Builds unified DatasetDict and uploads to Hugging Face."""
    print("\n📦 [Step 4] Packaging Datasets into Hugging Face DatasetDict...")
    
    if not RAW_FILE.exists() or not CURATED_FILE.exists():
        raise FileNotFoundError(f"Missing required data files: {RAW_FILE} or {CURATED_FILE}")

    # 1. Load splits
    raw_dataset = load_jsonl_to_dataset(RAW_FILE)
    curated_dataset = load_jsonl_to_dataset(CURATED_FILE)

    # 2. Construct unified DatasetDict
    dataset_dict = DatasetDict({
        "raw": raw_dataset,
        "curated": curated_dataset
    })

    print(f"✅ Created DatasetDict successfully:")
    print(f"   • Split 'raw': {len(raw_dataset)} records")
    print(f"   • Split 'curated': {len(curated_dataset)} records")

    # 3. Save locally
    LOCAL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(LOCAL_EXPORT_DIR))
    print(f"💾 Saved local copy to: {LOCAL_EXPORT_DIR}")

    # 4. Push to Hugging Face Hub if token available
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME", "lexsi-participant")
    target_repo = repo_id or os.getenv("HF_DATASET_REPO") or f"{hf_username}/ree-hydrometallurgy-alignment"

    # Write Dataset Card
    readme_path = LOCAL_EXPORT_DIR / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(create_dataset_card(len(raw_dataset), len(curated_dataset), target_repo))

    if hf_token and hf_token.strip():
        print(f"🚀 Pushing DatasetDict to Hugging Face Hub: {target_repo}...")
        try:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True, private=private)
            
            dataset_dict.push_to_hub(
                repo_id=target_repo,
                token=hf_token,
                private=private,
                commit_message="Add raw and CuratorKIT-curated REE Hydrometallurgy splits (Lexsi Hackathon)"
            )
            
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=target_repo,
                repo_type="dataset",
                token=hf_token
            )
            print(f"🎉 Successfully uploaded dataset to: https://huggingface.co/datasets/{target_repo}")
        except Exception as e:
            print(f"⚠️ Hugging Face Hub push skipped or encountered: {e}")
            print("Local DatasetDict is ready and can be pushed at any time.")
    else:
        print("ℹ️ Note: HF_TOKEN not configured yet in .env.")
        print(f"   Local DatasetDict is fully packaged at {LOCAL_EXPORT_DIR}/ ready to push.")

    return dataset_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Package and Push REE DatasetDict to Hugging Face")
    parser.add_argument("--repo-id", type=str, default=None, help="Target HF dataset repository ID (e.g. username/repo-name)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    package_and_push_dataset(repo_id=args.repo_id, private=args.private)


if __name__ == "__main__":
    main()
