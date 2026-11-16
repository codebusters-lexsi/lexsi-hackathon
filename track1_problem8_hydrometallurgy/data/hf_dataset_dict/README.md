---
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
| **`raw`** | Raw synthetic generation from Cohere Labs `command-r-plus-08-2024` and Groq | **821** |
| **`curated`** | Filtered through **Lexsi CuratorKIT** (MinHash dedup, length bounds, stoichiometry & domain verification) | **776** |

## 🧪 Curation Pipeline (Lexsi CuratorKIT)
- **Deduplicator Gate**: 5-gram token shingling with MinHash Jaccard similarity threshold >= 0.82.
- **LengthFilter Gate**: Filtered out degenerate, truncated, or unclosed markdown responses (85 <= words <= 1600).
- **Stoichiometry & Chemistry Verifier**: Trapped false saponification byproducts (hallucinated alcohols) and unrealistic separation factors (beta > 2.0).
- **Retention Rate**: **94.52%**

## 💬 Schema
Conforms to the standard Hugging Face conversational ChatML template:
```json
{
  "messages": [
    {"role": "user", "content": "<Technical metallurgical inquiry>"},
    {"role": "assistant", "content": "<Chemically and mathematically rigorous engineering response>"}
  ],
  "metadata": {
    "subdomain": "Extractant Chemistry & Saponification",
    "extractant": "PC-88A",
    "target_elements": ["Nd", "Pr"]
  }
}
```

## 🚀 Model Post-Training
Used with **Lexsi AlignTune** to train two comparative PEFT/LoRA adapters on **Cohere Labs' dense open-weights model** (`CohereLabs/tiny-aya-global`):
1. `tiny-aya-global-ree-raw`: Trained on the `raw` split.
2. `tiny-aya-global-ree-curated`: Trained on the `curated` split.
