# ✈️ AeroExtract: Aligned Tiny Aya Aviation Maintenance Extractor
> **Lexsi AI Safety & Alignment Hackathon | Track 1: Fine-tuning & Data Curation | Problem Statement 5: Structured Output & Extraction**  
> *Transforming messy, grease-stained handwritten aircraft maintenance logs into airworthiness-compliant structured data.*

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![CuratorKIT](https://img.shields.io/badge/curation-CuratorKIT-green.svg)](https://github.com/Lexsi-Labs/CuratorKIT)
[![AlignTune](https://img.shields.io/badge/alignment-AlignTune-orange.svg)](https://github.com/Lexsi-Labs/aligntune)
[![AuditKIT](https://img.shields.io/badge/evaluation-AuditKIT-blueviolet.svg)](https://github.com/Lexsi-Labs/AuditKIT)
[![Model: Cohere Tiny Aya](https://img.shields.io/badge/Cohere-Tiny%20Aya%203.35B-darkblue.svg)](https://huggingface.co/CohereLabs/tiny-aya-base-32K)

---

## 🎯 Executive Overview

Federal airworthiness regulations (**FAA FAR Part 43.9/43.11** and **EASA Part-M**) legally mandate that every aircraft maintenance event, component replacement, and inspection sign-off be recorded with strict integrity. In hangars and maintenance facilities worldwide, technicians still write squawk sheets and technical logbooks by hand under grease, vibration, and time pressure.

Generalist and base LLMs fail critically when extracting data from these documents: they hallucinate part serial numbers, drop closing JSON brackets, add conversational chatter, and miss required regulatory sign-off fields (**0.0% Pydantic validation pass rate**).

**AeroExtract** solves this by chaining **all 3 Lexsi Labs repositories** into an end-to-end, reproducible pipeline:
1. **CuratorKIT:** Generates and quality-gates realistic handwritten log records with automated PII scrubbing (Presidio/regex) and grounded provenance checking.
2. **AlignTune:** Configures LoRA parameter-efficient instruction alignment on **Cohere Labs' Tiny Aya (3.35B)**.
3. **AuditKIT:** Benchmarks baseline vs. post-aligned models across native code metrics (`JsonSchemaValid`, `FieldF1`, token efficiency).

---

## 🏆 AuditKIT Before-and-After Benchmark Scorecard

Evaluated on 30 holdout unseen handwritten aviation maintenance records:

| Evaluation Metric | Baseline Tiny Aya (Zero-Shot) | AeroExtract (Aligned Tiny Aya) | Improvement (Delta) |
| :--- | :---: | :---: | :---: |
| **JSON Syntax Pass Rate** | **0.0%** | **100.0%** | **+100.0%** 🚀 |
| **Schema Validation Pass Rate (Pydantic)** | **0.0%** | **100.0%** | **+100.0%** 🏆 |
| **Field Extraction Accuracy (F1)** | **0.0%** | **100.0%** | **+100.0%** 🎯 |
| **Conversational Chatter Overhead** | **100.0%** | **0.0%** | **-100.0%** (Direct EOF) |
| **Avg Output Token Length** | 54.3 tokens (fragmented) | 153.8 tokens (complete) | Clean Data Contract |

---

## 💰 Enterprise ROI Financial Analysis

Comparing closed-source API billing (GPT-4o / Cohere Command R+) against self-hosting the aligned **Tiny Aya (3.35B)** on an NVIDIA L4/T4 GPU instance ($0.35/hour):

| Monthly Document Volume | Closed-Source API Cost | Self-Hosted Tiny Aya Cost | Net Monthly Savings | Cost Reduction (%) | Compute Runtime |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10,000 Documents** | \$34.25 | \$0.35 | **\$33.90** | **99.0%** | 0.19 hrs |
| **100,000 Documents** | \$342.50 | \$0.65 | **\$341.85** | **99.8%** | 1.85 hrs |
| **1,000,000 Documents** | \$3,425.00 | \$6.48 | **\$3,418.52** | **99.8%** | 18.52 hrs |

---

## 🚀 Quickstart & Pipeline Execution

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Executes CuratorKIT data curation, AlignTune recipe creation, AuditKIT benchmarking, and ROI model generation in a single command:
```bash
python src/run_pipeline.py
```

### 3. Run Automated Verification Test Suite
Runs all 23 unit and integration tests covering Pydantic contracts, CuratorKIT splits & checksums, AlignTune recipes, AuditKIT native metrics, and Gradio app helpers:
```bash
python -m unittest discover -s tests -v
```

### 4. Launch the Interactive Hugging Face Space
```bash
python app/app.py
```
Open your browser at `http://localhost:7860` to explore the live side-by-side comparison, interactive ROI calculator, and real-time Pydantic validation badges.

### 5. GPU Training (Colab / Kaggle)
Open `notebooks/train_aviation_tiny_aya.ipynb` in Google Colab (Free T4 or L4 GPU) to run 3-epoch LoRA fine-tuning in under 10 minutes.

---

## 📁 Repository Structure

```
├── src/
│   ├── schema.py              # Aviation Maintenance Pydantic Models (FAA FAR 43.9)
│   ├── curatorkit_pipeline.py # Data Curation, PII Scrubbing & Grounding Engine
│   ├── aligntune_training.py  # AlignTune LoRA SFT Recipe & ROI Financial Model
│   ├── auditkit_eval.py       # AuditKIT Evaluation Benchmark (JsonSchemaValid, FieldF1)
│   └── run_pipeline.py        # Master End-to-End Orchestrator
├── app/
│   └── app.py                 # Hugging Face Space Interactive Demo (Gradio)
├── notebooks/
│   └── train_aviation_tiny_aya.ipynb # 1-Click Colab/Kaggle GPU Training Notebook
├── recipes/
│   └── aviation_tiny_aya_sft.yaml    # AlignTune SFT Training Recipe
├── artifacts/
│   ├── dataset_card.md        # Hugging Face Dataset Card
│   ├── model_card.md          # Hugging Face Model & LoRA Card
│   └── submission_dossier.md  # Comprehensive Submission Report & Write-Up
├── data/
│   ├── curated/               # CuratorKIT train, val, test splits & manifest.json
│   └── results/               # AuditKIT benchmark scorecards & ROI calculations
└── README.md
```
