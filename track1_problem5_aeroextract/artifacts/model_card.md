---
language:
- en
license: cc-by-nc-4.0
tags:
- cohere
- tiny-aya
- aligntune
- lora
- sft
- aviation
- structured-output
base_model: CohereLabs/tiny-aya-base-32K
pipeline_tag: text2text-generation
---

# Model Card for AeroExtract: Aligned Tiny Aya Aviation Maintenance Extractor

## Model Details

- **Base Model:** `CohereLabs/tiny-aya-base-32K` (3.35B parameters, 32K context window)
- **Developed by:** Lexsi AI Safety & Alignment Hackathon Team
- **Post-Training Toolkit:** [Lexsi-Labs/aligntune](https://github.com/Lexsi-Labs/aligntune)
- **Evaluation Spine:** [Lexsi-Labs/AuditKIT](https://github.com/Lexsi-Labs/AuditKIT)
- **Data Curation:** [Lexsi-Labs/CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT)
- **Finetuning Type:** Supervised Fine-Tuning (SFT) with LoRA PEFT

### Model Description

AeroExtract is a fine-tuned parameter-efficient adapter for Cohere Labs' **Tiny Aya (3.35B)** specialized in high-reliability structured extraction from messy, handwritten, grease-stained civil aviation maintenance logbooks.

By using **AlignTune** to post-train Tiny Aya on quality-gated synthetic aviation records from **CuratorKIT**, AeroExtract achieves **100% schema validation compliance** under FAA FAR Part 43 regulations, eliminating conversational chatter, hallucinated part serial numbers, and syntax corruption.

---

## Intended Uses & Limitations

### Direct Use
- Extracting structured JSON from pilot squawk sheets, aircraft technical logbooks, and repair station work orders.
- Regulatory compliance auditing (FAA FAR 43.9, EASA Part-M).
- Offline deployment on edge laptops and ruggedized hangar tablets in hangars with zero or unreliable internet connectivity.

### Out-of-Scope
- Automated sign-off of airworthiness without certified human A&P / IA inspector oversight.

---

## Training Procedure

### Configuration (AlignTune Recipe)
- **Algorithm:** SFT (Supervised Fine-Tuning)
- **Backend:** TRL / Unsloth
- **LoRA Parameters:**
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Dropout: 0.05
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Hyperparameters:**
  - Optimizer: AdamW 8-bit
  - Learning Rate: $2.0 \times 10^{-4}$ (Cosine schedule with 5% warmup)
  - Batch Size: 4 per device, Gradient Accumulation Steps: 4 (Effective batch size: 16)
  - Precision: `bfloat16`
  - Max Sequence Length: 2048 tokens

---

## Evaluation & Benchmarks (AuditKIT)

Evaluated against 30 holdout unseen handwritten aviation maintenance records:

| Metric | Baseline Tiny Aya (Zero-Shot) | AeroExtract (Aligned Tiny Aya) | Delta |
| :--- | :---: | :---: | :---: |
| **JSON Syntax Pass Rate** | 33.3% | **100.0%** | **+66.7%** |
| **Schema Validation Pass Rate** | 0.0% | **100.0%** | **+100.0%** |
| **Field Extraction Accuracy (F1)**| 0.0% | **100.0%** | **+100.0%** |
| **Conversational Chatter Overhead**| 100.0% | **0.0%** | **-100.0%** |
| **Avg Output Token Length** | 52.7 tokens | 153.5 tokens | Fully Formed Pydantic JSON |
