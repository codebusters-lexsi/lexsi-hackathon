# 🛡️ Lexsi AI Safety & Alignment Hackathon
> **Track 1: Fine-Tuning & Data Curation**  
> *Organized by Lexsi Labs × Cohere Labs × Hugging Face*  
> *Submitted by Rishi Cherukuri*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![CuratorKIT](https://img.shields.io/badge/curation-CuratorKIT-green.svg)](https://github.com/Lexsi-Labs/CuratorKIT)
[![AlignTune](https://img.shields.io/badge/alignment-AlignTune-orange.svg)](https://github.com/Lexsi-Labs/aligntune)
[![AuditKIT](https://img.shields.io/badge/evaluation-AuditKIT-blueviolet.svg)](https://github.com/Lexsi-Labs/AuditKIT)
[![Cohere Labs](https://img.shields.io/badge/model-Cohere%20Tiny%20Aya%20(3.35B)-darkblue.svg)](https://huggingface.co/CohereLabs/tiny-aya-global)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

---

## 📖 Executive Abstract

For this hackathon, we tackled two complementary problem statements within **Track 1 (Fine-Tuning & Data Curation)** to demonstrate the versatility, depth, and mathematical rigor of the **Lexsi Labs open-source lifecycle stack (`CuratorKIT` $\rightarrow$ `AlignTune` $\rightarrow$ `AuditKIT`)** using Cohere Labs' dense open-weights model, **Tiny Aya (3.35B)**:

1. **Problem Statement 5: Structured Output and Extraction (AeroExtract)** — Transforming unconstrained, grease-stained handwritten aircraft maintenance logs into airworthiness-compliant structured data under **FAA FAR Part 43** and **EASA Part-M** standards.
2. **Problem Statement 8: Data-Scarce Domains (Rare Earth Hydrometallurgy)** — Solving data scarcity in critical mineral processing (countercurrent $\text{Nd}^{3+}/\text{Pr}^{3+}$ solvent extraction using PC-88A), empirically proving that data curation with `CuratorKIT` measurably outperforms raw synthetic generation.

```
                         ┌─────────────────────────────────────────────────────────┐
                         │               Lexsi Labs Lifecycle Stack                │
                         └────────────────────────────┬────────────────────────────┘
                                                      │
                     ┌────────────────────────────────┴────────────────────────────────┐
                     ▼                                                                 ▼
      ┌─────────────────────────────┐                                   ┌─────────────────────────────┐
      │ Track 1, Problem 5: AeroExtract │                               │ Track 1, Problem 8: REE Metallurgy │
      └──────────────┬──────────────┘                                   └──────────────┬──────────────┘
                     │                                                                 │
    1. CuratorKIT: PII scrubbing, ATA chapter                          1. CuratorKIT: MinHash dedup, length bounds,
       grounding, SHA-256 provenance manifests.                           stoichiometry & parameter verification.
                     │                                                                 │
    2. AlignTune: LoRA SFT recipe on Tiny Aya                          2. AlignTune: Comparative LoRA training
       optimized for rugged hangar tablets.                               (Raw Baseline vs. Curated Adapter).
                     │                                                                 │
    3. AuditKIT: JsonSchemaValid, FieldF1,                             3. AuditKIT: Chemical accuracy, McCabe-Thiele
       zero-chatter evaluation.                                           realism, hallucination reduction.
                     │                                                                 │
                     ▼                                                                 ▼
      ┌─────────────────────────────┐                                   ┌─────────────────────────────┐
      │ 100% Pydantic Schema Pass   │                                   │ 96.1% Overall Scientific Score│
      │ 99.8% Enterprise Cost Saving│                                   │ -87.1% Hallucination Drop   │
      └─────────────────────────────┘                                   └─────────────────────────────┘
```

---

## ✈️ Project 1 — Track 1, Problem 5: AeroExtract
📂 **Dedicated Folder**: [`track1_problem5_aeroextract/`](track1_problem5_aeroextract/)  
📄 **Comprehensive Sub-Project README**: [`track1_problem5_aeroextract/README.md`](track1_problem5_aeroextract/README.md)

### The Challenge
Federal airworthiness regulations legally mandate that every aircraft maintenance event and component replacement be recorded with zero tolerance for error. Base foundation models fail completely on messy handwritten logs—yielding **0.0% Pydantic schema validation**, corrupting JSON brackets, and emitting conversational chatter that halts automated MRO software pipelines.

### The Solution (Chaining Lexsi Stack)
1. **CuratorKIT**: Ingested 200 technician records, performed automated regex/Presidio PII redacting, grounded tasks against ATA 100 chapter taxonomies, and emitted verified SHA-256 provenance manifests.
2. **AlignTune**: Configured an efficient LoRA parameter-efficient fine-tuning recipe ($r=16, \alpha=32$) on **Cohere Tiny Aya (3.35B)** designed for edge deployment on ruggedized hangar tablets and offline airstrips.
3. **AuditKIT**: Evaluated holdout logs against structured metrics (`JsonSchemaValid`, `FieldF1`, `IsJson`, and chatter overhead).

### Key Results & Enterprise ROI
- **Schema Compliance**: Lifted JSON syntax and strict Pydantic validation pass rates from **0.0% to 100.0%** with a **1.0 Field F1**.
- **Chatter Overhead**: Completely eliminated pre-amble/post-amble conversational chatter (**0.0%**, direct EOF token).
- **Financial ROI**: Self-hosting on an NVIDIA L4 GPU instance ($0.35/hr) yields **99.8% cost savings** over closed-source APIs (\$6.48 vs. \$3,425 per 1M documents) with complete data sovereignty.

---

## 🔬 Project 2 — Track 1, Problem 8: Rare Earth Hydrometallurgy
📂 **Dedicated Folder**: [`track1_problem8_hydrometallurgy/`](track1_problem8_hydrometallurgy/)  
📄 **Comprehensive Sub-Project README**: [`track1_problem8_hydrometallurgy/README.md`](track1_problem8_hydrometallurgy/README.md)  
📝 **Publication Article**: [`track1_problem8_hydrometallurgy/HF_ARTICLE.md`](track1_problem8_hydrometallurgy/HF_ARTICLE.md)

### The Challenge
In hyper-specialized domains like Rare Earth Element (REE) separation, public training data is virtually non-existent. While frontier LLMs can generate synthetic data, uncurated samples contain plausible-sounding but chemically dangerous hallucinations—such as quoting physically impossible separation factors ($\beta > 2.0$), inventing non-existent alcohol saponification byproducts, and advising toxic caustic dosing that causes colloidal hydroxide precipitation.

### The Solution (Chaining Lexsi Stack)
1. **CuratorKIT**: Generated 821 technical dialogues via Cohere Labs Command R+, then passed them through MinHash 5-gram token shingling (Jaccard $\ge 0.82$), length bounding, and chemical stoichiometry verification. Caught and filtered 45 dangerous hallucinations, yielding **776 high-purity dialogues (94.52% retention)** packaged into a Hugging Face `DatasetDict`.
2. **AlignTune**: Trained two comparative LoRA adapters ($r=16, \alpha=32$) on **Cohere Tiny Aya (3.35B)**: Adapter A (trained on raw data) vs. Adapter B (trained on Curated data).
3. **AuditKIT**: Benchmarked the models across reaction stoichiometry, McCabe-Thiele extraction stage realism, and hallucination frequency.

### Key Results & Empirical Proof
- **Scientific Accuracy**: The curated adapter elevated overall scientific accuracy from **53.6% (raw) to 96.1%** (**+79.3% relative gain**).
- **Hallucination Suppression**: Reduced critical chemical hallucinations from **32.5% to 4.2%** (**-87.1% reduction**).
- **Core Proof**: Conclusively proved that rigorous data curation with `CuratorKIT` is the indispensable bottleneck for specialized domain post-training.

---

## 📊 Summary Scorecard Across Both Tracks

| Dimension | Track 1, Problem 5: AeroExtract | Track 1, Problem 8: REE Hydrometallurgy |
| :--- | :---: | :---: |
| **Domain** | Aviation Maintenance (FAA / EASA) | Rare Earth Mineral Extraction ($\text{Nd}^{3+}/\text{Pr}^{3+}$) |
| **Base Model** | Cohere Labs Tiny Aya (3.35B) | Cohere Labs Tiny Aya (3.35B) |
| **CuratorKIT Role** | PII Redaction & ATA Chapter Grounding | MinHash Dedup, LengthFilter & Stoichiometry Verification |
| **AlignTune Recipe** | LoRA SFT ($r=16, \alpha=32$) | Comparative LoRA SFT ($r=16, \alpha=32$, Raw vs Curated) |
| **AuditKIT Metric** | `JsonSchemaValid`, `FieldF1`, Zero Chatter | Stoichiometry, Separation Factor Realism, Hallucination Rate |
| **Primary Metric Lift** | **0.0% $\rightarrow$ 100.0%** Pydantic Schema Pass | **53.6% $\rightarrow$ 96.1%** Scientific Accuracy |
| **Safety / Error Metric** | Chatter overhead dropped to **0.0%** | Hallucinations slashed from **32.5% $\rightarrow$ 4.2%** (-87.1%) |
| **Enterprise Impact** | **99.8% cost reduction** (\$6.48 vs \$3,425 / 1M docs) | Zero-defect chemistry for strategic mineral refining |

---

## 📁 Monorepo Directory Layout

```
lexsi-hackathon/
├── README.md                                # Master hackathon landing page & overview
├── .gitignore                               # Clean environment exclusions
│
├── track1_problem5_aeroextract/             # Track 1, Problem 5 Project
│   ├── README.md                            # Comprehensive AeroExtract documentation
│   ├── requirements.txt                     # Dependencies for AeroExtract
│   ├── src/
│   │   ├── schema.py                        # Pydantic models (FAA FAR 43.9)
│   │   ├── curatorkit_pipeline.py           # CuratorKIT ingestion & PII scrubber
│   │   ├── aligntune_training.py            # AlignTune LoRA recipe generator & ROI model
│   │   ├── auditkit_eval.py                 # AuditKIT benchmarking engine
│   │   └── run_pipeline.py                  # End-to-end master runner
│   ├── app/
│   │   └── app.py                           # Gradio interactive demo app
│   ├── notebooks/
│   │   └── train_aviation_tiny_aya.ipynb    # Google Colab GPU training notebook
│   ├── recipes/
│   │   └── aviation_tiny_aya_sft.yaml       # AlignTune SFT recipe
│   ├── data/                                # Curated aviation logs & holdout splits
│   ├── artifacts/                           # Dataset card & audit logs
│   └── tests/                               # 23 automated unit & integration tests
│
└── track1_problem8_hydrometallurgy/         # Track 1, Problem 8 Project
    ├── README.md                            # Comprehensive Hydrometallurgy documentation
    ├── HF_ARTICLE.md                        # Publication-ready Hugging Face Community Article
    ├── requirements.txt                     # Dependencies for Hydrometallurgy
    ├── pipeline.py                          # Master pipeline orchestrator
    ├── generate_rare_earth_dataset.py       # Asynchronous synthetic data generator
    ├── curate_rare_earth_dataset.py         # CuratorKIT MinHash & Stoichiometry pipeline
    ├── hf_dataset_sync.py                   # Hugging Face DatasetDict builder
    ├── train_aligntune_adapters.py          # AlignTune post-trainer (Raw vs Curated)
    ├── audit_comparative_models.py          # AuditKIT empirical benchmarking engine
    ├── raw_dataset.jsonl                    # 821 raw generated REE dialogues
    ├── curated_dataset.jsonl                # 776 CuratorKIT-curated dialogues (94.52% retention)
    ├── curation_rejected.jsonl              # 45 rejected samples with failure taxonomy
    ├── curation_manifest.json               # Full provenance trail
    ├── configs/                             # AlignTune YAML configs (Raw & Curated)
    ├── data/hf_dataset_dict/                # Packaged Hugging Face DatasetDict
    ├── models/adapters/                     # LoRA adapters for Tiny Aya (Raw & Curated)
    ├── results/                             # Comparative audit reports (JSON & Markdown)
    └── tests/                               # 11 automated verification tests
```

---

## ⚡ Quickstart

### 1. Clone the Monorepo
```bash
git clone https://github.com/rishi-cherukuri/lexsi-hackathon.git
cd lexsi-hackathon
```

### 2. Run Track 1, Problem 5 (AeroExtract)
```bash
cd track1_problem5_aeroextract
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_pipeline.py
python -m unittest discover -s tests -v
```

### 3. Run Track 1, Problem 8 (REE Hydrometallurgy)
```bash
cd ../track1_problem8_hydrometallurgy
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python pipeline.py --stage test
```

---

## 🔗 Artifact & Hub Links

- **GitHub Monorepo**: [https://github.com/rishi-cherukuri/lexsi-hackathon](https://github.com/rishi-cherukuri/lexsi-hackathon)
- **Hugging Face Model Collection**: *[Link to Model Collection]*
- **Hugging Face Dataset Collection**: *[Link to Dataset Collection]*
- **Hugging Face Community Article**: Included at [`track1_problem8_hydrometallurgy/HF_ARTICLE.md`](track1_problem8_hydrometallurgy/HF_ARTICLE.md)

---

## ⚖️ License
This project is licensed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.
