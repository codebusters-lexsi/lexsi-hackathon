# Lexsi AI Safety & Alignment Hackathon
**Track 1: Fine-Tuning & Data Curation — Problem 8: Data-Scarce Domains**  
*Organized by Lexsi Labs × Cohere Labs × Hugging Face*

## 🎯 Thesis
In data-scarce industrial engineering domains, raw synthetic generation produces plausible-sounding but chemically dangerous hallucinations. By chaining **Lexsi CuratorKIT** (data curation/filtering), **Lexsi AlignTune** (post-training on Cohere dense open-weights models), and **Lexsi AuditKIT** (evaluating and benchmarking), we empirically demonstrate that **curated synthetic data measurably outperforms raw generated data** (96.1% vs 53.6% scientific accuracy).

---

## 🔬 Domain: Rare Earth Element (REE) Hydrometallurgy
- **Focus**: Solvent extraction parameters for separating adjacent lanthanides (**Neodymium $\text{Nd}^{3+}$ from Praseodymium $\text{Pr}^{3+}$**).
- **Core Chemistry**: Dimeric extraction mechanisms with PC-88A / Cyanex 272 / P507 ($\text{Ln}^{3+} + 3(\text{HA})_2 \rightleftharpoons \text{Ln}(\text{HA}_2)_3 + 3\text{H}^+$).
- **Operating Windows**: Separation factor $\beta_{\text{Nd}/\text{Pr}} \approx 1.3 - 1.5$, saponification degree $30-40\%$, equilibrium pH $2.8 - 3.6$, scrub liquor $0.2-0.6\text{ M }\text{HCl}$, and strip liquor $2.0-3.5\text{ M }\text{HCl}$.
- **Industrial Troubleshooting**: Interphase crud remediation (colloidal silica/iron), phase inversion control, and third-phase avoidance using isodecanol/TBP modifiers.

---

## 📁 Repository Structure

```
├── raw_dataset.jsonl              # 821 raw generated REE dialogues (Cohere Command R+)
├── curated_dataset.jsonl          # 776 CuratorKIT-curated high-purity dialogues
├── curation_rejected.jsonl        # 45 rejected samples with failure taxonomy
├── curation_manifest.json         # Provenance trail & curation retention statistics (94.52%)
├── generate_rare_earth_dataset.py # Step 2: Asynchronous synthetic data generator
├── curate_rare_earth_dataset.py   # Step 3: Lexsi CuratorKIT filtering pipeline
├── hf_dataset_sync.py             # Step 4: Hugging Face DatasetDict builder & Hub sync
├── train_aligntune_adapters.py    # Step 5: Lexsi AlignTune trainer for raw vs curated LoRA
├── audit_comparative_models.py    # Step 5: Lexsi AuditKIT empirical benchmarking
├── HF_ARTICLE.md                  # Complete publication article for Hugging Face Hub
│
├── configs/
│   ├── aligntune_tiny_aya_raw.yaml      # AlignTune config for raw baseline adapter
│   └── aligntune_tiny_aya_curated.yaml  # AlignTune config for curated adapter
│
├── data/
│   └── hf_dataset_dict/           # Packaged Hugging Face DatasetDict (splits: 'raw' & 'curated')
│       ├── raw/
│       ├── curated/
│       └── README.md              # Official Dataset Card
│
├── models/
│   └── adapters/
│       ├── tiny-aya-global-raw/     # Adapter A: trained on raw data
│       └── tiny-aya-global-curated/ # Adapter B: trained on curated data
│
└── results/
    ├── comparative_audit_report.json    # Machine-readable evaluation report
    └── comparative_audit_summary.md     # Comparative benchmark matrix table
```

---

## ⚡ Step-by-Step Reproduction Guide

### Step 1: Environment Setup
```bash
$HOME/.local/bin/uv venv --python 3.11 venv
./venv/bin/pip install openai datasets huggingface_hub cohere pyarrow pandas pydantic python-dotenv
./venv/bin/pip install git+https://github.com/Lexsi-Labs/CuratorKIT.git
```

### Step 2: Synthetic Generation
```bash
./venv/bin/python generate_rare_earth_dataset.py --samples 1000 --concurrency 5 --provider cohere
```

### Step 3: CuratorKIT Curation (MinHash Dedup, LengthFilter, Stoichiometry Gate)
```bash
./venv/bin/python curate_rare_earth_dataset.py --raw raw_dataset.jsonl --curated curated_dataset.jsonl
```

### Step 4: Hugging Face DatasetDict Sync
```bash
./venv/bin/python hf_dataset_sync.py
```

### Step 5: AlignTune Post-Training & AuditKIT Evaluation
```bash
# Package/train both adapters on Cohere Tiny Aya Global
./venv/bin/python train_aligntune_adapters.py --split both

# Run empirical benchmark comparing raw vs curated
./venv/bin/python audit_comparative_models.py
```

---

## 📊 Summary of Results

| Benchmark Metric | Base Cohere Tiny Aya | Raw Post-Trained | Curated (CuratorKIT) | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Stoichiometry & Reaction Precision | 42.0% | 61.5% | **96.2%** | **+56.4%** |
| Separation Factor ($\beta$) Realism | 38.0% | 54.0% | **98.5%** | **+82.4%** |
| Saponification Chemical Validity | 35.0% | 48.0% | **97.0%** | **+102.1%** |
| Hallucination Rate | 48.0% | 32.5% | **4.2%** | **-87.1% reduction** |
| **Overall Scientific Score** | **42.8%** | **53.6%** | **96.1%** | **+79.3% overall gain** |
