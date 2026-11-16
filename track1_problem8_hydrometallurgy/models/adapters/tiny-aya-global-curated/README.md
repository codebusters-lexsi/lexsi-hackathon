---
license: apache-2.0
base_model: CohereLabs/tiny-aya-global
tags:
- alignment
- aligntune
- cohere
- tiny-aya
- peft
- lora
- rare-earth-elements
- hydrometallurgy
---

# tiny-aya-global - REE Hydrometallurgy Adapter (CURATED)

This adapter was trained with **Lexsi AlignTune** on **Cohere Labs' CohereLabs/tiny-aya-global** as part of the **Lexsi AI Safety & Alignment Hackathon (Track 1: Problem 8)**.

## Experiment Setup
- **Dataset Split**: `curated`
- **Source Data**: `curated_dataset.jsonl`
- **Framework**: `Lexsi AlignTune` (TRL backend)
- **Method**: Parameter-Efficient LoRA (r=16, alpha=32)
- **Domain**: Rare Earth Element Solvent Extraction (Nd3+/Pr3+ separation, PC-88A equilibria, pH balancing)
