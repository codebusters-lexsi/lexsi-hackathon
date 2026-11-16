---
tags:
- curatorkit
configs:
- config_name: sft_alpaca
  default: true
  description: curatorkit sft_alpaca export, sha256 09b764ebe5aa4eb579a81debc694f42a1f31f1dc8123f556df77a11cbfe49d62
  data_files:
  - split: test
    path: test/sft_alpaca.jsonl
  - split: train
    path: train/sft_alpaca.jsonl
  - split: val
    path: val/sft_alpaca.jsonl
- config_name: sft_messages
  description: curatorkit sft_messages export, sha256 60ae90d300d1f92e97c5f58a64d5c326ee4bfcde579471263e4770907b2c034f
  data_files:
  - split: test
    path: test/sft_messages.jsonl
  - split: train
    path: train/sft_messages.jsonl
  - split: val
    path: val/sft_messages.jsonl
---

# Dataset Card: lexsi-aviation-maintenance-extractions

## Summary

- **Run timestamp**: 2026-11-16T03:29:55.422323Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `cb547af89e3b7cfe`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 200 | 200 | — | 0 | 200 |
| SchemaGate | 200 | 200 | 0 | 0 | 200 |
| ExactDeduplicator | 200 | 200 | — | — | 200 |
| TextCleaner | 200 | 200 | — | — | 200 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 200/0 passed (N/A), 0 rejected
- **SchemaGate**: 200/200 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - exact_duplicates_removed: 0

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.21
- **python**: 3.9.6

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `sft_alpaca` | test (30), train (140), val (30) | `instruction`: string<br>`input`: string (empty when there is no input)<br>`output`: string |
| `sft_messages` | test (30), train (140), val (30) | `messages`: list of {role, content}; role is system, user or assistant |
