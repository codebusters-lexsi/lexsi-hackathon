"""
End-to-End Test Suite for Lexsi Hackathon Pipeline (Track 1, Problem 8)
Verifies:
  1. Dataset schemas & conversational formatting (raw & curated)
  2. CuratorKIT rejection logs and provenance manifest
  3. Hugging Face DatasetDict serialization and loading
  4. AlignTune configuration files and adapter artifacts
  5. AuditKIT evaluation reports and metric consistency
"""

import os
import json
import yaml
import pytest
from pathlib import Path
from datasets import load_from_disk

BASE_DIR = Path(__file__).resolve().parent.parent


def test_raw_dataset():
    raw_path = BASE_DIR / "raw_dataset.jsonl"
    assert raw_path.exists(), "raw_dataset.jsonl must exist"
    count = 0
    with open(raw_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            entry = json.loads(line)
            assert "messages" in entry, f"Sample {idx} missing 'messages'"
            msgs = entry["messages"]
            assert len(msgs) >= 2, f"Sample {idx} must have >= 2 messages"
            assert msgs[0]["role"] == "user"
            assert msgs[1]["role"] == "assistant"
            assert len(msgs[0]["content"].strip()) > 0
            assert len(msgs[1]["content"].strip()) > 0
            count += 1
    assert count >= 800, f"Expected at least 800 raw samples, found {count}"


def test_curated_dataset():
    curated_path = BASE_DIR / "curated_dataset.jsonl"
    assert curated_path.exists(), "curated_dataset.jsonl must exist"
    count = 0
    with open(curated_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            entry = json.loads(line)
            assert "messages" in entry, f"Curated sample {idx} missing 'messages'"
            msgs = entry["messages"]
            assert len(msgs) >= 2
            assert msgs[0]["role"] == "user"
            assert msgs[1]["role"] == "assistant"
            count += 1
    assert count >= 700, f"Expected at least 700 curated samples, found {count}"


def test_curation_manifest_and_rejected():
    manifest_path = BASE_DIR / "curation_manifest.json"
    rejected_path = BASE_DIR / "curation_rejected.jsonl"
    assert manifest_path.exists(), "curation_manifest.json must exist"
    assert rejected_path.exists(), "curation_rejected.jsonl must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "retention_rate" in manifest
    assert "total_raw_evaluated" in manifest
    assert "total_curated_retained" in manifest

    with open(rejected_path, "r", encoding="utf-8") as f:
        rej_count = sum(1 for line in f if line.strip())
    assert rej_count == manifest["total_rejected"], "Rejected log count must match manifest"


def test_dataset_dict_loading():
    dd_path = BASE_DIR / "data" / "hf_dataset_dict"
    assert dd_path.exists(), "HF DatasetDict folder must exist"
    dd = load_from_disk(str(dd_path))
    assert "raw" in dd and "curated" in dd, "DatasetDict must contain 'raw' and 'curated' splits"
    assert len(dd["raw"]) >= 800
    assert len(dd["curated"]) >= 700
    assert "messages" in dd["curated"].column_names


def test_aligntune_configs():
    raw_cfg = BASE_DIR / "configs" / "aligntune_tiny_aya_raw.yaml"
    curated_cfg = BASE_DIR / "configs" / "aligntune_tiny_aya_curated.yaml"
    assert raw_cfg.exists(), "Raw config YAML must exist"
    assert curated_cfg.exists(), "Curated config YAML must exist"

    for p in [raw_cfg, curated_cfg]:
        with open(p, "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
        assert c["model"]["name_or_path"] == "CohereLabs/tiny-aya-global"
        assert c["peft"]["peft_type"] == "LORA"
        assert c["peft"]["r"] == 16
        assert c["peft"]["lora_alpha"] == 32
        assert "target_modules" in c["peft"]


def test_adapters_artifacts():
    for name in ["tiny-aya-global-raw", "tiny-aya-global-curated"]:
        adapter_dir = BASE_DIR / "models" / "adapters" / name
        assert adapter_dir.exists(), f"Adapter directory {name} must exist"
        assert (adapter_dir / "adapter_config.json").exists(), f"adapter_config.json missing in {name}"
        assert (adapter_dir / "adapter_model.bin").exists(), f"adapter_model.bin missing in {name}"
        assert (adapter_dir / "README.md").exists(), f"Model Card missing in {name}"
        assert (adapter_dir / "training_meta.json").exists(), f"training_meta.json missing in {name}"


def test_audit_report_consistency():
    report_path = BASE_DIR / "results" / "comparative_audit_report.json"
    summary_path = BASE_DIR / "results" / "comparative_audit_summary.md"
    assert report_path.exists(), "comparative_audit_report.json must exist"
    assert summary_path.exists(), "comparative_audit_summary.md must exist"

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert "metrics" in rep
    metrics = rep["metrics"]
    assert metrics["overall_audit_score"]["curated_adapter"] > metrics["overall_audit_score"]["raw_adapter"]
    assert metrics["hallucination_rate"]["curated_adapter"] < metrics["hallucination_rate"]["raw_adapter"]


def test_curatorkit_import():
    import curatorkit
    from curatorkit import CuratorConfig
    from curatorkit.normalizers import MinHashDeduplicator, ExactDeduplicator
    assert curatorkit.__version__ is not None


def test_curation_heuristics_unit():
    """Unit tests for the chemical and structure checking functions."""
    from curate_rare_earth_dataset import (
        get_shingles,
        jaccard_similarity,
        check_length_and_structure,
        check_chemistry_and_stoichiometry,
    )

    # 1. Shingling & Jaccard
    s1 = get_shingles("neodymium praseodymium solvent extraction", k=3)
    s2 = get_shingles("neodymium praseodymium solvent extraction", k=3)
    assert jaccard_similarity(s1, s2) == 1.0

    s3 = get_shingles("completely unrelated sentence about cooking recipes and pasta", k=3)
    assert jaccard_similarity(s1, s3) == 0.0

    # 2. Length & Structure
    short_user = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    ok, err = check_length_and_structure(short_user)
    assert not ok and "user_prompt_too_short" in err

    long_unclosed_body = (
        "Here is a detailed explanation of the solvent extraction process for separating neodymium and praseodymium. "
        * 10
        + "```python def run_extraction():\n    pass\n"
    )
    unclosed_code = [
        {"role": "user", "content": "A sufficiently long user prompt explaining the extraction question in detail here."},
        {"role": "assistant", "content": long_unclosed_body}
    ]
    ok, err = check_length_and_structure(unclosed_code)
    assert not ok and "unclosed_markdown" in err

    # 3. Chemistry Verification
    valid_text = "Separation of Nd3+ and Pr3+ using PC-88A at equilibrium pH 3.2 with distribution ratio D_Nd."
    ok, err = check_chemistry_and_stoichiometry(valid_text)
    assert ok and err is None

    invalid_text = "This text talks about gold and silver extraction using cyanide with no lanthanides mentioned."
    ok, err = check_chemistry_and_stoichiometry(invalid_text)
    assert not ok and "missing_core_domain_concept" in err


def test_hf_dataset_card_and_article():
    """Verifies that the Hugging Face Dataset Card and Community Article are present and well-formed."""
    card_path = BASE_DIR / "data" / "hf_dataset_dict" / "README.md"
    article_path = BASE_DIR / "HF_ARTICLE.md"
    readme_path = BASE_DIR / "README.md"

    assert card_path.exists(), "HF Dataset Card README.md must exist"
    assert article_path.exists(), "HF_ARTICLE.md must exist"
    assert readme_path.exists(), "Root README.md must exist"

    card_content = card_path.read_text(encoding="utf-8")
    article_content = article_path.read_text(encoding="utf-8")
    readme_content = readme_path.read_text(encoding="utf-8")

    # Frontmatter and key markers
    assert card_content.startswith("---"), "Dataset Card must have YAML frontmatter"
    assert "Cohere" in card_content or "cohere" in card_content
    assert "CuratorKIT" in card_content

    assert article_content.startswith("---"), "HF_ARTICLE.md must have YAML frontmatter"
    assert "Track 1" in article_content
    assert "Problem 8" in article_content
    assert "CuratorKIT" in article_content
    assert "AlignTune" in article_content
    assert "AuditKIT" in article_content

    assert "Empirical Delta" in article_content or "Relative Gain" in readme_content


def test_pipeline_cli_help():
    """Verifies that pipeline.py CLI interface responds to --help cleanly."""
    import subprocess
    import sys
    res = subprocess.run([sys.executable, str(BASE_DIR / "pipeline.py"), "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Lexsi Track 1 Problem 8 Master Pipeline" in res.stdout

