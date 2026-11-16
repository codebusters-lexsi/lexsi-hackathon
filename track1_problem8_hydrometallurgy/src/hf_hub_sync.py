"""
Hugging Face Hub Sync Module
Pushes all artifacts (Datasets, Models, Adapters, Logs, Metrics) to Hugging Face.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from config import config


def get_hf_api(token: Optional[str] = None) -> HfApi:
    token = token or config.hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "Hugging Face token not found. Please set HF_TOKEN in your environment or .env file."
        )
    return HfApi(token=token)


def upload_dataset_to_hf(
    curated_dir: str,
    repo_id: Optional[str] = None,
    private: bool = False
) -> str:
    """Uploads the curated dataset and documentation to Hugging Face Dataset Hub."""
    token = config.hf_token or os.getenv("HF_TOKEN")
    api = get_hf_api(token)
    repo_id = repo_id or config.hf_dataset_repo
    
    print(f"\n📦 [Hugging Face Sync] Uploading Dataset Artifacts to: {repo_id}")
    
    # Create dataset repo if it doesn't already exist
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        private=private,
        exist_ok=True
    )

    # Upload all files in curated directory (curated_messages.jsonl, rejected.jsonl, README.md, manifest)
    upload_folder(
        folder_path=curated_dir,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Add Lexsi CuratorKIT curated alignment dataset (conversational messages JSONL)"
    )

    hf_url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"✅ Successfully uploaded dataset to: {hf_url}")
    return hf_url


def upload_model_artifacts_to_hf(
    artifacts_dir: str,
    repo_id: Optional[str] = None,
    private: bool = False
) -> str:
    """Uploads trained model adapters, checkpoints, logs, and evaluation metrics to Hugging Face Model Hub."""
    token = config.hf_token or os.getenv("HF_TOKEN")
    api = get_hf_api(token)
    repo_id = repo_id or config.hf_model_repo

    print(f"\n🚀 [Hugging Face Sync] Uploading Model & Adapter Artifacts to: {repo_id}")

    create_repo(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        private=private,
        exist_ok=True
    )

    upload_folder(
        folder_path=artifacts_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Add AlignTune post-trained adapter weights, training logs, and AuditKIT evaluations"
    )

    hf_url = f"https://huggingface.co/models/{repo_id}"
    print(f"✅ Successfully uploaded model artifacts to: {hf_url}")
    return hf_url


if __name__ == "__main__":
    print("HF Hub Sync utility ready.")
