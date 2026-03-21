"""
training/push_to_hub.py

Pushes a locally saved LoRA adapter to HuggingFace Hub.
Run this in Colab after train_adapter.py has finished.

Prerequisites:
    1. training/train_adapter.py has run and saved the adapter to outputs/<emotion>/
    2. A .env file exists with HF_TOKEN=your_write_token
    3. HF_REPO in training/config.py has been updated with your real HF username

Usage (in Colab):
    !python training/push_to_hub.py --emotion empathy
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from training.config import MODEL_ID, MAX_SEQ_LENGTH, HF_REPO, OUTPUT_DIR


def load_token() -> str:
    """Load HF write token from .env file."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN not found. Create a .env file with:\n"
            "    HF_TOKEN=hf_your_write_token_here\n"
            "Get a write token at: huggingface.co/settings/tokens"
        )
    return token


def push(emotion: str) -> None:
    adapter_dir = Path(OUTPUT_DIR)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_dir}. "
            f"Run training/train_adapter.py --emotion {emotion} first."
        )

    token = load_token()

    # Derive the repo ID: replace the placeholder emotion with the actual one
    # if the user has a custom OUTPUT_DIR structure, they can edit HF_REPO directly.
    repo_id = HF_REPO

    print(f"=== EmotionLoRA Push to Hub: {emotion} ===\n")
    print(f"Local adapter: {adapter_dir}")
    print(f"Target repo:   {repo_id}\n")

    print("Loading base model + adapter...")
    # Load just enough to get the model object with the adapter attached.
    # We do NOT merge weights — the adapter is pushed separately so the base
    # model stays untouched on the Hub.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print("Pushing adapter weights to Hub...")
    model.push_to_hub(repo_id, token=token)

    print("Pushing tokenizer to Hub...")
    tokenizer.push_to_hub(repo_id, token=token)

    print(f"\nDone. Adapter live at: https://huggingface.co/{repo_id}")
    print("Next step: add an entry to adapters/registry.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emotion",
        type=str,
        default="empathy",
        help="Which emotion adapter to push (default: empathy)",
    )
    args = parser.parse_args()
    push(args.emotion)
