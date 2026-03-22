"""
router/load_adapter.py

Loads a LoRA adapter onto the Mistral 7B base model for inference.

Steps:
1. Look up the emotion in adapters/registry.json → get repo_id
2. Load base model (Mistral 7B 4-bit) with Unsloth
3. Apply the PEFT adapter from HF Hub
4. Return (model, tokenizer) ready for generation

GPU required — run this in Colab, not Codespaces.

Usage:
    from router.load_adapter import load_adapter

    model, tokenizer = load_adapter("empathy")
"""

import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "adapters" / "registry.json"
MAX_SEQ_LENGTH = 2048


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found at {REGISTRY_PATH}")
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def load_adapter(emotion: str):
    """
    Load the base model and apply the LoRA adapter for the given emotion.

    Args:
        emotion: emotion name matching a key in adapters/registry.json
                 e.g. "empathy"

    Returns:
        (model, tokenizer) — a PEFT model with the adapter applied,
        ready for model.generate() calls.

    Raises:
        KeyError: if emotion is not in the registry
        ValueError: if the adapter status is not "trained"
    """
    # --- 1. Registry lookup ---
    registry = _load_registry()

    if emotion not in registry:
        available = list(registry.keys())
        raise KeyError(
            f"Emotion '{emotion}' not found in registry. "
            f"Available: {available}"
        )

    entry = registry[emotion]

    if entry.get("status") != "trained":
        raise ValueError(
            f"Adapter for '{emotion}' has status '{entry.get('status')}', "
            f"expected 'trained'. Train and push the adapter first."
        )

    repo_id = entry["repo_id"]
    base_model_id = entry["base_model"]

    # --- 2. Load base model with Unsloth ---
    # Unsloth must be imported before peft/transformers (import order matters).
    # This is a GPU-only import — will fail in Codespaces without a GPU.
    from unsloth import FastLanguageModel

    print(f"Loading base model: {base_model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,       # auto-detect: float16 on T4, bfloat16 on A100
        load_in_4bit=True,
    )

    # --- 3. Apply the PEFT adapter from HF Hub ---
    print(f"Applying adapter: {repo_id}")
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, repo_id)

    # Put model in inference mode — disables dropout, enables faster kernels
    FastLanguageModel.for_inference(model)

    print(f"Adapter '{emotion}' loaded and ready.")
    return model, tokenizer
