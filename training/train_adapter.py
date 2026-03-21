"""
training/train_adapter.py

Fine-tunes a LoRA adapter on Mistral 7B for a single emotion (default: empathy).
Saves the adapter locally to outputs/<emotion>/.

Run this in Google Colab (requires GPU + unsloth).
Do NOT run in Codespaces — unsloth needs CUDA.

Usage (in Colab):
    !python training/train_adapter.py --emotion empathy
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on the path so `from training.config import` works
# whether this script is run as `python training/train_adapter.py` or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_from_disk

# Unsloth and trl are Colab-only dependencies (not in requirements.txt).
# If you see an ImportError here in Codespaces, that is expected.
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

from training.config import (
    MODEL_ID,
    MAX_SEQ_LENGTH,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    WARMUP_STEPS,
    MAX_STEPS,
    LEARNING_RATE,
    OPTIMIZER,
    LR_SCHEDULER,
    SEED,
    OUTPUT_DIR,
)

# The sentinel written into the dataset by data/prepare.py.
# We replace it here with the actual tokenizer EOS token once we have it loaded.
EOS_SENTINEL = "<|EOS|>"


def load_model_and_tokenizer():
    """Load the base model and tokenizer via Unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,           # auto-detect: float16 on T4, bfloat16 on A100
        load_in_4bit=True,    # 4-bit quantization — essential for fitting 7B on T4
    )
    return model, tokenizer


def add_lora(model):
    """Attach LoRA adapters to the model via Unsloth's wrapper.

    Unsloth manages LoraConfig internally — do not pass a LoraConfig object here.
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's memory-efficient checkpointing
        random_state=SEED,
    )
    return model


def load_dataset(emotion: str, eos_token: str):
    """Load the processed HF Dataset and substitute the EOS sentinel."""
    dataset_path = Path(f"data/processed/{emotion}")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {dataset_path}. "
            f"Run: python data/prepare.py --emotion {emotion}"
        )

    dataset = load_from_disk(str(dataset_path))

    # Replace the placeholder with the real EOS token for this tokenizer.
    # This is done here (not in prepare.py) because the EOS token is model-specific.
    dataset = dataset.map(
        lambda row: {"text": row["text"].replace(EOS_SENTINEL, eos_token)},
        desc="Substituting EOS token",
    )

    print(f"Loaded {len(dataset)} examples from {dataset_path}")
    print(f"Sample (first 300 chars):\n{dataset[0]['text'][:300]}...\n")
    return dataset


def train(emotion: str) -> None:
    print(f"=== EmotionLoRA Training: {emotion} ===\n")

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    print("Attaching LoRA adapters...")
    model = add_lora(model)

    print("Loading dataset...")
    dataset = load_dataset(emotion, tokenizer.eos_token)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            optim=OPTIMIZER,
            lr_scheduler_type=LR_SCHEDULER,
            seed=SEED,
            output_dir=str(output_dir),
            logging_steps=10,
            save_strategy="no",       # adapter saved manually below
            fp16=True,                # float16 for T4
            report_to="none",         # no wandb/tensorboard
        ),
    )

    print("Starting training...")
    trainer.train()

    print(f"\nSaving adapter to {output_dir}/...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("Done. Adapter saved locally.")
    print(f"Next step: run training/push_to_hub.py to upload to HuggingFace Hub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emotion",
        type=str,
        default="empathy",
        help="Which emotion adapter to train (default: empathy)",
    )
    args = parser.parse_args()
    train(args.emotion)
