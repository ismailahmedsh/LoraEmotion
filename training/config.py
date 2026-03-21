"""
training/config.py

All training hyperparameters in one place.
Edit this file to tune the training run — do not hardcode values in train_adapter.py.

NOTE: HF_REPO must be updated with your actual HuggingFace username before pushing.
"""

# --- Model ---
MODEL_ID = "unsloth/mistral-7b-v0.3-bnb-4bit"
# Pre-quantized Mistral 7B hosted by Unsloth. Faster to download than the
# raw mistralai version because the 4-bit quantization is already done.
MAX_SEQ_LENGTH = 2048

# --- LoRA ---
LORA_R = 16
# Rank: controls the size of the LoRA weight matrices added to the model.
# Higher rank = more expressive adapter, more memory, slower training.
# 16 is a good default for style/personality fine-tuning.

LORA_ALPHA = 16
# Scaling factor for LoRA updates. Rule of thumb: set equal to LORA_R.
# The effective scale = LORA_ALPHA / LORA_R = 1.0 at these values.

LORA_DROPOUT = 0.0
# Dropout rate during training. 0.0 keeps Unsloth's Triton kernel optimizations
# active. Only increase if the model overfits badly.

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
# Which layers inside Mistral to attach LoRA to.
# These are the attention layers (q/k/v/o) and the feed-forward layers (gate/up/down).
# Targeting all 7 gives the adapter the most influence over the model's behavior.

# --- Training ---
BATCH_SIZE = 2
# Examples processed per GPU step. Keep at 2 for Colab T4 (15GB VRAM).

GRAD_ACCUM_STEPS = 4
# Gradient accumulation: simulate a larger batch without using more memory.
# Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 8.

WARMUP_STEPS = 5
# Steps where the learning rate ramps up from 0 to LEARNING_RATE.
# Prevents large, destabilizing updates at the very start of training.

MAX_STEPS = 60
# Total training steps. At effective batch 8 with 61 examples,
# one full pass (epoch) ≈ 8 steps. 60 steps ≈ 7-8 epochs.
# Enough to learn the empathy style without overfitting.

LEARNING_RATE = 2e-4
# How large each weight update is. 2e-4 is standard for LoRA fine-tuning.
# Too high = unstable training. Too low = barely learns.

OPTIMIZER = "adamw_8bit"
# 8-bit AdamW from bitsandbytes. Same as regular AdamW but uses 8x less
# memory for the optimizer state. Essential on T4 with a 7B model.

LR_SCHEDULER = "linear"
# Learning rate schedule. "linear" decays LR from LEARNING_RATE to 0
# over MAX_STEPS. Simple and reliable.

SEED = 3407
# Random seed for reproducibility. 3407 is Unsloth's default.

# --- Output ---
OUTPUT_DIR = "outputs/empathy"
# Where to save the adapter locally after training (in Colab).

HF_REPO = "ismailahmedsh/emotionlora-empathy"
# HuggingFace Hub repo to push the adapter to.
# Replace YOUR_HF_USERNAME with your actual HF username before running push_to_hub.py.
