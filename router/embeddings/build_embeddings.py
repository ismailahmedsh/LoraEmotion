"""
router/embeddings/build_embeddings.py

Reads a JSONL file of training examples for one emotion, embeds every
"instruction" field using MiniLM-L6, and saves the mean vector as a .npy file.

The mean vector is the centroid of all example embeddings — a single point
that represents "what a message triggering this emotion typically looks like".
The router uses these centroids to score incoming messages.

Usage (run from repo root):
    python router/embeddings/build_embeddings.py --emotion empathy

Output:
    router/embeddings/empathy.npy   (shape: (384,), dtype: float32)

Run this once per emotion after adding or updating training data.
CPU only — no GPU needed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running directly from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from router.embed import embed

EMBEDDINGS_DIR = Path(__file__).resolve().parent


def load_instructions(emotion: str) -> list[str]:
    """Load all instruction strings from the emotion's JSONL file."""
    path = Path(f"data/{emotion}_examples.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    instructions = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}")
            if "instruction" not in row:
                raise ValueError(f"Missing 'instruction' field on line {i} of {path}")
            instructions.append(row["instruction"])

    return instructions


def build(emotion: str) -> None:
    print(f"Loading instructions for '{emotion}'...")
    instructions = load_instructions(emotion)
    print(f"  {len(instructions)} examples found")

    print("Embedding instructions (this takes ~10s on CPU)...")
    vectors = np.stack([embed(text) for text in instructions])
    # vectors.shape == (n_examples, 384)

    centroid = vectors.mean(axis=0)
    # centroid.shape == (384,)

    out_path = EMBEDDINGS_DIR / f"{emotion}.npy"
    np.save(out_path, centroid)
    print(f"Saved centroid to {out_path}")
    print(f"  Shape: {centroid.shape}, dtype: {centroid.dtype}")
    print(f"  L2 norm: {np.linalg.norm(centroid):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emotion",
        type=str,
        default="empathy",
        help="Which emotion to build embeddings for (default: empathy)",
    )
    args = parser.parse_args()
    build(args.emotion)
