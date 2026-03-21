"""
data/prepare.py

Converts a raw JSONL file of empathy examples into a HuggingFace Dataset
with a single "text" column formatted for Unsloth/SFTTrainer.

Usage:
    python data/prepare.py --emotion empathy

Output:
    data/processed/empathy/   (HuggingFace Dataset directory)
"""

import argparse
import json
from pathlib import Path
from datasets import Dataset


# Alpaca-style prompt template — same format used in Unsloth's official notebooks
# Source: github.com/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Alpaca.ipynb
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# EOS token placeholder — replaced at runtime with the actual tokenizer EOS token.
# During data prep we don't have the tokenizer loaded, so we use a sentinel
# that train_adapter.py will substitute before training.
EOS_SENTINEL = "<|EOS|>"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}")
            for field in ("instruction", "input", "output"):
                if field not in row:
                    raise ValueError(f"Missing field '{field}' on line {i} of {path}")
            rows.append(row)
    return rows


def format_example(row: dict) -> str:
    return ALPACA_PROMPT.format(
        row["instruction"],
        row["input"],
        row["output"],
    ) + EOS_SENTINEL


def prepare(emotion: str) -> None:
    raw_path = Path(f"data/{emotion}_examples.jsonl")
    out_path = Path(f"data/processed/{emotion}")

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    rows = load_jsonl(raw_path)
    print(f"Loaded {len(rows)} examples from {raw_path}")

    texts = [format_example(row) for row in rows]
    dataset = Dataset.from_dict({"text": texts})

    out_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_path))
    print(f"Saved dataset to {out_path}/")
    print(f"Dataset schema: {dataset.features}")
    print(f"First example preview:\n{texts[0][:300]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emotion",
        type=str,
        default="empathy",
        help="Which emotion to prepare (default: empathy)",
    )
    args = parser.parse_args()
    prepare(args.emotion)
