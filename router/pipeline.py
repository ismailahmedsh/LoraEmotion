"""
router/pipeline.py

Top-level pipeline: message → route → load adapter → generate response.

If no emotion matches above the confidence threshold, the message is logged
to router/unknown_buffer.jsonl for future review (Phase 4 self-evolution).

GPU required for generation — run this in Colab.
The Router and embed steps are CPU-only and run anywhere.

Usage:
    from router.pipeline import Pipeline

    pipeline = Pipeline()
    result = pipeline.run("I feel like nobody understands me")
    print(result["emotion"])    # "empathy"
    print(result["confidence"]) # 0.73
    print(result["response"])   # "That sounds really isolating..."
"""

import json
import datetime
from pathlib import Path

from router.router import Router
from router.load_adapter import load_adapter, _load_registry

# Same Alpaca format the empathy adapter was trained on.
# Instruction describes the task; input is the user message; response is left
# blank so the model fills it in during generation.
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

UNKNOWN_BUFFER_PATH = Path(__file__).resolve().parent / "unknown_buffer.jsonl"
MAX_NEW_TOKENS = 256


class Pipeline:
    def __init__(self, threshold: float = 0.50):
        """
        Initialise the pipeline.

        threshold: passed to Router — minimum cosine similarity to accept a match.
        The model and tokenizer are loaded lazily on the first run() call.
        """
        self.router = Router(threshold=threshold)
        self._model = None
        self._tokenizer = None
        self._loaded_emotion: str | None = None

    def _load(self, emotion: str) -> None:
        """Load the adapter for emotion (skips if already loaded)."""
        if self._loaded_emotion == emotion:
            return
        self._model, self._tokenizer = load_adapter(emotion)
        self._loaded_emotion = emotion

    def _generate(self, emotion: str, message: str) -> str:
        """Format the prompt and generate a response."""
        registry = _load_registry()
        instruction = registry[emotion].get(
            "instruction",
            f"Respond appropriately for the '{emotion}' emotion.",
        )
        prompt = ALPACA_PROMPT.format(instruction, message, "")

        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )
        # Decode only the newly generated tokens (skip the prompt)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _log_unknown(self, message: str, confidence: float) -> None:
        """Append an unknown message to the buffer file for later review."""
        entry = {
            "message": message,
            "confidence": round(confidence, 4),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(UNKNOWN_BUFFER_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def run(self, message: str) -> dict:
        """
        Route a message and generate a response.

        Returns a dict with:
            emotion     — matched emotion or "unknown"
            confidence  — cosine similarity score (0–1)
            response    — generated text, or None if unknown
            routed      — True if an adapter was loaded, False if unknown
        """
        emotion, confidence = self.router.route(message)

        if emotion == "unknown":
            self._log_unknown(message, confidence)
            return {
                "emotion": "unknown",
                "confidence": confidence,
                "response": None,
                "routed": False,
            }

        self._load(emotion)
        response = self._generate(emotion, message)

        return {
            "emotion": emotion,
            "confidence": confidence,
            "response": response,
            "routed": True,
        }
