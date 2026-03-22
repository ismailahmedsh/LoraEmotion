"""
router/embed.py

Embeds a text string using MiniLM-L6-v2 (sentence-transformers).
Returns a 384-dimensional numpy array — the semantic "fingerprint" of the text.

This runs on CPU. No GPU needed.

Usage:
    from router.embed import embed
    vec = embed("I feel really overwhelmed today")
    # vec.shape == (384,)
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Lazy-loaded singleton — model loads once, reused for all calls.
_model: SentenceTransformer | None = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray:
    """
    Embed a single string.

    Returns a numpy array of shape (384,) with float32 values.
    The vector is NOT normalised here — normalisation happens in router.py
    during cosine similarity scoring.
    """
    if not isinstance(text, str):
        raise TypeError(f"embed() expects a str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError("embed() received an empty string")

    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True)
    return vec.astype(np.float32)
