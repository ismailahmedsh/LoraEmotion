"""
router/router.py

Routes an incoming message to the best-matching emotion adapter.

How it works:
1. Loads precomputed centroid vectors from router/embeddings/*.npy
2. Embeds the incoming message using MiniLM-L6
3. Computes cosine similarity between the message and each centroid
4. Returns the top match if confidence >= threshold, else "unknown"

Runs on CPU. No GPU needed.

Usage:
    from router.router import Router

    router = Router()
    emotion, confidence = router.route("I feel like nobody listens to me")
    # emotion == "empathy", confidence == 0.72 (example)
"""

from pathlib import Path

import numpy as np

from router.embed import embed

EMBEDDINGS_DIR = Path(__file__).resolve().parent / "embeddings"
DEFAULT_THRESHOLD = 0.50


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Returns a float in [-1, 1]. Higher = more similar.
    1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class Router:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        """
        Load all centroid vectors from router/embeddings/*.npy.

        threshold: minimum cosine similarity to accept a match.
                   Below this → returns "unknown".
        """
        self.threshold = threshold
        self.centroids: dict[str, np.ndarray] = {}
        self._load_centroids()

    def _load_centroids(self) -> None:
        npy_files = list(EMBEDDINGS_DIR.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(
                f"No centroid files found in {EMBEDDINGS_DIR}. "
                "Run router/embeddings/build_embeddings.py first."
            )
        for path in npy_files:
            emotion = path.stem  # e.g. "empathy.npy" → "empathy"
            self.centroids[emotion] = np.load(path).astype(np.float32)
        print(f"Router loaded {len(self.centroids)} emotion(s): {list(self.centroids)}")

    def route(self, message: str) -> tuple[str, float]:
        """
        Route a message to the best-matching emotion.

        Returns:
            (emotion, confidence) where emotion is a key from registry.json,
            or ("unknown", best_score) if no match exceeds the threshold.
        """
        message_vec = embed(message)

        scores: dict[str, float] = {
            emotion: _cosine_similarity(message_vec, centroid)
            for emotion, centroid in self.centroids.items()
        }

        best_emotion = max(scores, key=lambda e: scores[e])
        best_score = scores[best_emotion]

        if best_score >= self.threshold:
            return best_emotion, best_score
        return "unknown", best_score

    def scores(self, message: str) -> dict[str, float]:
        """
        Return similarity scores for all emotions — useful for debugging.

        Example:
            {"empathy": 0.71, "curiosity": 0.38}
        """
        message_vec = embed(message)
        return {
            emotion: _cosine_similarity(message_vec, centroid)
            for emotion, centroid in self.centroids.items()
        }
