"""
Shared ATTACK-BERT encoder.

This encoder is used across all phases.
Do NOT duplicate this elsewhere.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class AttackBERTEncoder:
    """
    Singleton-style wrapper for ATTACK-BERT.
    """

    _model = None

    def __init__(self, model_name: str = "basel/ATTACK-BERT"):
        if AttackBERTEncoder._model is None:
            AttackBERTEncoder._model = SentenceTransformer(model_name)

        self.model = AttackBERTEncoder._model

    def encode(self, text: str) -> np.ndarray:
        """
        Encode input text into a normalized embedding.
        """
        vec = self.model.encode(text, show_progress_bar=False)
        vec = vec.astype("float32")
        vec /= np.linalg.norm(vec)
        return vec
