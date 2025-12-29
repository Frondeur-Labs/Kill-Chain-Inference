"""
Artifact loading utilities.

Loads all compiled artifacts generated in Phase A.
This module is infrastructure-only and should not contain logic.
"""

from pathlib import Path
import pickle
from typing import Dict
import numpy as np


class ArtifactStore:
    """
    Container for static artifacts.
    """

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir

        self.phase_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.ttp_labels: Dict[str, str] = {}
        self.mitigation_stats: Dict[str, dict] = {}

    def load(self) -> None:
        """
        Load all required artifacts into memory.
        """
        self.phase_embeddings = self._load("phase_embeddings.pkl")
        self.ttp_labels = self._load("ttp_labels.pkl")
        self.mitigation_stats = self._load("mitigation_stats.pkl")

    def _load(self, filename: str):
        path = self.artifact_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
