"""
Transformer baseline path construction.

Used for comparison and interpretability only.
"""

import numpy as np
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity([a], [b])[0][0])


def build_paths(predicted_dict, encoder, max_paths: int = 3):
    """
    Build a small set of baseline kill-chain paths using
    per-phase transformer predictions.
    """
    phase_order = [
        "recon",
        "weapon",
        "delivery",
        "exploit",
        "install",
        "c2",
        "objectives",
    ]

    tech_emb_map = {}

    # --------------------------------------------------
    # Encode all unique techniques once
    # --------------------------------------------------
    for phase, preds in predicted_dict.items():
        for tech, _ in preds:
            key = (phase, tech)
            if key not in tech_emb_map:
                tech_emb_map[key] = encoder.encode(tech)

    # --------------------------------------------------
    # Enumerate phase-wise combinations
    # --------------------------------------------------
    all_combos = list(product(*[predicted_dict[p] for p in phase_order]))

    paths = []

    # --------------------------------------------------
    # Score baseline paths (progress-visible)
    # --------------------------------------------------
    for combo in tqdm(
        all_combos,
        desc="[Transformer] Building baseline paths",
        leave=False,
    ):
        score = 0.0
        path = []

        for i, (tech, prob) in enumerate(combo):
            phase_name = phase_order[i].capitalize()
            path.append((phase_name, tech, prob))
            score += prob

            if i > 0:
                prev_tech = combo[i - 1][0]
                score += cosine_sim(
                    tech_emb_map[(phase_order[i - 1], prev_tech)],
                    tech_emb_map[(phase_order[i], tech)],
                )

        paths.append((score, path))

    paths.sort(key=lambda x: x[0], reverse=True)
    return paths[:max_paths]
