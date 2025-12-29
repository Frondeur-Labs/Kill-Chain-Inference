"""
Phase-wise prior inference using transformer models.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from core.priors.phase_models import load_phase_model


@torch.no_grad()
def predict_topk_for_phase(
    phase_name: str,
    text_embedding: np.ndarray,
    model_root,
    device,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Predict top-K TTPs for a given phase.
    """
    input_dim = text_embedding.shape[0]
    model, le = load_phase_model(
        phase_name,
        input_dim,
        model_root,
        device,
    )

    x = torch.tensor(text_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    idxs = probs.argsort()[::-1][:k]
    return [(le.classes_[i], float(probs[i])) for i in idxs]
