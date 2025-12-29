"""
Symbolic kill-chain decision model.

This module defines:
- Phase-to-phase transition dynamics over ATT&CK techniques
- Multi-objective reward computation for partial and complete kill chains
- Greedy rollout generation for long-horizon value estimation

The MDP acts as a symbolic teacher for training the Policy–Value Network.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


# --------------------------------------------------
# Utility
# --------------------------------------------------
def sim01(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalized cosine similarity in [0, 1].
    """
    if a is None or b is None:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    cos = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


# --------------------------------------------------
# Transition Table Construction
# --------------------------------------------------
def build_transition_table(
    phase_embeddings: Dict[str, Dict[str, np.ndarray]],
    phase_order: List[str],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Construct a soft transition kernel between techniques in adjacent phases
    using semantic similarity.
    """
    table: Dict[Tuple[str, str], Dict[str, float]] = {}

    for i in range(len(phase_order) - 1):
        src_phase = phase_order[i]
        dst_phase = phase_order[i + 1]

        src_map = phase_embeddings.get(src_phase, {})
        dst_map = phase_embeddings.get(dst_phase, {})

        for src_ttp, src_emb in src_map.items():
            sims = [(dst, sim01(src_emb, dst_emb)) for dst, dst_emb in dst_map.items()]
            scores = np.array([max(1e-8, s) for _, s in sims], dtype=np.float32)

            if scores.sum() > 0:
                probs = scores / scores.sum()
            else:
                probs = np.ones_like(scores) / len(scores)

            table[(src_ttp, src_phase)] = {
                sims[j][0]: float(probs[j]) for j in range(len(sims))
            }

    return table


# --------------------------------------------------
# Reward Weights (paper-consistent)
# --------------------------------------------------
ALPHA = 0.4  # relevance
BETA = 0.2  # cohesion
GAMMA = 0.4  # transition consistency

W_S = 1.0  # positive score weight
W_D = 1.0  # defensive penalty weight

S_d = 50.0  # detection scaling
S_m = 20.0  # mitigation scaling


# --------------------------------------------------
# Reward Computation
# --------------------------------------------------
def compute_reward_components(
    path: List[str],
    phase_order: List[str],
    transition_table: Dict[Tuple[str, str], Dict[str, float]],
    mitigation_stats: Dict[str, dict],
    phase1_probs: Dict[str, Dict[str, float]],
    phase_embeddings: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, float]:
    """
    Compute multi-objective reward components for a kill-chain path.
    """
    # Phase relevance
    rel_vals = [
        phase1_probs.get(phase_order[i], {}).get(ttp, 0.0) for i, ttp in enumerate(path)
    ]
    Rel = float(np.mean(rel_vals)) if rel_vals else 0.0

    # Semantic cohesion
    coh_vals = [
        sim01(
            phase_embeddings[phase_order[i - 1]].get(path[i - 1]),
            phase_embeddings[phase_order[i]].get(path[i]),
        )
        for i in range(1, len(path))
    ]
    Coh = float(np.mean(coh_vals)) if coh_vals else 0.0

    # Transition plausibility
    trans_vals = [
        transition_table.get((path[i], phase_order[i]), {}).get(path[i + 1], 0.0)
        for i in range(len(path) - 1)
    ]
    Trans = float(np.mean(trans_vals)) if trans_vals else 0.0

    # Defensive exposure
    det_count = 0
    mit_count = 0
    for ttp in path:
        stats = mitigation_stats.get(ttp, {})
        det_count += stats.get("detection_count", 0)
        mit_count += stats.get("mitigation_count", 0)

    D_pen = min(1.0, det_count / S_d)
    M_pen = min(1.0, mit_count / S_m)
    D_val = 1.0 - ((1.0 - D_pen) * (1.0 - M_pen))

    S_val = ALPHA * Rel + BETA * Coh + GAMMA * Trans
    R_val = W_S * S_val - W_D * D_val

    return {
        "Rel": Rel,
        "Coh": Coh,
        "Trans": Trans,
        "S": S_val,
        "Detection_Count": det_count,
        "Mitigation_Count": mit_count,
        "D": D_val,
        "R": float(max(-1.0, min(1.0, R_val))),
    }


# --------------------------------------------------
# MDP Wrapper
# --------------------------------------------------
class KillChainMDP:
    """
    Symbolic MDP over ATT&CK phases and techniques.

    Provides:
    - Reward evaluation
    - Transition probabilities
    - Greedy rollout for long-horizon value estimation
    """

    def __init__(
        self,
        phase_order: List[str],
        phase_embeddings: Dict[str, Dict[str, np.ndarray]],
        transition_table: Dict[Tuple[str, str], Dict[str, float]],
        mitigation_stats: Dict[str, dict],
    ):
        self.phase_order = phase_order
        self.phase_embeddings = phase_embeddings
        self.transition_table = transition_table
        self.mitigation_stats = mitigation_stats

    # --------------------------------------------------
    # Evaluation (used in main / analysis)
    # --------------------------------------------------
    def evaluate_path_verbose(self, path: List[str], phase1_probs):
        return compute_reward_components(
            path=path,
            phase_order=self.phase_order,
            transition_table=self.transition_table,
            mitigation_stats=self.mitigation_stats,
            phase1_probs=phase1_probs,
            phase_embeddings=self.phase_embeddings,
        )

    # --------------------------------------------------
    # Transition Access
    # --------------------------------------------------
    def transition_probs(self, ttp: str, phase_idx: int) -> Dict[str, float]:
        """
        Return transition probabilities from (phase_idx, ttp) to next phase.
        """
        if phase_idx >= len(self.phase_order) - 1:
            return {}
        phase = self.phase_order[phase_idx]
        return self.transition_table.get((ttp, phase), {})

    # --------------------------------------------------
    # Rollout (teacher signal for PV training)
    # --------------------------------------------------
    def rollout(
        self,
        start_ttp: str,
        start_phase_idx: int,
        phase1_probs: Dict[str, Dict[str, float]],
        max_depth: Optional[int] = None,
    ) -> float:
        """
        Greedy rollout until final phase to estimate long-horizon reward.
        """
        path = [start_ttp]
        phase_idx = start_phase_idx

        depth_limit = max_depth or (len(self.phase_order) - 1)

        while phase_idx < depth_limit:
            trans = self.transition_probs(path[-1], phase_idx)
            if not trans:
                break

            # Greedy selection
            next_ttp = max(trans.items(), key=lambda x: x[1])[0]
            path.append(next_ttp)
            phase_idx += 1

        reward = compute_reward_components(
            path=path,
            phase_order=self.phase_order,
            transition_table=self.transition_table,
            mitigation_stats=self.mitigation_stats,
            phase1_probs=phase1_probs,
            phase_embeddings=self.phase_embeddings,
        )

        return float(reward["R"])
