"""
End-to-end execution pipeline.

Flow:
1. Encode CTI input
2. Infer per-phase technique priors (Transformer)
3. Build baseline transformer paths (interpretability only)
4. Evaluate baseline paths using symbolic MDP
5. Train context-specific Policy–Value Network (MDP teacher)
6. Infer final kill-chain using Policy–Value guided MCTS
"""

from pathlib import Path
import os
import random
import warnings
import numpy as np
import torch

from core.encoder import AttackBERTEncoder
from core.priors.inference import predict_topk_for_phase
from core.paths.builder import build_paths
from core.artifacts import ArtifactStore
from core.mdp import KillChainMDP, build_transition_table
from core.pvnet import (
    PolicyValueNet,
    build_pv_dataset,
    train_pvnet,
    collect_pseudo_reports_from_priors,
)
from core.mcts import MCTS, Node


# --------------------------------------------------
# Environment & Reproducibility
# --------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Transformer Phase Priors
# --------------------------------------------------
def infer_phase_priors(text: str, model_root: Path, k: int = 3):
    """
    Infer phase-conditioned technique priors from CTI text.
    """
    encoder = AttackBERTEncoder()
    text_emb = encoder.encode(text)

    phase_keys = [
        "recon",
        "weapon",
        "delivery",
        "exploit",
        "install",
        "c2",
        "objectives",
    ]

    priors = {}
    for phase in phase_keys:
        priors[phase] = predict_topk_for_phase(
            phase_name=phase,
            text_embedding=text_emb,
            model_root=model_root,
            device=device,
            k=k,
        )

    return priors


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    # --------------------------------------------------
    # Input
    # --------------------------------------------------
    input_path = BASE_DIR / "sample_input.txt"
    text = input_path.read_text(encoding="utf-8").strip()

    model_root = BASE_DIR / "models"
    encoder = AttackBERTEncoder()

    # --------------------------------------------------
    # Transformer Priors
    # --------------------------------------------------
    priors = infer_phase_priors(text, model_root, k=3)

    # --------------------------------------------------
    # Baseline Transformer Paths (Interpretability)
    # --------------------------------------------------
    baseline_paths = build_paths(priors, encoder, max_paths=3)

    print("\n=== Transformer Baseline Paths ===")
    for i, (score, path) in enumerate(baseline_paths, 1):
        ttp_seq = [t for _, t, _ in path]
        print(f"\nPath {i}:")
        print("  TTPs:", " -> ".join(ttp_seq))
        print(f"  Heuristic Score: {score:.4f}")

    # --------------------------------------------------
    # Load Artifacts
    # --------------------------------------------------
    store = ArtifactStore(BASE_DIR / "artifacts")
    store.load()

    phase_order = [
        "Reconnaissance",
        "Weaponization",
        "Delivery",
        "Exploitation",
        "Installation",
        "Command_and_Control",
        "Actions_on_Objectives",
    ]

    transition_table = build_transition_table(
        store.phase_embeddings,
        phase_order,
    )

    mdp = KillChainMDP(
        phase_order=phase_order,
        phase_embeddings=store.phase_embeddings,
        transition_table=transition_table,
        mitigation_stats=store.mitigation_stats,
    )

    # --------------------------------------------------
    # Symbolic Evaluation of Baselines
    # --------------------------------------------------
    print("\n=== Symbolic Evaluation (MDP) ===")

    name_to_ttp = {v: k for k, v in store.ttp_labels.items()}

    phase_prob_map = {
        phase.capitalize(): {name_to_ttp[t]: p for t, p in plist if t in name_to_ttp}
        for phase, plist in priors.items()
    }

    for i, (_, path) in enumerate(baseline_paths, 1):
        ttp_path = [name_to_ttp[t] for _, t, _ in path if t in name_to_ttp]
        comps = mdp.evaluate_path_verbose(ttp_path, phase_prob_map)

        print(f"\nPath {i}:")
        print("  Final Reward (R):", f"{comps['R']:.4f}")
        print(
            "  Rel / Coh / Trans:",
            f"{comps['Rel']:.3f}",
            f"{comps['Coh']:.3f}",
            f"{comps['Trans']:.3f}",
        )
        print(
            "  Detection / Mitigation:",
            comps["Detection_Count"],
            comps["Mitigation_Count"],
        )

    # --------------------------------------------------
    # Train Context-Specific Policy–Value Network
    # --------------------------------------------------
    print("\n=== Training Policy–Value Network (MDP Teacher) ===")

    ttp_descriptions = {ttp_id: name for ttp_id, name in store.ttp_labels.items()}

    pseudo_reports = collect_pseudo_reports_from_priors(
        phase_priors=phase_prob_map,
        ttp_descriptions=ttp_descriptions,
        min_prior=0.05,
    )

    if not pseudo_reports:
        pseudo_reports = [text]

    pv_dataset = build_pv_dataset(
        texts=pseudo_reports,
        mdp=mdp,
        phase_order=phase_order,
        phase_priors=phase_prob_map,
        encoder=encoder,
    )

    pv_model = PolicyValueNet().to(device)

    train_pvnet(
        model=pv_model,
        dataset=pv_dataset,
        epochs=5,
        batch_size=16,
        device=device,
        save_path=BASE_DIR / "pvnet.pt",
    )

    pv_model.eval()

    # --------------------------------------------------
    # PV-guided MCTS Planning
    # --------------------------------------------------
    print("\n=== PV-guided MCTS Planned Chain ===")

    def pv_wrapper(ctx_emb, cand_embs):
        if isinstance(ctx_emb, np.ndarray):
            ctx_emb = torch.from_numpy(ctx_emb)
        if isinstance(cand_embs, np.ndarray):
            cand_embs = torch.from_numpy(cand_embs)

        ctx = ctx_emb.unsqueeze(0).float().to(device)
        cand = cand_embs.unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits, value = pv_model(ctx, cand)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        return probs, float(value.item())

    mcts = MCTS(
        pv_fn=pv_wrapper,
        phase_order=phase_order,
        phase_embeddings=store.phase_embeddings,
        ctx_encoder=encoder,
        sims=100,
        top_m=8,
    )

    root = Node(phase_idx=-1, ttp_id=None, prior=1.0)
    mcts.run(root, text)

    # --------------------------------------------------
    # Kill-Chain Extraction (Guaranteed 7 Phases)
    # --------------------------------------------------
    planned_chain = []
    node = root

    while node.phase_idx < len(phase_order) - 1:
        if node.children:
            node = max(node.children.values(), key=lambda c: c.N)
        else:
            next_phase = phase_order[node.phase_idx + 1]
            ttp_map = store.phase_embeddings.get(next_phase, {})

            ttp_ids = list(ttp_map.keys())
            cand_embs = np.stack([ttp_map[t] for t in ttp_ids], axis=0)

            probs, _ = pv_wrapper(mcts.ctx_emb, cand_embs)
            best = int(np.argmax(probs))

            node = Node(
                phase_idx=node.phase_idx + 1,
                ttp_id=ttp_ids[best],
                parent=node,
                prior=float(probs[best]),
            )

        phase = phase_order[node.phase_idx]
        ttp = node.ttp_id
        name = store.ttp_labels.get(ttp, "Unknown")
        planned_chain.append(f"{phase}:{ttp} ({name})")

    print("\nPlanned Kill-Chain:")
    print("  " + " -> ".join(planned_chain))
