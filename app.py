"""
Streamlit frontend for symbolic kill-chain inference.

This application provides:
1. CTI text input
2. Transformer-based phase priors
3. Baseline path construction
4. Symbolic MDP evaluation
5. Context-specific PVNet training
6. PV-guided MCTS planning
7. MCTS tree visualization
"""

from pathlib import Path
import warnings
import random
import numpy as np
import torch
import streamlit as st

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

from ui.tree_viz import (
    build_mcts_graph,
    summarize_graph,
    show_interactive_tree,
)


# --------------------------------------------------
# Environment & Reproducibility
# --------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Constants
# --------------------------------------------------
PHASE_KEYS = [
    "recon",
    "weapon",
    "delivery",
    "exploit",
    "install",
    "c2",
    "objectives",
]

PHASE_ORDER = [
    "Reconnaissance",
    "Weaponization",
    "Delivery",
    "Exploitation",
    "Installation",
    "Command_and_Control",
    "Actions_on_Objectives",
]


# --------------------------------------------------
# Transformer Phase Priors
# --------------------------------------------------
def infer_phase_priors(text: str, model_root: Path, k: int):
    encoder = AttackBERTEncoder()
    text_emb = encoder.encode(text)

    priors = {}
    for phase in PHASE_KEYS:
        priors[phase] = predict_topk_for_phase(
            phase_name=phase,
            text_embedding=text_emb,
            model_root=model_root,
            device=device,
            k=k,
        )
    return priors


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Kill-Chain Reasoning", layout="wide")
st.title("Cyber Kill-Chain Inference")

BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "artifacts"

encoder = AttackBERTEncoder()

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("Configuration")

top_k = st.sidebar.slider("Transformer Top-K per Phase", 1, 10, 2)
mcts_sims = st.sidebar.slider("MCTS Simulations", 25, 300, 100, step=25)
mcts_top_m = st.sidebar.slider("MCTS Expansion Width (top-M)", 1, 20, 2)
train_epochs = st.sidebar.slider("PVNet Training Epochs", 1, 20, 3)

# --------------------------------------------------
# Input
# --------------------------------------------------

default_text = (BASE_DIR / "sample_input.txt").read_text(encoding="utf-8")
text = st.text_area(
    "Paste CTI / Incident Report",
    value=default_text,
    height=260,
)

run_button = st.button("Run Analysis")

st.markdown(
    """
    <style>
    /* Text area border (orange) */
    textarea {
        border: 2px solid #ff7f0e !important;
        border-radius: 6px;
    }

    /* Run button default */
    div.stButton > button {
        background-color: #262730;
        color: white;
        border-radius: 6px;
        border: 1px solid #555;
        transition: all 0.2s ease-in-out;
    }

    /* Run button hover (green) */
    div.stButton > button:hover {
        background-color: #2ca02c;
        color: white;
        border-color: #2ca02c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------
if run_button and text.strip():

    progress = st.progress(0)
    status = st.empty()

    # --------------------------------------------------
    # 1. Transformer Priors
    # --------------------------------------------------
    status.text("Inferring transformer-based phase priors...")
    priors = infer_phase_priors(text, MODEL_ROOT, k=top_k)
    progress.progress(10)

    # --------------------------------------------------
    # 2. Baseline Transformer Paths
    # --------------------------------------------------
    status.text("Building baseline transformer paths...")
    baseline_paths = build_paths(priors, encoder, max_paths=3)
    progress.progress(20)

    # status.text("Building baseline transformer paths...")
    # path_progress = st.progress(0)

    # # Simulated progress while build_paths runs
    # baseline_paths = None
    # for i in range(0, 80, 5):
    #     path_progress.progress(i)

    # baseline_paths = build_paths(priors, encoder, max_paths=3)
    # path_progress.progress(100)

    st.subheader("Transformer Baseline Paths")
    for i, (score, path) in enumerate(baseline_paths, 1):
        ttp_seq = [t for _, t, _ in path]
        st.markdown(f"**Path {i}**")
        st.code(" -> ".join(ttp_seq))
        st.caption(f"Heuristic Score: {score:.4f}")

    # --------------------------------------------------
    # 3. Load Artifacts & MDP
    # --------------------------------------------------
    status.text("Loading symbolic artifacts...")
    store = ArtifactStore(ARTIFACT_DIR)
    store.load()

    transition_table = build_transition_table(
        store.phase_embeddings,
        PHASE_ORDER,
    )

    mdp = KillChainMDP(
        phase_order=PHASE_ORDER,
        phase_embeddings=store.phase_embeddings,
        transition_table=transition_table,
        mitigation_stats=store.mitigation_stats,
    )
    progress.progress(35)

    # --------------------------------------------------
    # 4. Symbolic Evaluation
    # --------------------------------------------------
    status.text("Evaluating baseline paths symbolically...")
    name_to_ttp = {v: k for k, v in store.ttp_labels.items()}

    phase_prob_map = {
        phase.capitalize(): {name_to_ttp[t]: p for t, p in plist if t in name_to_ttp}
        for phase, plist in priors.items()
    }

    st.subheader("Symbolic MDP Evaluation")
    for i, (_, path) in enumerate(baseline_paths, 1):
        ttp_path = [name_to_ttp[t] for _, t, _ in path if t in name_to_ttp]
        comps = mdp.evaluate_path_verbose(ttp_path, phase_prob_map)

        st.markdown(f"**Path {i}**")
        st.write(
            f"Reward: `{comps['R']:.4f}` | "
            f"Rel: `{comps['Rel']:.3f}` | "
            f"Coh: `{comps['Coh']:.3f}` | "
            f"Trans: `{comps['Trans']:.3f}`"
        )
        st.write(
            f"Detection: `{comps['Detection_Count']}` | "
            f"Mitigation: `{comps['Mitigation_Count']}`"
        )

    progress.progress(50)

    # --------------------------------------------------
    # 5. PVNet Training
    # --------------------------------------------------
    status.text("Training context-specific Policy–Value Network...")
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
        phase_order=PHASE_ORDER,
        phase_priors=phase_prob_map,
        encoder=encoder,
    )

    pv_model = PolicyValueNet().to(device)

    train_pvnet(
        model=pv_model,
        dataset=pv_dataset,
        epochs=train_epochs,
        batch_size=16,
        device=device,
        save_path=BASE_DIR / "pvnet.pt",
    )

    pv_model.eval()
    progress.progress(70)

    # --------------------------------------------------
    # 6. MCTS Planning
    # --------------------------------------------------
    status.text("Running PV-guided Monte Carlo Tree Search...")

    def pv_wrapper(ctx_emb, cand_embs):
        ctx = torch.tensor(ctx_emb).unsqueeze(0).float().to(device)
        cand = torch.tensor(cand_embs).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits, value = pv_model(ctx, cand)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs, float(value.item())

    mcts = MCTS(
        pv_fn=pv_wrapper,
        phase_order=PHASE_ORDER,
        phase_embeddings=store.phase_embeddings,
        ctx_encoder=encoder,
        sims=mcts_sims,
        top_m=mcts_top_m,
    )

    root = Node(phase_idx=-1, ttp_id=None, prior=1.0)
    mcts.run(root, text)

    progress.progress(85)

    # --------------------------------------------------
    # 7. Final Kill-Chain (Pure Streamlit UI)
    # --------------------------------------------------

    st.subheader("Final Planned Kill-Chain")

    # Build final chain as a phase -> (ttp, name) mapping
    final_chain = {}
    node = root

    while node.phase_idx < len(PHASE_ORDER) - 1:
        if node.children:
            node = max(node.children.values(), key=lambda c: c.N)
        else:
            next_phase = PHASE_ORDER[node.phase_idx + 1]
            ttp_map = store.phase_embeddings.get(next_phase, {})

            if not ttp_map:
                break

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

        phase = PHASE_ORDER[node.phase_idx]
        ttp = node.ttp_id
        name = store.ttp_labels.get(ttp, "Unknown")
        final_chain[phase] = (ttp, name)

    # ---- Grid rendering (1 row × N phases) ----
    cols = st.columns(len(PHASE_ORDER))

    for col, phase in zip(cols, PHASE_ORDER):
        with col:
            st.markdown(f"**{phase}**")
            if phase in final_chain:
                ttp, name = final_chain[phase]
                st.write(f"`{ttp}`")
                st.caption(name)
            else:
                st.write("—")
                st.caption("No technique selected")

    # --------------------------------------------------
    # 8. MCTS Tree Visualization
    # --------------------------------------------------
    st.subheader("MCTS Search Tree")

    graph = build_mcts_graph(
        trace=mcts.trace,
        phase_order=PHASE_ORDER,
        ttp_label_map=store.ttp_labels,
    )

    stats = summarize_graph(graph)
    st.caption(
        f"Nodes: {stats['nodes']} | "
        f"Edges: {stats['edges']} | "
        f"Max Depth: {stats['max_depth']}"
    )

    show_interactive_tree(graph)

    progress.progress(100)
    status.text("Analysis complete.")
