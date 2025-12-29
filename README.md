# Policy–Value Guided Kill-Chain Inference

This repository implements a bounded and explainable system for synthesizing plausible multi-stage cyber kill chains from unstructured CTI text. The system integrates transformer-based semantic priors, a symbolic Markov Decision Process (MDP), and a Policy–Value guided Monte Carlo Tree Search (MCTS).

The objective is hypothesis generation and structured reasoning, not attack prediction, exploitation, or attribution.

---

System Overview
---------------

The execution pipeline consists of six stages:

1. Context EncodingThe input CTI text is encoded using an AttackBERT-style encoder to obtain a dense semantic representation.
2. Transformer Phase PriorsPhase-specific transformer models infer top-k technique priors independently for each kill-chain phase.
3. Baseline Path Constructionansformer priors are combined to construct interpretable baseline kill-chain paths using probability mass and semantic similarity.
4. Symbolic MDP EvaluationBaseline paths are evaluated using a symbolic reward model that captures phase relevance, semantic cohesion, transition consistency, and defensive pressure.
5. Policy–Value Network TrainingA context-specific Policy–Value Network (PVNet) is trained using MDP rollouts as teacher supervision. Training is lightweight and performed at runtime for the given input context.
6. PV-guided MCTS Planning
   Monte Carlo Tree Search uses the trained PVNet to synthesize a final, full-depth 7-stage kill chain.

---

Repository Structure
--------------------

artifacts/
Precomputed embeddings, technique labels, and mitigation statistics.

core/
encoder.py – AttackBERT encoder wrapper
priors/ – Transformer-based phase inference
paths/ – Baseline transformer path construction
mdp.py – Symbolic MDP and reward computation
pvnet.py – Policy–Value Network and training utilities
mcts.py – Policy–Value guided Monte Carlo Tree Search
artifacts.py – Artifact loading utilities

models/
Trained transformer phase models.

sample_input.txt
Example CTI / incident report input.

main.py
End-to-end execution entry point.

requirements.txt
Python dependencies.
---

Installation
------------

1. Create a virtual environment

python -m venv venvsource venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

---

Running the System
------------------

1. Provide inputEdit sample_input.txt and paste a CTI-style report. For meaningful results, the input should describe attacker behavior across multiple phases and contain sufficient detail.
2. Execute

python main.py

OR

streamlit run app.py

---

Output Interpretation
---------------------

Transformer Baseline Paths
These paths are constructed purely from transformer priors and semantic similarity. They serve as an interpretable baseline for comparison.

Symbolic Evaluation (MDP)
Each baseline path is evaluated using the symbolic reward model. Reported metrics include relevance, coherence, transition consistency, and detection/mitigation pressure.

Policy–Value Network Training
A context-specific PVNet is trained using MDP rollouts. A dataset summary is printed, showing how many techniques, transitions, and training samples were generated.

Warnings from TensorFlow, CUDA, or sklearn versioning can be safely ignored.

PV-guided MCTS Planned Chain
The final output is a synthesized kill chain spanning all 7 phases. This chain represents the highest-value trajectory discovered by PV-guided MCTS.

---

Key Parameters and Tuning
-------------------------

k (Transformer priors)
  Number of top techniques selected per phase.

top_m (MCTS)
  Maximum number of candidate techniques expanded per phase.

sims (MCTS)
  Number of Monte Carlo simulations.

c_puct (MCTS)
  Exploration versus exploitation balance.

epochs (PVNet)
  Number of training epochs for the context-specific PVNet.

Increasing k, top_m, or sims increases exploration at the cost of runtime.

---

Scope and Limitations
---------------------

This system does not predict real attacks.
It does not attribute threat actors.
It does not exploit vulnerabilities.

All outputs are plausible, explainable hypotheses intended for analysis and research.

---

Intended Use
------------

Academic research
Explainable CTI analysis
Kill-chain reasoning experiments
Cognitive modeling of attacker behavior

---

License
-------

Research and educational use only.
