"""
Bounded Monte Carlo Tree Search guided by a policy–value network.

This implementation supports optional search tracing for
visualization and analysis without altering search behavior.
"""

import math
import numpy as np
from typing import Dict, Optional
from uuid import uuid4


# --------------------------------------------------
# Tree Node
# --------------------------------------------------
class Node:
    def __init__(
        self,
        phase_idx: int,
        ttp_id: Optional[str],
        parent: Optional["Node"] = None,
        prior: float = 0.0,
    ):
        # Stable identifier for tracing and visualization
        self.id = uuid4().hex

        self.phase_idx = phase_idx
        self.ttp_id = ttp_id
        self.parent = parent
        self.children: Dict[str, Node] = {}

        self.N = 0  # visit count
        self.W = 0.0  # total value
        self.Q = 0.0  # mean value
        self.P = float(prior)  # prior probability


# --------------------------------------------------
# Monte Carlo Tree Search
# --------------------------------------------------
class MCTS:
    def __init__(
        self,
        pv_fn,
        phase_order,
        phase_embeddings,
        ctx_encoder,
        c_puct: float = 2.5,
        sims: int = 75,
        top_m: int = 10,
    ):
        """
        Args:
            pv_fn:
                Callable(ctx_embedding, candidate_embeddings)
                -> (policy_probs, value)
            phase_order:
                Ordered list of kill-chain phases
            phase_embeddings:
                Mapping {phase: {ttp_id: embedding}}
            ctx_encoder:
                Text-to-embedding encoder
            c_puct:
                Exploration constant
            sims:
                Number of MCTS simulations
            top_m:
                Max candidates expanded per phase
        """
        self.pv = pv_fn
        self.phase_order = phase_order
        self.phase_embeddings = phase_embeddings
        self.ctx_encoder = ctx_encoder

        self.c_puct = c_puct
        self.sims = sims
        self.top_m = top_m

        self.ctx_emb = None

        # Trace registry: node_id -> node snapshot
        self._trace_map: Dict[str, Dict] = {}

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def run(self, root: Node, text: str):
        """
        Execute MCTS simulations from the root node.
        """
        self.ctx_emb = self.ctx_encoder.encode(text)
        self._trace_map = {}

        # Record root explicitly
        self._record(root)

        for _ in range(self.sims):
            node = root
            path = [node]

            # -------------------------
            # Selection
            # -------------------------
            while node.children:
                node = max(
                    node.children.values(),
                    # key=lambda c: c.Q
                    # + self.c_puct * c.P * math.sqrt(node.N + 1) / (1 + c.N),
                    key=lambda c: c.Q
                    + (self.c_puct * (1.0 + 0.15 * c.phase_idx))
                    * c.P
                    * math.sqrt(node.N + 1)
                    / (1 + c.N),
                )
                path.append(node)

            # -------------------------
            # Expansion + Rollout
            # -------------------------
            if node.phase_idx < len(self.phase_order) - 1:
                self._expand(node)
                value = self._rollout_to_end(node)
                value *= 1.0 + 0.1 * node.phase_idx
            else:
                value = node.Q

            # -------------------------
            # Backup
            # -------------------------
            for n in reversed(path):
                n.N += 1
                n.W += value
                n.Q = n.W / n.N
                self._record(n)

    # --------------------------------------------------
    # Expansion (PURE MCTS)
    # --------------------------------------------------
    def _expand(self, node: Node):
        """
        Expand the node by adding policy-guided children
        for the next phase only.
        """
        next_phase = self.phase_order[node.phase_idx + 1]
        ttp_map = self.phase_embeddings.get(next_phase, {})

        if not ttp_map:
            return

        ttp_ids = list(ttp_map.keys())
        cand_embs = np.stack([ttp_map[t] for t in ttp_ids], axis=0)

        # probs, _ = self.pv(self.ctx_emb, cand_embs)
        # probs = np.asarray(probs, dtype=float)

        # if probs.sum() <= 0:
        #     probs = np.ones_like(probs) / len(probs)
        # else:
        #     probs /= probs.sum()

        probs, _ = self.pv(self.ctx_emb, cand_embs)
        probs = np.asarray(probs, dtype=float)

        # ---- PATCH: prevent late-phase starvation ----
        uniform = np.ones_like(probs) / len(probs)

        # Blend PV prior with uniform prior
        alpha = 0.6 if node.phase_idx >= 4 else 0.85
        probs = alpha * probs + (1.0 - alpha) * uniform

        probs /= probs.sum()

        # order = np.argsort(probs)[::-1][: self.top_m]
        width = self.top_m + max(0, node.phase_idx - 3) * 2
        order = np.argsort(probs)[::-1][: min(width, len(probs))]

        # ---- PATCH: force minimal expansion if starving ----
        if len(order) == 0 and len(probs) > 0:
            order = [int(np.argmax(probs))]

        for idx in order:
            ttp = ttp_ids[idx]
            if ttp in node.children:
                continue

            child = Node(
                phase_idx=node.phase_idx + 1,
                ttp_id=ttp,
                parent=node,
                prior=float(probs[idx]),
            )
            node.children[ttp] = child
            self._record(child)

    # --------------------------------------------------
    # Rollout (NOT part of tree)
    # --------------------------------------------------
    def _rollout_to_end(self, start_node: Node) -> float:
        """
        Greedy rollout to the final phase using the policy network.
        Rollout nodes are NOT part of the search tree.
        """
        node = start_node
        value = 0.0

        while node.phase_idx < len(self.phase_order) - 1:
            next_phase = self.phase_order[node.phase_idx + 1]
            ttp_map = self.phase_embeddings.get(next_phase, {})

            if not ttp_map:
                break

            ttp_ids = list(ttp_map.keys())
            cand_embs = np.stack([ttp_map[t] for t in ttp_ids], axis=0)

            probs, value = self.pv(self.ctx_emb, cand_embs)
            best = int(np.argmax(probs))

            node = Node(
                phase_idx=node.phase_idx + 1,
                ttp_id=ttp_ids[best],
                parent=None,  # rollout nodes are detached
                prior=float(probs[best]),
            )
        if node.phase_idx == len(self.phase_order) - 1:
            value += 0.5

        return float(value)
        # return float(value)

    # --------------------------------------------------
    # Trace Recording (DEDUPLICATED)
    # --------------------------------------------------
    def _record(self, node: Node):
        """
        Record or update node state for visualization.
        Ensures exactly one entry per node.
        """
        self._trace_map[node.id] = {
            "id": node.id,
            "parent": node.parent.id if node.parent else None,
            "phase_idx": node.phase_idx,
            "ttp_id": node.ttp_id,
            "N": node.N,
            "Q": node.Q,
            "P": node.P,
        }

    # --------------------------------------------------
    # Trace Accessor
    # --------------------------------------------------
    @property
    def trace(self):
        """
        Return trace as a list for downstream compatibility.
        """
        return list(self._trace_map.values())
