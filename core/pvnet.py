"""
Policy–Value Network and training utilities for symbolic kill-chain reasoning.

This module provides:
1. Dataset construction using symbolic MDP rollouts
2. Policy–Value Network architecture
3. Supervised training loop

The network is trained offline and remains frozen during inference.
"""

from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ==================================================
# Pseudo-report Construction
# ==================================================
def collect_pseudo_reports_from_priors(
    phase_priors: Dict[str, Dict[str, float]],
    ttp_descriptions: Dict[str, str],
    min_prior: float = 0.05,
) -> List[str]:
    """
    Construct pseudo-reports from technique descriptions
    selected by transformer phase priors.
    """
    reports = []
    for _, priors in phase_priors.items():
        for ttp, p in priors.items():
            if p >= min_prior and ttp in ttp_descriptions:
                reports.append(ttp_descriptions[ttp])
    return list(dict.fromkeys(reports))


# ==================================================
# Dataset Construction
# ==================================================
def build_pv_dataset(
    texts: List[str],
    mdp,
    phase_order: List[str],
    phase_priors: Dict[str, Dict[str, float]],
    encoder,
    max_candidates: int = 8,
    min_prior: float = 0.05,
):
    """
    Build supervised training samples for the Policy–Value Network
    using symbolic MDP rollouts.
    """
    dataset = []

    total_prior_ttps = 0
    total_mdp_edges = 0
    candidate_counts = []

    for text in tqdm(texts, desc="Building PV dataset"):
        ctx_emb = encoder.encode(text).astype(np.float32)

        for phase_idx, phase in enumerate(phase_order[:-1]):
            priors = phase_priors.get(phase, {})
            if not priors:
                continue

            total_prior_ttps += len(priors)

            for ttp, p_prior in priors.items():
                if p_prior < min_prior:
                    continue

                transitions = mdp.transition_probs(ttp, phase_idx)
                if not transitions:
                    continue

                total_mdp_edges += len(transitions)

                cand_ttps = sorted(
                    transitions.items(), key=lambda x: x[1], reverse=True
                )[:max_candidates]

                candidate_counts.append(len(cand_ttps))

                next_phase = phase_order[phase_idx + 1]
                emb_map = mdp.phase_embeddings.get(next_phase, {})

                cand_embs = []
                rewards = []

                for ttp_next, _ in cand_ttps:
                    emb = emb_map.get(ttp_next)
                    if emb is None:
                        continue

                    cand_embs.append(emb.astype(np.float32))

                    r = mdp.rollout(
                        start_ttp=ttp_next,
                        start_phase_idx=phase_idx + 1,
                        phase1_probs=phase_priors,
                    )
                    rewards.append(r)

                if not cand_embs:
                    continue

                rewards = np.asarray(rewards, dtype=np.float32)
                policy_target = np.exp(rewards / max(1e-6, rewards.std() + 1e-6))
                policy_target /= policy_target.sum()

                value_target = float(
                    mdp.rollout(
                        start_ttp=ttp,
                        start_phase_idx=phase_idx,
                        phase1_probs=phase_priors,
                    )
                )

                dataset.append(
                    {
                        "ctx_emb": ctx_emb,
                        "cand_embs": np.stack(cand_embs),
                        "policy_target": policy_target,
                        "value": value_target,
                    }
                )

    avg_cands = (
        sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
    )

    print("\n[PV Dataset Summary]")
    print(f"  Transformer prior techniques seen: {total_prior_ttps}")
    print(f"  MDP transition edges evaluated:    {total_mdp_edges}")
    print(f"  PV training samples generated:     {len(dataset)}")
    print(f"  Avg candidates per sample:         {avg_cands:.2f}")

    return dataset


# ==================================================
# Torch Dataset Wrapper
# ==================================================
class PVDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return (
            torch.tensor(x["ctx_emb"], dtype=torch.float32),
            torch.tensor(x["cand_embs"], dtype=torch.float32),
            torch.tensor(x["policy_target"], dtype=torch.float32),
            torch.tensor(x["value"], dtype=torch.float32),
        )


# ==================================================
# Policy–Value Network Architecture
# ==================================================
class SetAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn, _ = self.attn(
            x, x, x, key_padding_mask=~mask if mask is not None else None
        )
        x = self.norm1(x + self.dropout(attn))
        ff = self.ff(x)
        return self.norm2(x + self.dropout(ff))


class PolicyValueNet(nn.Module):
    """
    Policy–Value Network for symbolic kill-chain inference.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        proj_dim: int = 256,
        hidden: int = 256,
        heads: int = 4,
        set_blocks: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ctx_proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )

        self.cand_proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )

        self.cross_attn = nn.MultiheadAttention(proj_dim, heads, batch_first=True)

        self.set_blocks = nn.ModuleList(
            [SetAttentionBlock(proj_dim, heads, dropout) for _ in range(set_blocks)]
        )

        self.film = nn.Linear(proj_dim, proj_dim * 2)

        self.policy_head = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(proj_dim * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ctx_emb, cand_embs, mask: Optional[torch.Tensor] = None):
        B, N, _ = cand_embs.shape

        ctx = self.ctx_proj(ctx_emb)
        cand = self.cand_proj(cand_embs.view(B * N, -1)).view(B, N, -1)

        ctx_seq = ctx.unsqueeze(1)
        attn, _ = self.cross_attn(
            ctx_seq, cand, cand, key_padding_mask=~mask if mask is not None else None
        )
        ctx_enh = (ctx_seq + attn).squeeze(1)

        for blk in self.set_blocks:
            cand = blk(cand, mask)

        gamma, beta = self.film(ctx_enh).chunk(2, dim=-1)
        cand = cand * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        pol_in = torch.cat([cand, ctx_enh.unsqueeze(1).expand(-1, N, -1)], dim=-1)
        logits = self.policy_head(pol_in.view(B * N, -1)).view(B, N)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        mean_pool = cand.mean(1)
        max_pool, _ = cand.max(1)

        value = self.value_head(
            torch.cat([ctx_enh, mean_pool, max_pool], dim=-1)
        ).squeeze(-1)

        return logits, value


# ==================================================
# Training Loop
# ==================================================
def train_pvnet(
    model: PolicyValueNet,
    dataset,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cpu",
    save_path: str = "pvnet.pt",
):
    """
    Train the Policy–Value Network using supervised MDP targets.
    """
    ds = PVDataset(dataset)

    if len(ds) < 2:
        train_ds = ds
        val_ds = None
    else:
        n_val = max(1, int(0.1 * len(ds)))
        train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size) if val_ds is not None else None

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    kl = nn.KLDivLoss(reduction="batchmean")
    mse = nn.MSELoss()

    best_val = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        for ctx, cand, p_t, v_t in train_loader:
            ctx, cand, p_t, v_t = (
                ctx.to(device),
                cand.to(device),
                p_t.to(device),
                v_t.to(device),
            )

            opt.zero_grad()
            logits, val = model(ctx, cand)
            loss = kl(torch.log_softmax(logits, dim=1), p_t) + mse(val, v_t)
            loss.backward()
            opt.step()

        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for ctx, cand, p_t, v_t in val_loader:
                    ctx, cand, p_t, v_t = (
                        ctx.to(device),
                        cand.to(device),
                        p_t.to(device),
                        v_t.to(device),
                    )
                    logits, val = model(ctx, cand)
                    val_loss += float(
                        kl(torch.log_softmax(logits, dim=1), p_t) + mse(val, v_t)
                    )

        print(f"[PVNet] Epoch {ep}/{epochs} | val_loss={val_loss:.4f}")

        if val_loader is None or val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
