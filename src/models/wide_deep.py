"""Wide & Deep neural network for draft-level blue-side win probability.

Inputs:
    blue_ids: LongTensor [B, 5] — champion ids in role order (TOP, JG, MID, BOT, SUP)
    red_ids:  LongTensor [B, 5] — same shape, for the red team

Output:
    logits: FloatTensor [B] — pre-sigmoid score; apply torch.sigmoid for blue_win_prob.

Special tokens:
    id 0 -> "__PAD__" (unfilled slot)
    id 1 -> "__UNK__" (unknown champion at inference)
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

ROLE_ORDER: list[str] = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
PAD_TOKEN: str = "__PAD__"
UNK_TOKEN: str = "__UNK__"
PAD_ID: int = 0
UNK_ID: int = 1


class WideDeepDraftNet(nn.Module):
    """Wide & Deep architecture for predicting blue-side win probability."""

    def __init__(
        self,
        num_champions: int,
        embedding_dim: int = 32,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_champions = num_champions
        self.embedding_dim = embedding_dim

        # Shared embedding for the deep branch
        self.embedding = nn.Embedding(num_champions, embedding_dim, padding_idx=PAD_ID)

        # Wide branch: per-slot, per-side linear contribution per champion id.
        # Implemented as 10 slot-specific weight vectors over the full champion vocab.
        # Total params: 10 * num_champions (1 logit per (slot, champion)).
        self.wide_weights = nn.Embedding(num_champions, 10, padding_idx=PAD_ID)
        nn.init.zeros_(self.wide_weights.weight)

        # Deep branch: blue mean + red mean + per-slot blue + per-slot red embeddings
        deep_in = embedding_dim * 2 + embedding_dim * 10
        layers: list[nn.Module] = []
        prev = deep_in
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, blue_ids: torch.Tensor, red_ids: torch.Tensor) -> torch.Tensor:
        """Return [B] logits. Apply sigmoid externally for probability."""
        if blue_ids.shape[-1] != 5 or red_ids.shape[-1] != 5:
            raise ValueError(
                f"Expected 5 slots per side, got blue={blue_ids.shape}, red={red_ids.shape}"
            )

        b_emb = self.embedding(blue_ids)  # [B, 5, D]
        r_emb = self.embedding(red_ids)   # [B, 5, D]

        # Mean over non-pad slots (avoid dividing by zero)
        b_mask = (blue_ids != PAD_ID).float().unsqueeze(-1)  # [B, 5, 1]
        r_mask = (red_ids != PAD_ID).float().unsqueeze(-1)
        b_mean = (b_emb * b_mask).sum(dim=1) / b_mask.sum(dim=1).clamp(min=1.0)
        r_mean = (r_emb * r_mask).sum(dim=1) / r_mask.sum(dim=1).clamp(min=1.0)

        # Per-slot embeddings flattened (preserves slot identity for deep branch)
        b_flat = b_emb.reshape(b_emb.shape[0], -1)
        r_flat = r_emb.reshape(r_emb.shape[0], -1)

        deep_in = torch.cat([b_mean, r_mean, b_flat, r_flat], dim=1)
        deep_logit = self.deep(deep_in).squeeze(-1)  # [B]

        # Wide: each champion contributes a learned scalar per (side, slot).
        # Slots 0-4 are blue TOP..UTILITY; slots 5-9 are red TOP..UTILITY.
        # Mask out PAD ids so they contribute zero (padding_idx=0 already returns zeros).
        wide_b = self.wide_weights(blue_ids)  # [B, 5, 10]
        wide_r = self.wide_weights(red_ids)
        # Pick the slot-specific column for each position.
        slot_b = torch.arange(5, device=blue_ids.device).view(1, 5, 1)
        slot_r = torch.arange(5, 10, device=red_ids.device).view(1, 5, 1)
        wide_b = wide_b.gather(2, slot_b.expand(blue_ids.shape[0], 5, 1)).squeeze(-1)
        wide_r = wide_r.gather(2, slot_r.expand(red_ids.shape[0], 5, 1)).squeeze(-1)
        wide_logit = wide_b.sum(dim=1) + wide_r.sum(dim=1)

        return deep_logit + wide_logit
