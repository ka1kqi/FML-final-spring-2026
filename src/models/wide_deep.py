"""Wide & Deep neural network for draft-level blue-side win probability.

Architecture switch (selected via config["architecture"]):
    - "legacy_flat_mlp": original mean+per-slot-flat MLP (back-compat for old artifacts)
    - "v2_pairwise":     role-aware pooled embeddings + intra/cross-team pairwise
                         dot-products (TeamCompNet-inspired). The default for newly
                         trained models; substantially better calibration & ranking.

Inputs (both modes):
    blue_ids: LongTensor [B, 5] — champion ids in role order (TOP, JG, MID, BOT, SUP)
    red_ids:  LongTensor [B, 5]

Output:
    logits: FloatTensor [B] — pre-sigmoid score; apply torch.sigmoid for blue_win_prob.

Special tokens:
    id 0 -> "__PAD__" (unfilled slot, never contributes)
    id 1 -> "__UNK__" (unknown champion at inference)

Wide branch (shared across architectures):
    nn.Embedding(num_champions, 10, padding_idx=PAD_ID); slots 0-4 are blue
    TOP..UTILITY, slots 5-9 are red TOP..UTILITY. Equivalent to a role-slot
    one-hot LR but ~100x faster via embedding gather.
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
N_ROLES: int = 5

LEGACY_ARCH = "legacy_flat_mlp"
V2_ARCH = "v2_pairwise"


# ---------------------------------------------------------------------------
# Shared wide branch (unchanged from v1).
# ---------------------------------------------------------------------------
class _WideBranch(nn.Module):
    """Per-(side, slot) scalar weights per champion id.

    PAD id (0) contributes zero: its weight row is initialized to zero (via
    nn.init.zeros_) and its gradient is masked by ``padding_idx``, so it stays
    zero through training even after many SGD steps.
    """

    def __init__(self, num_champions: int) -> None:
        super().__init__()
        self.weights = nn.Embedding(num_champions, 10, padding_idx=PAD_ID)
        nn.init.zeros_(self.weights.weight)  # neutral start; let training fill in biases

    def forward(self, blue_ids: torch.Tensor, red_ids: torch.Tensor) -> torch.Tensor:
        wide_b = self.weights(blue_ids)  # [B, 5, 10]
        wide_r = self.weights(red_ids)
        slot_b = torch.arange(5, device=blue_ids.device).view(1, 5, 1)
        slot_r = torch.arange(5, 10, device=red_ids.device).view(1, 5, 1)
        wide_b = wide_b.gather(2, slot_b.expand(blue_ids.shape[0], 5, 1)).squeeze(-1)
        wide_r = wide_r.gather(2, slot_r.expand(red_ids.shape[0], 5, 1)).squeeze(-1)
        return wide_b.sum(dim=1) + wide_r.sum(dim=1)


# ---------------------------------------------------------------------------
# Legacy deep branch: blue_mean + red_mean + per-slot embeddings flattened.
# Kept for back-compat with existing wide_deep.pt artifacts.
# ---------------------------------------------------------------------------
class _LegacyFlatDeepBranch(nn.Module):
    def __init__(
        self,
        num_champions: int,
        embedding_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_champions, embedding_dim, padding_idx=PAD_ID)
        deep_in = embedding_dim * 2 + embedding_dim * 10
        layers: list[nn.Module] = []
        prev = deep_in
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, blue_ids: torch.Tensor, red_ids: torch.Tensor) -> torch.Tensor:
        b_emb = self.embedding(blue_ids)
        r_emb = self.embedding(red_ids)
        b_mask = (blue_ids != PAD_ID).float().unsqueeze(-1)
        r_mask = (red_ids != PAD_ID).float().unsqueeze(-1)
        b_mean = (b_emb * b_mask).sum(dim=1) / b_mask.sum(dim=1).clamp(min=1.0)
        r_mean = (r_emb * r_mask).sum(dim=1) / r_mask.sum(dim=1).clamp(min=1.0)
        b_flat = b_emb.reshape(b_emb.shape[0], -1)
        r_flat = r_emb.reshape(r_emb.shape[0], -1)
        return self.mlp(torch.cat([b_mean, r_mean, b_flat, r_flat], dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# v2 pairwise deep branch: role-aware embeddings + intra/cross-team pairwise.
# Inspired by feature/draft-pipeline-v2's TeamCompNet, distilled to one file.
# ---------------------------------------------------------------------------
class _PairwiseDeepBranch(nn.Module):
    """Champion+role embeddings → pooled means + pooled diff/product +
    intra-team pairwise dot products + cross-team matchup dot product → MLP.

    All pooling and pairwise statistics are PAD-masked: empty slots contribute
    nothing to means or pairwise sums and never inflate the divisor.
    """

    def __init__(
        self,
        num_champions: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.champion_emb = nn.Embedding(num_champions, embedding_dim, padding_idx=PAD_ID)
        # role_emb: 5 learned vectors for TOP..UTILITY. No padding row needed
        # because role indices are always 0-4 (slot positions are real).
        self.role_emb = nn.Embedding(N_ROLES, embedding_dim)

        # Feature vector layout [pooled blue, pooled red, diff, product,
        #                        blue intra-syn (1), red intra-syn (1), cross matchup (1)]
        feat_dim = embedding_dim * 4 + 3
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _team_features(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (role-aware embeddings [B,5,D], pooled [B,D], intra dot [B,1])."""
        emb = self.champion_emb(ids)  # [B, 5, D]
        roles = torch.arange(N_ROLES, device=ids.device).unsqueeze(0).expand(ids.shape[0], N_ROLES)
        emb = emb + self.role_emb(roles)  # add role bias
        mask = (ids != PAD_ID).float().unsqueeze(-1)  # [B, 5, 1]
        emb = emb * mask  # zero out PAD rows
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = emb.sum(dim=1) / denom  # [B, D]

        # Intra-team pairwise dot product, mean over off-diagonal pairs.
        sim = torch.bmm(emb, emb.transpose(1, 2))  # [B, 5, 5]
        flat_mask = mask.squeeze(-1)  # [B, 5]
        pair_mask = flat_mask.unsqueeze(2) * flat_mask.unsqueeze(1)  # [B, 5, 5]
        diag = torch.eye(5, device=ids.device).unsqueeze(0)
        pair_mask = pair_mask * (1.0 - diag)
        pair_count = pair_mask.sum(dim=(1, 2)).clamp(min=1.0)
        intra = (sim * pair_mask).sum(dim=(1, 2)) / pair_count  # [B]
        return emb, pooled, intra.unsqueeze(-1)

    def forward(self, blue_ids: torch.Tensor, red_ids: torch.Tensor) -> torch.Tensor:
        b_emb, b_pool, b_intra = self._team_features(blue_ids)
        r_emb, r_pool, r_intra = self._team_features(red_ids)

        diff = b_pool - r_pool
        prod = b_pool * r_pool

        # Cross-team matchup: mean dot product between every (blue, red) pair,
        # masking pairs that include a PAD slot.
        b_mask = (blue_ids != PAD_ID).float()  # [B, 5]
        r_mask = (red_ids != PAD_ID).float()
        cross_sim = torch.bmm(b_emb, r_emb.transpose(1, 2))  # [B, 5, 5]
        cross_mask = b_mask.unsqueeze(2) * r_mask.unsqueeze(1)
        cross_count = cross_mask.sum(dim=(1, 2)).clamp(min=1.0)
        cross = (cross_sim * cross_mask).sum(dim=(1, 2)) / cross_count
        cross = cross.unsqueeze(-1)

        feats = torch.cat([b_pool, r_pool, diff, prod, b_intra, r_intra, cross], dim=1)
        return self.mlp(feats).squeeze(-1)


# ---------------------------------------------------------------------------
# Public model: dispatches to the right deep branch based on config.
# ---------------------------------------------------------------------------
class WideDeepDraftNet(nn.Module):
    """Wide & Deep architecture for predicting blue-side win probability.

    The model is intentionally asymmetric — blue and red embeddings are treated
    differently via separate slot columns in the wide branch. This reflects the
    real blue-side advantage in League of Legends; do not "fix" by adding
    symmetry constraints.

    Args:
        num_champions: vocab size including PAD (0) and UNK (1).
        embedding_dim: champion embedding dim. Default 32.
        hidden_dims: legacy MLP layer widths (used when architecture=="legacy_flat_mlp").
        hidden_dim: pairwise MLP hidden width (used when architecture=="v2_pairwise").
        dropout: dropout in deep branch MLP.
        architecture: "legacy_flat_mlp" | "v2_pairwise". Default "v2_pairwise".
        combine: "sum" | "concat" — how to merge wide+deep logits. Default "sum".
            "concat" adds a 1-layer Linear(2,1) head on top, so the output stays a
            single logit; the small extra head's weights start at 0.5 each so a
            freshly initialised concat model behaves identically to a sum model.
    """

    def __init__(
        self,
        num_champions: int,
        embedding_dim: int = 32,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
        architecture: str = V2_ARCH,
        hidden_dim: int = 128,
        combine: str = "sum",
    ) -> None:
        super().__init__()
        self.num_champions = num_champions
        self.embedding_dim = embedding_dim
        self.architecture = architecture
        self.combine = combine

        self.wide = _WideBranch(num_champions)

        if architecture == LEGACY_ARCH:
            self.deep = _LegacyFlatDeepBranch(num_champions, embedding_dim, hidden_dims, dropout)
        elif architecture == V2_ARCH:
            self.deep = _PairwiseDeepBranch(num_champions, embedding_dim, hidden_dim, dropout)
        else:
            raise ValueError(
                f"unknown architecture {architecture!r}; expected one of "
                f"{LEGACY_ARCH!r}, {V2_ARCH!r}"
            )

        if combine == "concat":
            head = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                head.weight.fill_(0.5)
            self.combine_head = head
        elif combine != "sum":
            raise ValueError(f"combine must be 'sum' or 'concat', got {combine!r}")

    # ----- v2 helpers -----
    @property
    def champion_embedding(self) -> nn.Embedding:
        """Direct handle to the deep-branch champion embedding for pretrain init."""
        if isinstance(self.deep, _PairwiseDeepBranch):
            return self.deep.champion_emb
        if isinstance(self.deep, _LegacyFlatDeepBranch):
            return self.deep.embedding
        raise RuntimeError("unsupported deep branch")

    def forward(self, blue_ids: torch.Tensor, red_ids: torch.Tensor) -> torch.Tensor:
        if blue_ids.shape[-1] != 5 or red_ids.shape[-1] != 5:
            raise ValueError(
                f"Expected 5 slots per side, got blue={blue_ids.shape}, red={red_ids.shape}"
            )
        wide_logit = self.wide(blue_ids, red_ids)
        deep_logit = self.deep(blue_ids, red_ids)
        if self.combine == "sum":
            return wide_logit + deep_logit
        # concat
        stacked = torch.stack([wide_logit, deep_logit], dim=1)  # [B, 2]
        return self.combine_head(stacked).squeeze(-1)
