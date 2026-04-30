"""
Champion encoding strategies:
- One-hot: binary vector of length num_champions per team
- Learned embeddings: dense vectors via PyTorch nn.Embedding
"""

import numpy as np
import torch
import torch.nn as nn


def one_hot_encode(champion_ids: list[int], num_champions: int = 170) -> np.ndarray:
    """Return a binary vector with 1s at each champion's index."""
    vec = np.zeros(num_champions, dtype=np.float32)
    for cid in champion_ids:
        if cid is not None and 0 <= int(cid) < num_champions:
            vec[int(cid)] = 1.0
    return vec


def build_team_vector(team_champion_ids: list[int], num_champions: int = 170) -> np.ndarray:
    """Build a one-hot vector for a full 5-champion team."""
    return one_hot_encode(team_champion_ids, num_champions)


def build_match_vector(blue_ids: list[int], red_ids: list[int], num_champions: int = 170) -> np.ndarray:
    """Concatenate blue and red team vectors into a single match feature vector."""
    blue_vec = build_team_vector(blue_ids, num_champions)
    red_vec = build_team_vector(red_ids, num_champions)
    return np.concatenate([blue_vec, red_vec])


class ChampionEmbedding(nn.Module):
    """Learnable dense embedding for champion IDs."""

    def __init__(self, num_champions: int = 170, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_champions, embedding_dim)

    def forward(self, champion_ids):
        # champion_ids: (batch_size, num_champions_per_team)
        # returns: (batch_size, num_champions_per_team, embedding_dim)
        return self.embedding(champion_ids)

    def embed_team(self, champion_ids):
        """Mean-pool embeddings for a team into a single vector."""
        embedded = self.forward(champion_ids)
        return embedded.mean(dim=-2)