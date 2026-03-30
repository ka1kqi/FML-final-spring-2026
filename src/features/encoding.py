"""
Champion encoding strategies:
- One-hot: binary vector of length num_champions per team
- Learned embeddings: dense vectors via PyTorch nn.Embedding
"""

import numpy as np
import pandas as pd
import torch.nn as nn


def one_hot_encode(champion_ids: list[int], num_champions: int = 170) -> np.ndarray:
    """Return a binary vector with 1s at each champion's index."""
    raise NotImplementedError


def build_team_vector(team_champion_ids: list[int], num_champions: int = 170) -> np.ndarray:
    """Build a one-hot vector for a full 5-champion team."""
    raise NotImplementedError


def build_match_vector(blue_ids: list[int], red_ids: list[int], num_champions: int = 170) -> np.ndarray:
    """Concatenate blue and red team vectors into a single match feature vector."""
    raise NotImplementedError


class ChampionEmbedding(nn.Module):
    """Learnable dense embedding for champion IDs."""

    def __init__(self, num_champions: int = 170, embedding_dim: int = 32):
        super().__init__()
        raise NotImplementedError

    def forward(self, champion_ids):
        raise NotImplementedError
