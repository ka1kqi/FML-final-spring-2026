from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn


class ChampionEmbedding:
    def __init__(self, vocab_size: int, embed_dim: int):
        scale = (6.0 / (vocab_size + embed_dim)) ** 0.5
        self.weight = torch.empty(vocab_size, embed_dim).uniform_(-scale, scale)
        self.weight = self.weight.requires_grad_(True)

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weight[ids]

    def parameters(self):
        return [self.weight]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        if context is None:
            attn_out, _ = self.attention(x, x, x)
        else:
            attn_out, _ = self.attention(x, context, context)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class EndToEndSetTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.embedding = ChampionEmbedding(vocab_size, d_model)
        self.self_attn_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.cross_attn = TransformerBlock(d_model, num_heads, dropout=dropout)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(query, x, x)
        pooled = self.pool_norm(pooled)
        return pooled.squeeze(1)

    def encode_team(self, team_ids: torch.Tensor, opponent_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(team_ids)
        opp = self.embedding(opponent_ids)
        for layer in self.self_attn_layers:
            x = layer(x)
        x = self.cross_attn(x, context=opp)
        return self.pool(x)

    def forward(self, blue_ids: torch.Tensor, red_ids: torch.Tensor) -> torch.Tensor:
        blue_repr = self.encode_team(blue_ids, red_ids)
        red_repr = self.encode_team(red_ids, blue_ids)
        combined = torch.cat([blue_repr, red_repr], dim=1)
        return self.classifier(combined)


def load_e2e_model(model_path: Path, vocab_size: int) -> EndToEndSetTransformer:
    model = EndToEndSetTransformer(vocab_size=vocab_size)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_blue_win_prob(
    model: EndToEndSetTransformer,
    blue_champions: List[str],
    red_champions: List[str],
    champ_to_id: Dict[str, int],
) -> float:
    blue_ids = [champ_to_id[name] for name in blue_champions]
    red_ids = [champ_to_id[name] for name in red_champions]
    blue_tensor = torch.LongTensor([blue_ids])
    red_tensor = torch.LongTensor([red_ids])
    with torch.no_grad():
        prob = model(blue_tensor, red_tensor).item()
    return float(prob)
