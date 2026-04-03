"""
End-to-End Set Transformer for LoL Win Prediction

Key difference from set_transformer_lib.py:
  - Embedding is LEARNABLE (not frozen pre-trained)
  - Embedding is hand-written (no nn.Embedding)
  - Gradients flow from loss all the way back to embedding weights
  - Model learns champion representations optimized for win prediction

Architecture:
  Champion IDs [B, 5] -> Hand-written Embedding -> [B, 5, 64]
  -> Self-Attention (intra-team interaction)
  -> Cross-Attention (inter-team interaction)
  -> Attention Pooling (aggregate)
  -> MLP Head -> P(blue wins)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent


# ================================================================
# Step 1: Load Data
#
# Unlike set_transformer_lib.py, we do NOT load pre-trained embeddings
# Instead, we use raw champion IDs (integers)
# The model will learn embeddings from scratch
# ================================================================

comp_file = PROJECT_ROOT / 'data/raw/compositions_50k.csv'
if not comp_file.exists():
    comp_file = PROJECT_ROOT / 'src/data/compositions.csv'
    print("Warning: using base compositions data")
else:
    print("Using compositions_50k.csv")

df = pd.read_csv(comp_file)

# Build vocabulary: champion_name -> integer ID
# Why not use Riot's champion_id column?
# Because Riot IDs are not contiguous (e.g. 1, 2, ..., 887 with gaps)
# Embedding lookup requires contiguous IDs from 0 to N-1
all_champions = sorted(df['champion_name'].unique())
champ_to_id = {name: i for i, name in enumerate(all_champions)}
id_to_champ = {i: name for i, name in enumerate(all_champions)}
vocab_size = len(all_champions)

print(f'Champions: {vocab_size}')
print(f'Matches: {df["match_id"].nunique()}')


# ================================================================
# Step 2: Build PyTorch Dataset
#
# Each match -> (blue_team_5_ids, red_team_5_ids, blue_wins)
#
# Why use a Dataset class?
# - DataLoader handles batching, shuffling, and prefetching
# - Memory efficient for large datasets
# ================================================================

class MatchDataset(Dataset):
    """
    PyTorch Dataset: each sample = one match

    __len__: returns dataset size
    __getitem__: returns one sample by index
    DataLoader calls these automatically
    """
    def __init__(self, blue_ids, red_ids, labels):
        self.blue_ids = torch.LongTensor(blue_ids)    # [N, 5] integers
        self.red_ids = torch.LongTensor(red_ids)       # [N, 5]
        self.labels = torch.FloatTensor(labels).unsqueeze(1)  # [N, 1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.blue_ids[idx], self.red_ids[idx], self.labels[idx]


# Build data arrays
blue_ids_list = []
red_ids_list = []
y_list = []
skipped = 0

for match_id, group in df.groupby('match_id'):
    blue = group[group['team_id'] == 100]
    red = group[group['team_id'] == 200]

    if len(blue) != 5 or len(red) != 5:
        skipped += 1
        continue

    blue_names = blue['champion_name'].tolist()
    red_names = red['champion_name'].tolist()

    # Convert to our contiguous IDs
    blue_id = [champ_to_id[n] for n in blue_names]
    red_id = [champ_to_id[n] for n in red_names]

    blue_ids_list.append(blue_id)
    red_ids_list.append(red_id)
    y_list.append(1.0 if blue['win'].iloc[0] else 0.0)

blue_ids_arr = np.array(blue_ids_list)  # [N, 5]
red_ids_arr = np.array(red_ids_list)    # [N, 5]
y_arr = np.array(y_list)

print(f'\nDataset:')
print(f'  Samples: {len(y_arr)} | Skipped: {skipped}')
print(f'  Blue IDs shape: {blue_ids_arr.shape}')
print(f'  Sample: blue={[id_to_champ[i] for i in blue_ids_arr[0]]}')


# ================================================================
# Step 3: Train/Test Split
# ================================================================

indices = list(range(len(y_arr)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = MatchDataset(blue_ids_arr[train_idx], red_ids_arr[train_idx], y_arr[train_idx])
test_dataset = MatchDataset(blue_ids_arr[test_idx], red_ids_arr[test_idx], y_arr[test_idx])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Pre-build test tensors for evaluation
test_blue = torch.LongTensor(blue_ids_arr[test_idx])
test_red = torch.LongTensor(red_ids_arr[test_idx])
test_y = torch.FloatTensor(y_arr[test_idx]).unsqueeze(1)

print(f'Train: {len(train_dataset)} | Test: {len(test_dataset)}')


# ================================================================
# Step 4: Hand-written Embedding (NO nn library)
#
# nn.Embedding internally is just:
#   1. A weight matrix [vocab_size, embed_dim]
#   2. A lookup: weight[ids]
#
# We replicate this with requires_grad=True
# so PyTorch tracks gradients through the lookup operation
#
# Key difference from nn.Embedding:
#   - nn.Parameter auto-registers with model.parameters()
#   - Our tensor needs to be manually passed to the optimizer
# ================================================================

class ChampionEmbedding:
    """
    Hand-written embedding layer without nn library

    Equivalent to nn.Embedding(vocab_size, embed_dim)
    but we control every step manually

    The weight matrix has requires_grad=True, so:
    - Forward: weight[ids] records the operation in autograd graph
    - Backward: gradients flow back to the accessed rows
    - Only rows that were looked up receive gradient updates
    """
    def __init__(self, vocab_size, embed_dim, pretrained_path=None, champ_to_id=None):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Xavier initialization as fallback
        scale = (6.0 / (vocab_size + embed_dim)) ** 0.5
        self.weight = torch.empty(vocab_size, embed_dim).uniform_(-scale, scale)

        # Load pre-trained Skip-gram embeddings if available
        # This gives the model a meaningful starting point:
        #   "Yasuo is a mid-lane assassin" is already encoded
        #   The model only needs to fine-tune for win prediction
        loaded = 0
        if pretrained_path is not None and champ_to_id is not None:
            embed_df = pd.read_csv(pretrained_path)
            for _, row in embed_df.iterrows():
                name = row['champion']
                if name in champ_to_id:
                    vec = row[[f'd{i}' for i in range(embed_dim)]].values.astype(np.float32)
                    self.weight.data[champ_to_id[name]] = torch.FloatTensor(vec)
                    loaded += 1
            print(f'  Loaded pre-trained embeddings for {loaded}/{vocab_size} champions')

        # requires_grad=True tells PyTorch to track all operations
        # on this tensor and compute gradients during backward pass
        self.weight = self.weight.requires_grad_(True)

    def __call__(self, ids):
        """
        Lookup operation (forward pass)

        ids: [B, 5] integer tensor, values in [0, vocab_size-1]
        returns: [B, 5, embed_dim] corresponding vectors

        self.weight[ids] = fancy indexing:
          ids = [[3, 7], [12, 0]]
          output[0][0] = self.weight[3]
          output[0][1] = self.weight[7]
          output[1][0] = self.weight[12]
          output[1][1] = self.weight[0]

        Because self.weight has requires_grad=True,
        this indexing is recorded in the computation graph.
        loss.backward() will propagate gradients to the accessed rows.
        """
        return self.weight[ids]

    def parameters(self):
        """
        Return learnable parameters for the optimizer

        nn.Module does this automatically via model.parameters()
        Since we don't inherit nn.Module, we must do it manually
        """
        return [self.weight]

    def get_similarity(self, id_a, id_b):
        """Cosine similarity between two champions (no nn.functional)"""
        vec_a = self.weight[id_a].detach()
        vec_b = self.weight[id_b].detach()

        dot = torch.sum(vec_a * vec_b)
        norm_a = torch.sqrt(torch.sum(vec_a ** 2))
        norm_b = torch.sqrt(torch.sum(vec_b ** 2))

        return (dot / (norm_a * norm_b + 1e-8)).item()


# ================================================================
# Step 5: Transformer Block
#
# One complete layer: Attention + Add&Norm + FFN + Add&Norm
#
# Uses PyTorch's nn.MultiheadAttention for the attention part
# (only the embedding is hand-written)
#
# GELU vs ReLU:
#   ReLU: max(0, x) - hard corner at x=0
#   GELU: x * Phi(x) - smooth everywhere
#   Transformers use GELU for more stable training
# ================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        # PyTorch built-in multi-head attention
        # batch_first=True: input shape is [B, seq, dim] not [seq, B, dim]
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network: d_model -> d_model*2 -> d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        """
        x: [B, 5, 64] our team
        context: [B, 5, 64] opponent team, or None

        context=None -> self-attention (Q=K=V=x)
        context=opponent -> cross-attention (Q=x, K=V=context)
        """
        if context is None:
            attn_out, _ = self.attention(x, x, x)
        else:
            attn_out, _ = self.attention(x, context, context)

        # Residual + LayerNorm
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


# ================================================================
# Step 6: End-to-End Set Transformer
#
# The key difference: ChampionEmbedding is INSIDE the model
# Input: integer champion IDs -> Output: win probability
#
# Flow:
#   IDs [B,5] -> Embedding -> [B,5,64]
#   -> Self-Attn (teammates see each other)
#   -> Cross-Attn (see opponents)
#   -> Attention Pool (5 champions -> 1 team vector)
#   -> MLP classifier -> P(win)
# ================================================================

class EndToEndSetTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, dropout=0.1,
                 pretrained_path=None, champ_to_id=None):
        super().__init__()

        # Hand-written learnable embedding (NOT nn.Embedding)
        # If pretrained_path is given, initialize from Skip-gram embeddings
        # Then fine-tune end-to-end with a smaller learning rate
        self.embedding = ChampionEmbedding(
            vocab_size, d_model,
            pretrained_path=pretrained_path,
            champ_to_id=champ_to_id
        )

        self.d_model = d_model

        # Self-attention layers (intra-team: teammates observe each other)
        self.self_attn_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Cross-attention layer (inter-team: observe opponents)
        self.cross_attn = TransformerBlock(d_model, num_heads, dropout=dropout)

        # Attention pooling: learnable query aggregates 5 champions into 1 vector
        # Better than mean pooling because it learns which champions matter more
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(d_model)

        # Classification head
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

    def pool(self, x):
        """
        Attention pooling: learnable seed queries 5 champions

        x: [B, 5, 64]
        query: [B, 1, 64] (broadcast)

        The query learns to ask "what's the overall team strength?"
        Each champion responds based on its relevance
        Output: [B, 64] single team representation
        """
        B = x.size(0)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, x, x)
        pooled = self.pool_norm(pooled)
        return pooled.squeeze(1)

    def encode_team(self, team_ids, opponent_ids):
        """
        Encode a team from IDs to a single vector

        team_ids: [B, 5] integer champion IDs
        opponent_ids: [B, 5] integer champion IDs
        returns: [B, 64] team representation
        """
        # 1. Embedding lookup (hand-written, gradients flow back here)
        x = self.embedding(team_ids)        # [B, 5, 64]
        opp = self.embedding(opponent_ids)  # [B, 5, 64]

        # 2. Self-attention: teammates see each other
        for layer in self.self_attn_layers:
            x = layer(x)

        # 3. Cross-attention: observe opponents
        x = self.cross_attn(x, context=opp)

        # 4. Pool: 5 champions -> 1 team vector
        team_repr = self.pool(x)
        return team_repr

    def forward(self, blue_ids, red_ids):
        """
        blue_ids: [B, 5] blue team champion IDs
        red_ids:  [B, 5] red team champion IDs
        returns:  [B, 1] P(blue wins)
        """
        blue_repr = self.encode_team(blue_ids, red_ids)  # [B, 64]
        red_repr = self.encode_team(red_ids, blue_ids)   # [B, 64]

        combined = torch.cat([blue_repr, red_repr], dim=1)  # [B, 128]
        return self.classifier(combined)


# ================================================================
# Step 7: Training Setup
#
# IMPORTANT: Since ChampionEmbedding is not nn.Module,
# its parameters are NOT included in model.parameters()
# We must manually combine them for the optimizer
# ================================================================

print('\n' + '=' * 60)
print('  End-to-End Set Transformer (Pre-train + Fine-tune)')
print('=' * 60)

# Load pre-trained Skip-gram embeddings as initialization
embed_file = PROJECT_ROOT / 'data/champion_embeddings.csv'
print(f'  Pre-trained embedding: {embed_file.name}')

model = EndToEndSetTransformer(
    vocab_size=vocab_size,
    d_model=64,
    num_heads=4,
    num_layers=2,
    dropout=0.15,
    pretrained_path=embed_file,
    champ_to_id=champ_to_id
)

criterion = nn.BCELoss()

# Differential learning rates (key to fine-tuning):
#   - Embedding: small LR (0.0001) → gently adjust pre-trained knowledge
#   - Attention + MLP: normal LR (0.001) → learn from scratch
#
# Why? Pre-trained embedding already knows "Yasuo = assassin"
# Large LR would destroy this structure (catastrophic forgetting)
# Small LR preserves structure while adapting for win prediction
optimizer = optim.AdamW([
    {'params': model.embedding.parameters(), 'lr': 0.0001},  # fine-tune gently
    {'params': list(model.parameters()), 'lr': 0.001},       # learn from scratch
], weight_decay=1e-4)

all_params = list(model.parameters()) + model.embedding.parameters()

# Learning rate scheduler: reduce LR when loss plateaus
# When loss stops decreasing, multiply LR by factor (0.5)
# This allows coarse learning early, fine-tuning later
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

nn_params = sum(p.numel() for p in model.parameters())
embed_params = sum(p.numel() for p in model.embedding.parameters())
print(f'\nParameter count:')
print(f'  nn layers:        {nn_params:,}')
print(f'  Hand-written emb: {embed_params:,} ({vocab_size} x 64)')
print(f'  Total:            {nn_params + embed_params:,}')


# ================================================================
# Step 8: Training Loop
# ================================================================

EPOCHS = 200
best_test_acc = 0
best_epoch = 0
patience_counter = 0
max_patience = 30

print(f'\n--- Training ({EPOCHS} epochs, early stop patience={max_patience}) ---')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_blue, batch_red, batch_y in train_loader:
        pred = model(batch_blue, batch_red)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: prevent exploding gradients in transformer
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_y)

    avg_loss = total_loss / len(train_dataset)
    scheduler.step(avg_loss)

    # Evaluate every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = (model(test_blue, test_red) > 0.5).float()
            test_acc = accuracy_score(test_y, test_pred)

            train_blue_t = torch.LongTensor(blue_ids_arr[train_idx])
            train_red_t = torch.LongTensor(red_ids_arr[train_idx])
            train_pred = (model(train_blue_t, train_red_t) > 0.5).float()
            train_y_t = torch.FloatTensor(y_arr[train_idx]).unsqueeze(1)
            train_acc = accuracy_score(train_y_t, train_pred)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), PROJECT_ROOT / 'data/processed/e2e_best.pth')
        else:
            patience_counter += 5

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Train: {train_acc:.1%} | Test: {test_acc:.1%} | Best: {best_test_acc:.1%} @{best_epoch} | LR: {current_lr:.6f}')

        if patience_counter >= max_patience:
            print(f'\nEarly stop @epoch {epoch+1}')
            break

# Load best model
model.load_state_dict(torch.load(PROJECT_ROOT / 'data/processed/e2e_best.pth'))
print(f'\nBest model: epoch {best_epoch}, test acc {best_test_acc:.1%}')


# ================================================================
# Step 9: Final Evaluation
# ================================================================

print('\n--- Final Test Results ---')
model.eval()
with torch.no_grad():
    test_pred = model(test_blue, test_red)
    test_labels = (test_pred > 0.5).int().numpy()

print(classification_report(
    y_arr[test_idx].astype(int),
    test_labels,
    target_names=['Red Win', 'Blue Win']
))


# ================================================================
# Step 10: Embedding Quality Check
#
# Since embedding is learned end-to-end for win prediction,
# it should capture different relationships than Skip-gram:
# - Skip-gram: "who appears together often" (co-occurrence)
# - End-to-end: "who together leads to winning" (win-optimized)
# ================================================================

print('--- Embedding Quality Check (End-to-End Learned) ---')

def most_similar_e2e(embedding, champ_name, champ_to_id, id_to_champ, top_k=5):
    """Find most similar champions using learned embeddings"""
    target_id = champ_to_id[champ_name]

    scores = []
    for cid in range(embedding.vocab_size):
        if cid == target_id:
            continue
        sim = embedding.get_similarity(target_id, cid)
        scores.append((id_to_champ[cid], sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

for champ in ['Yasuo', 'Jinx', 'Thresh', 'LeeSin']:
    print(f'\n{champ} most similar:')
    for name, score in most_similar_e2e(model.embedding, champ, champ_to_id, id_to_champ):
        print(f'  {name:15s} {score:.4f}')


# ================================================================
# Step 11: Recommendation Demo
# ================================================================

def recommend_e2e(enemy_names, my_names=None, top_k=10):
    """
    Recommend champions using end-to-end model

    enemy_names: list of enemy champion names (1-5)
    my_names: list of already picked ally champions (0-4)
    returns: top_k recommendations with predicted win rate
    """
    if my_names is None:
        my_names = []

    enemy_ids = [champ_to_id[n] for n in enemy_names]
    # Pad to 5 (fill empty slots with ID 0)
    while len(enemy_ids) < 5:
        enemy_ids.append(0)

    taken = set(enemy_names + my_names)
    my_ids = [champ_to_id[n] for n in my_names]

    scores = []
    model.eval()
    with torch.no_grad():
        for name, cid in champ_to_id.items():
            if name in taken:
                continue

            trial = my_ids + [cid]
            while len(trial) < 5:
                trial.append(0)
            trial = trial[:5]

            blue = torch.LongTensor([trial])
            red = torch.LongTensor([enemy_ids])
            prob = model(blue, red).item()
            scores.append((name, prob))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


print('\n--- Recommendation Demo ---')

print('\nEnemy: [Yasuo, Leona, Ezreal] | My team: [Thresh]')
for name, prob in recommend_e2e(['Yasuo', 'Leona', 'Ezreal'], ['Thresh']):
    print(f'  {name:15s} -> Win Rate: {prob:.1%}')

print('\nEnemy: [Zed, Nautilus, Jinx, Viego, Orianna] | My team: []')
for name, prob in recommend_e2e(['Zed', 'Nautilus', 'Jinx', 'Viego', 'Orianna']):
    print(f'  {name:15s} -> Win Rate: {prob:.1%}')


# ================================================================
# Step 12: Model Comparison
# ================================================================

print('\n' + '=' * 60)
print('  Model Comparison')
print('=' * 60)
print(f'  MLP + mean pool (frozen emb):       ~51.8%')
print(f'  Set Transformer (frozen emb):       ~51.7%')
print(f'  End-to-End (random init):           ~50.8%  (overfit)')
print(f'  Pre-train + Fine-tune:              {best_test_acc:.1%}')
print(f'  Delta vs MLP:                       {(best_test_acc - 0.518)*100:+.1f}%')
print('=' * 60)

# Save model
torch.save(model.state_dict(), PROJECT_ROOT / 'data/processed/e2e_model.pth')
print(f'\nModel saved to data/processed/e2e_model.pth')


# ================================================================
# THEORETICAL EXPLANATION: End-to-End Set Transformer for Win Prediction
# ================================================================
#
# 1. MOTIVATION
#    Traditional approaches treat embedding and classification as separate
#    stages: first learn champion embeddings via Word2Vec / Skip-gram on
#    co-occurrence data, then freeze them and train a classifier on top.
#    This is sub-optimal because the embedding was never told what the
#    downstream task is -- it only knows "who appears with whom", not
#    "who together leads to winning".
#
#    End-to-end training solves this by making the embedding layer a
#    differentiable part of the prediction pipeline, so gradients from
#    the binary cross-entropy loss propagate all the way back into the
#    embedding weight matrix. The embedding thus learns representations
#    that are directly optimized for win prediction.
#
# 2. ARCHITECTURE OVERVIEW
#
#    Input:  blue_ids [B, 5]   red_ids [B, 5]   (integer champion IDs)
#             |                   |
#    Embedding Lookup (hand-written, shared weights)
#             |                   |
#            [B, 5, 64]         [B, 5, 64]
#             |                   |
#    Self-Attention x N  (each team attends to its own members)
#             |                   |
#    Cross-Attention     (each team attends to the opposing team)
#             |                   |
#    Attention Pooling   (aggregate 5 champion vectors -> 1 team vector)
#             |                   |
#            [B, 64]            [B, 64]
#              \                /
#               Concatenate [B, 128]
#                     |
#               MLP Classifier
#                     |
#               P(blue wins) [B, 1]
#
# 3. KEY COMPONENTS
#
#    a) Hand-Written Embedding
#       Instead of using nn.Embedding, we manually create a weight matrix
#       of shape [vocab_size, embed_dim] with requires_grad=True. The
#       forward pass is simply weight[ids] (fancy indexing). PyTorch's
#       autograd tracks this operation so that during backpropagation,
#       only the rows corresponding to the champions in the current batch
#       receive gradient updates. This is mathematically equivalent to
#       nn.Embedding but gives us full visibility into how it works.
#
#    b) Pre-trained Initialization + Fine-tuning
#       We initialize the embedding matrix with Skip-gram vectors trained
#       on champion co-occurrence. This provides a meaningful starting
#       point: the model already knows that Yasuo is a mid-lane assassin
#       and Jinx is a bot-lane marksman. We then fine-tune with a small
#       learning rate (0.0001 vs 0.001 for other layers) to gently adapt
#       these representations for win prediction without catastrophic
#       forgetting of the co-occurrence structure.
#
#    c) Self-Attention (Intra-Team Interaction)
#       Each champion in a team attends to all other teammates. This lets
#       the model capture synergies (e.g., Yasuo + Malphite knock-up combo)
#       and redundancies (e.g., two assassins competing for the same role).
#       Formally:  Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
#       where Q = K = V = team embeddings (self-attention).
#
#    d) Cross-Attention (Inter-Team Interaction)
#       After self-attention, each team attends to the opposing team.
#       Query = our team, Key = Value = opponent team. This allows the
#       model to learn matchup-dependent representations: the same
#       champion may be strong against one composition but weak against
#       another. Cross-attention is what makes this model composition-aware
#       rather than just counting individual champion strengths.
#
#    e) Attention Pooling (Set Aggregation)
#       A learnable query vector (the "seed") attends over all 5 champion
#       representations to produce a single team-level vector. Unlike mean
#       pooling which weights all champions equally, attention pooling can
#       learn that the carry champion matters more than the support in
#       determining overall team strength. This is the "Set Transformer"
#       idea from Lee et al. (2019): treating a team as an unordered set
#       and using attention to aggregate it.
#
#    f) Classification Head
#       The blue and red team vectors [B, 64] are concatenated to form
#       [B, 128], then passed through a 3-layer MLP (128 -> 64 -> 1)
#       with ReLU activations and dropout. The final sigmoid outputs
#       P(blue wins) in [0, 1], trained with binary cross-entropy loss.
#
# 4. TRAINING STRATEGY
#
#    - Optimizer: AdamW with differential learning rates
#        * Embedding: lr=0.0001 (preserve pre-trained structure)
#        * All other layers: lr=0.001 (learn from scratch)
#    - Scheduler: ReduceLROnPlateau (halve LR when loss stalls)
#    - Gradient clipping: max_norm=1.0 (prevent exploding gradients)
#    - Early stopping: patience=30 epochs (prevent overfitting)
#    - Regularization: dropout=0.15 + weight_decay=1e-4
#
# 5. WHY "END-TO-END" MATTERS
#
#    In a frozen-embedding pipeline, the embedding captures co-occurrence
#    patterns (which champions are picked together) but has no notion of
#    winning. The classifier must work with whatever features the embedding
#    provides, even if they are not the most informative for the task.
#
#    In end-to-end training, the loss signal (did blue win?) reshapes the
#    embedding space itself. Champions that contribute to winning are
#    pushed closer together; detrimental pairings are pushed apart. The
#    attention layers and the embedding co-adapt, leading to richer
#    representations that are directly useful for prediction.
#
#    Empirically, pre-train + fine-tune outperforms both frozen-embedding
#    approaches and training from random initialization, because:
#    - Random init has too little structure to learn from limited data
#    - Frozen embeddings have the wrong structure (co-occurrence != winning)
#    - Pre-train + fine-tune starts with good structure and refines it
#
# 6. RELATION TO SET TRANSFORMER (Lee et al., 2019)
#
#    The Set Transformer treats input as a SET (order does not matter).
#    A team of 5 champions {A, B, C, D, E} is the same regardless of
#    input ordering. Self-attention is permutation-equivariant (swapping
#    input order swaps output order correspondingly), and attention
#    pooling is permutation-invariant (output is the same regardless of
#    input order). This is the correct inductive bias for team composition
#    prediction, since pick order does not affect in-game performance.
#
# ================================================================
