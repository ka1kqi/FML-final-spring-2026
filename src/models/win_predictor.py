import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent


#  Load embeddings 

embed_df = pd.read_csv(BASE_DIR / '../../data/champion_embeddings.csv')

embed_dict = {}
for _, row in embed_df.iterrows():
    name = row['champion']
    vec = row[[f'd{i}' for i in range(64)]].values.astype(np.float32)
    embed_dict[name] = vec

print(f'Loaded {len(embed_dict)} champion embeddings')


# Build dataset

comp_df = pd.read_csv(BASE_DIR / '../data/compositions.csv')

X_list = []
y_list = []
skipped = 0

for match_id, group in comp_df.groupby('match_id'):
    blue = group[group['team_id'] == 100]
    red = group[group['team_id'] == 200]

    if len(blue) != 5 or len(red) != 5:
        skipped += 1
        continue

    blue_names = blue['champion_name'].tolist()
    red_names = red['champion_name'].tolist()

    if not all(n in embed_dict for n in blue_names + red_names):
        skipped += 1
        continue

    blue_mean = np.mean([embed_dict[n] for n in blue_names], axis=0)
    red_mean = np.mean([embed_dict[n] for n in red_names], axis=0)

    x = np.concatenate([blue_mean, red_mean])  # [128]
    X_list.append(x)
    y_list.append(1.0 if blue['win'].iloc[0] else 0.0)

X = np.array(X_list)
y = np.array(y_list)

print(f'Samples: {len(X)} | Skipped: {skipped}')
print(f'X shape: {X.shape}')
print(f'Blue wins: {y.sum():.0f} | Red wins: {len(y)-y.sum():.0f}')


#   Train/Test split 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'Train: {len(X_train)} | Test: {len(X_test)}')

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


#  MLP Model 

class WinPredictor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 128 -> 64
            nn.ReLU(),
            nn.Dropout(0.3),                    # prevent overfitting
            nn.Linear(hidden_dim, 32),          # 64 -> 32
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),                   # 32 -> 1
            nn.Sigmoid()                        # output probability
        )

    def forward(self, x):
        return self.net(x)


model = WinPredictor()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f'\nModel:\n{model}')
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')


# Training loop 

print('\n--- Training ---')
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        pred = model(batch_X)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_X)

    # Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_pred = (model(X_train_t) > 0.5).float()
            test_pred = (model(X_test_t) > 0.5).float()

            train_acc = accuracy_score(y_train_t, train_pred)
            test_acc = accuracy_score(y_test_t, test_pred)
            avg_loss = total_loss / len(X_train)

        print(f'Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1%} | Test Acc: {test_acc:.1%}')


# Final evaluation 

print('\n--- Final Test Results ---')
model.eval()
with torch.no_grad():
    test_pred = model(X_test_t)
    test_labels = (test_pred > 0.5).int().numpy()

print(classification_report(
    y_test.astype(int),
    test_labels,
    target_names=['Red Win', 'Blue Win']
))


# Recommendation demo

def recommend(enemy_champs, my_champs=None, top_k=10):
    """
    enemy_champs: list of enemy champion names (1-5)
    my_champs: list of my team's already picked champions (0-4)
    returns: top_k recommended champions with predicted win rate
    """
    if my_champs is None:
        my_champs = []

    enemy_vecs = [embed_dict[n] for n in enemy_champs]
    enemy_mean = np.mean(enemy_vecs, axis=0)

    # Already picked champions can't be picked again
    taken = set(enemy_champs + my_champs)

    scores = []
    for champ_name, vec in embed_dict.items():
        if champ_name in taken:
            continue

        # Simulate: my_champs + this candidate vs enemy
        my_vecs = [embed_dict[n] for n in my_champs] + [vec]
        my_mean = np.mean(my_vecs, axis=0)

        x = torch.FloatTensor(np.concatenate([my_mean, enemy_mean])).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            win_prob = model(x).item()

        scores.append((champ_name, win_prob))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


print('\n--- Recommendation Demo ---')
print('\nEnemy: [Yasuo, Leona, Ezreal] | My team: [Thresh]')
print('Top 10 picks:')
for name, prob in recommend(['Yasuo', 'Leona', 'Ezreal'], ['Thresh']):
    print(f'  {name:15s} → Win Rate: {prob:.1%}')

print('\nEnemy: [Zed, Nautilus, Jinx, Viego, Orianna] | My team: []')
print('Top 10 picks:')
for name, prob in recommend(['Zed', 'Nautilus', 'Jinx', 'Viego', 'Orianna']):
    print(f'  {name:15s} → Win Rate: {prob:.1%}')


# Save model 

torch.save(model.state_dict(), BASE_DIR / '../data/win_predictor.pth')
print('\nModel saved to data/win_predictor.pth')
