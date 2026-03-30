import pandas as pd
import numpy as np
from pathlib import Path
import random
import math

#load and group data
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / '../data/compositions.csv')

print(df.shape)   
print(df.head(10))

def get_team(group, win):
    team = group[group['win'] == win]['champion_name'].tolist()
    return team

grouped = df.groupby('match_id')

matches = []
for match_id, group in grouped:
    winners = get_team(group, True)
    losers = get_team(group, False)
    if len(winners) == 5 and len(losers) == 5:
        matches.append({
            'match_id': match_id,
            'winners': winners,
            'losers': losers
        })

matches_df = pd.DataFrame(matches)
print(f'effective games: {len(matches_df)}')
print(matches_df.head())




#set up champ list
all_champions = sorted(df['champion_name'].unique())
vocab_size = len(all_champions)

champ_to_id = {name: i for i, name in enumerate(all_champions)}
id_to_champ = {i: name for i, name in enumerate(all_champions)}

print(f'champ_num: {vocab_size}')
print(f'Yasuo ID: {champ_to_id.get("Yasuo")}')





#generate training pairs
def generate_pairs(matches_df, champ_to_id, neg_k=5):

    all_ids = list(range(len(champ_to_id)))
    pairs = []
    
    for _, row in matches_df.iterrows():
        for team in [row['winners'], row['losers']]:
            team_ids = [champ_to_id[name] for name in team]
            team_set = set(team_ids)
            
            for i in range(5):
                for j in range(5):
                    if i == j:
                        continue
                    
                    pairs.append((team_ids[i], team_ids[j], 1.0))
                    
                    
                    for _ in range(neg_k):
                        neg = random.choice(all_ids)
                        while neg in team_set:
                            neg = random.choice(all_ids)
                        pairs.append((team_ids[i], neg, 0.0))
    
    random.shuffle(pairs)
    return pairs

pairs = generate_pairs(matches_df, champ_to_id, neg_k=5)

pairs_df = pd.DataFrame(pairs, columns=['center', 'context', 'label'])
print(f'sample_size: {len(pairs_df)}')
print(f'our: {(pairs_df["label"]==1).sum()}, neg: {(pairs_df["label"]==0).sum()}')
print(pairs_df.head(10))






#embedding algo (to vector)
class Champion2Vec:
    def __init__(self, vocab_size, embed_dim=64):

        scale = np.sqrt(6.0 / (vocab_size + embed_dim))
        self.W_center = np.random.uniform(-scale, scale, (vocab_size, embed_dim))
        self.W_context = np.random.uniform(-scale, scale, (vocab_size, embed_dim))
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
    
    def sigmoid(self, x):
      
        x = np.clip(x, -20, 20)
        return 1.0 / (1.0 + np.exp(-x))
    
    def train_step(self, center_id, context_id, label, lr):

        vec_c = self.W_center[center_id]   
        vec_o = self.W_context[context_id]   
        
        dot = np.dot(vec_c, vec_o)       
        prob = self.sigmoid(dot)
        
        grad = prob - label   
        
        self.W_center[center_id]   -= lr * grad * vec_o
        self.W_context[context_id] -= lr * grad * vec_c
        

        eps = 1e-7
        loss = -(label * np.log(prob + eps) + (1 - label) * np.log(1 - prob + eps))
        return loss
    
    def train(self, pairs, epochs=10, lr=0.025):
        total_steps = len(pairs) * epochs
        step = 0
        
        for epoch in range(epochs):
            random.shuffle(pairs)
            total_loss = 0.0
            
            for center, context, label in pairs:
                current_lr = max(lr * (1.0 - step / total_steps), lr * 0.01)
                loss = self.train_step(int(center), int(context), label, current_lr)
                total_loss += loss
                step += 1
            
            avg = total_loss / len(pairs)
            print(f'Epoch {epoch+1}/{epochs} | Loss: {avg:.4f}')
    
    def get_embedding(self, champ_id):
        return self.W_center[champ_id]
    





#training
model = Champion2Vec(vocab_size, embed_dim=64)
model.train(pairs, epochs=10, lr=0.025)



#validation
def most_similar(model, champ_name, top_k=5):
    target = model.get_embedding(champ_to_id[champ_name])
    
    # cos simi
    scores = []
    for cid in range(model.vocab_size):
        if id_to_champ[cid] == champ_name:
            continue
        vec = model.get_embedding(cid)
        cos_sim = np.dot(target, vec) / (np.linalg.norm(target) * np.linalg.norm(vec) + 1e-8)
        scores.append((id_to_champ[cid], cos_sim))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# teast case
for champ in ['Yasuo', 'Jinx', 'Thresh', 'LeeSin']:
    print(f'\n与 {champ} most similar:')
    for name, score in most_similar(model, champ):
        print(f'  {name}: {score:.4f}')




#save

embed_data = []
for cid in range(vocab_size):
    row = {'champion': id_to_champ[cid]}
    vec = model.get_embedding(cid)
    for d in range(model.embed_dim):
        row[f'd{d}'] = vec[d]
    embed_data.append(row)

embed_df = pd.DataFrame(embed_data)
embed_df.to_csv('data/champion_embeddings.csv', index=False)
print(f'Saved: {embed_df.shape}')
print(embed_df.head())