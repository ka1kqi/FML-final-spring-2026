import numpy as np
import json

embed_data = np.load('data/processed/draft_models/champion2vec.npz')
weights = embed_data['weights']
vocab = embed_data['vocab'].tolist()
embed_dict = {vocab[i]: weights[i] for i in range(len(vocab))}

def top_synergy(champ_a):
    a = embed_dict[champ_a]
    u_syn_a, v_syn_a = a[0:16], a[16:32]
    
    scores = []
    for champ_b in vocab:
        if champ_a == champ_b: continue
        b = embed_dict[champ_b]
        u_syn_b, v_syn_b = b[0:16], b[16:32]
        
        # Synergy is A receiving from B + B receiving from A
        syn = np.dot(u_syn_a, v_syn_b) + np.dot(u_syn_b, v_syn_a)
        scores.append((champ_b, syn))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"Top 5 Synergy for {champ_a}:")
    for c, s in scores[:5]:
        print(f"  {c}: {s:.4f}")

def top_counters(champ_a):
    a = embed_dict[champ_a]
    u_match_a, v_match_a = a[32:48], a[48:64]
    
    scores = []
    for champ_b in vocab:
        if champ_a == champ_b: continue
        b = embed_dict[champ_b]
        u_match_b, v_match_b = b[32:48], b[48:64]
        
        # A counters B
        counter_score = np.dot(u_match_a, v_match_b) - np.dot(u_match_b, v_match_a)
        scores.append((champ_b, counter_score))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"Top 5 Champions that {champ_a} Counters:")
    for c, s in scores[:5]:
        print(f"  {c}: {s:.4f}")
        
    print(f"Top 5 Champions that Counter {champ_a}:")
    for c, s in scores[-5:]:
        print(f"  {c}: {-s:.4f}")

top_synergy("Yasuo")
top_counters("Yasuo")

