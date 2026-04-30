import numpy as np
import json

embed_data = np.load('data/processed/draft_models/champion2vec.npz')
weights = embed_data['weights']
vocab = embed_data['vocab'].tolist()
embed_dict = {vocab[i]: weights[i] for i in range(len(vocab))}

def analyze(champ_a, champ_b):
    a = embed_dict[champ_a]
    b = embed_dict[champ_b]
    
    u_syn_a, v_syn_a = a[0:16], a[16:32]
    u_syn_b, v_syn_b = b[0:16], b[16:32]
    
    u_match_a, v_match_a = a[32:48], a[48:64]
    u_match_b, v_match_b = b[32:48], b[48:64]
    
    syn = np.dot(u_syn_a, v_syn_b) + np.dot(u_syn_b, v_syn_a)
    a_counters_b = np.dot(u_match_a, v_match_b) - np.dot(u_match_b, v_match_a)
    
    print(f"--- {champ_a} vs {champ_b} ---")
    print(f"Synergy Score: {syn:.4f}")
    if a_counters_b > 0:
        print(f"Counter: {champ_a} counters {champ_b} (Score: {a_counters_b:.4f})")
    else:
        print(f"Counter: {champ_b} counters {champ_a} (Score: {-a_counters_b:.4f})")

analyze("Yasuo", "Malphite")
analyze("Yasuo", "Teemo")
analyze("Sylas", "Malphite")
analyze("Vayne", "Rammus")

