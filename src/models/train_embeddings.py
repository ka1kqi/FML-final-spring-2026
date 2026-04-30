"""
Performance Matrix Factorization Embeddings
Generates champion embeddings by performing TruncatedSVD on the 
historical Synergy and Matchup performance matrices.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

def build_performance_matrices(comp_df, vocab):
    """
    Builds Synergy and Matchup matrices from historical compositions.
    Applies Bayesian smoothing to pull rare matchups toward 50.0.
    """
    n = len(vocab)
    champ_to_idx = {champ: i for i, champ in enumerate(vocab)}
    
    # Store sums and counts to calculate averages
    syn_sum = np.full((n, n), 50.0 * 5) # Bayesian prior: 5 games of 50.0 score
    syn_count = np.full((n, n), 5.0)
    
    match_sum = np.full((n, n), 50.0 * 5)
    match_count = np.full((n, n), 5.0)
    
    for match_id, group in comp_df.groupby("match_id"):
        blue = group[group["team_id"] == 100]
        red = group[group["team_id"] == 200]
        
        if len(blue) != 5 or len(red) != 5: continue
            
        blue_champs = blue["champion_name"].tolist()
        red_champs = red["champion_name"].tolist()
        
        blue_scores = blue["champ_score"].tolist()
        red_scores = red["champ_score"].tolist()
        
        # Filter strictly to vocab
        if not all(c in champ_to_idx for c in blue_champs + red_champs):
            continue
            
        # Synergy
        for i in range(5):
            for j in range(i+1, 5):
                idx_i = champ_to_idx[blue_champs[i]]
                idx_j = champ_to_idx[blue_champs[j]]
                # Asymmetric addition: Yasuo's score when Malphite is ally
                syn_sum[idx_i, idx_j] += blue_scores[i]
                # Malphite's score when Yasuo is ally
                syn_sum[idx_j, idx_i] += blue_scores[j]
                syn_count[idx_i, idx_j] += 1
                syn_count[idx_j, idx_i] += 1
                
                idx_ri = champ_to_idx[red_champs[i]]
                idx_rj = champ_to_idx[red_champs[j]]
                syn_sum[idx_ri, idx_rj] += red_scores[i]
                syn_sum[idx_rj, idx_ri] += red_scores[j]
                syn_count[idx_ri, idx_rj] += 1
                syn_count[idx_rj, idx_ri] += 1
                
        # Matchup
        for i, b in enumerate(blue_champs):
            for j, r in enumerate(red_champs):
                idx_b = champ_to_idx[b]
                idx_r = champ_to_idx[r]
                
                # B vs R -> blue score for champion B
                match_sum[idx_b, idx_r] += blue_scores[i]
                match_count[idx_b, idx_r] += 1
                
                # R vs B -> red score for champion R
                match_sum[idx_r, idx_b] += red_scores[j]
                match_count[idx_r, idx_b] += 1
                
    # Calculate means
    S = syn_sum / syn_count
    M = match_sum / match_count
    
    # Center the matrices around 0 (since 50 is average) to help MF
    S = S - 50.0
    M = M - 50.0
    
    return S, M

class CustomMatrixFactorization:
    """
    Pure-numpy Matrix Factorization using Stochastic Gradient Descent (SGD).
    Learns embeddings U and V such that U * V^T approximates the target matrix.
    """
    def __init__(self, n_items, n_components=32, lr=0.01, reg=0.02, epochs=50):
        self.n_items = n_items
        self.n_components = n_components
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        
        scale = 1.0 / np.sqrt(n_components)
        self.U = np.random.normal(0, scale, (n_items, n_components))
        self.V = np.random.normal(0, scale, (n_items, n_components))
        
    def fit_transform(self, matrix, name=""):
        print(f"    Training Custom MF ({name}) for {self.epochs} epochs...")
        n_pairs = self.n_items * self.n_items
        
        for epoch in range(self.epochs):
            total_error = 0.0
            
            # Randomize order for SGD
            indices = np.random.permutation(n_pairs)
            for idx in indices:
                i = idx // self.n_items
                j = idx % self.n_items
                
                target = matrix[i, j]
                pred = np.dot(self.U[i], self.V[j])
                err = target - pred
                total_error += err ** 2
                
                u_i = self.U[i].copy()
                v_j = self.V[j].copy()
                
                # Gradient update with L2 regularization
                self.U[i] += self.lr * (err * v_j - self.reg * u_i)
                self.V[j] += self.lr * (err * u_i - self.reg * v_j)
            
            if (epoch + 1) % 10 == 0:
                rmse = np.sqrt(total_error / n_pairs)
                print(f"      Epoch {epoch+1}/{self.epochs} | RMSE: {rmse:.4f}")
        return self.U, self.V

def train_champion2vec(comp_df, embed_dim=64):
    """
    Generate embeddings using Custom Matrix Factorization.
    Returns:
        dict: champion_name -> numpy array (shape: embed_dim)
        list: vocabulary (champion names)
    """
    # 1. Build vocabulary
    vocab = sorted(comp_df["champion_name"].unique().tolist())
    
    # 2. Build matrices
    print("  Building Performance Matrices...")
    S, M = build_performance_matrices(comp_df, vocab)
    
    # 3. Factorize matrices
    print("  Running Custom Matrix Factorization...")
    quarter_dim = embed_dim // 4  # 64 // 4 = 16 dimensions per component
    n_champs = len(vocab)
    
    mf_syn = CustomMatrixFactorization(n_champs, n_components=quarter_dim, lr=0.01, reg=0.02, epochs=50)
    u_syn, v_syn = mf_syn.fit_transform(S, name="Synergy")
    
    mf_match = CustomMatrixFactorization(n_champs, n_components=quarter_dim, lr=0.01, reg=0.02, epochs=50)
    u_match, v_match = mf_match.fit_transform(M, name="Matchup")
    
    # 4. Concatenate
    # Each champion gets [16 U_syn | 16 V_syn | 16 U_match | 16 V_match] = 64 dimensions total
    # PATH A: We remove L2 normalization to preserve raw magnitudes for direct score prediction.
    embeddings = np.hstack([u_syn, v_syn, u_match, v_match])
    
    # 5. Build dictionary
    embed_dict = {champ: embeddings[i] for i, champ in enumerate(vocab)}
    
    return embed_dict, vocab




def most_similar(query_champ, embed_dict, top_k=5):
    """
    Find champions with the most similar full-embedding profile by
    true cosine similarity. This measures archetype overlap (how a
    champion reacts to allies/enemies and how it affects them) — it
    does NOT measure synergy with the query. Use `top_synergies` for
    that.
    """
    if query_champ not in embed_dict:
        return []

    q_vec = embed_dict[query_champ]
    q_norm = np.linalg.norm(q_vec)
    if q_norm < 1e-8:
        return []

    sims = []
    for c, vec in embed_dict.items():
        if c == query_champ:
            continue
        denom = q_norm * np.linalg.norm(vec)
        if denom < 1e-8:
            continue
        sim = float(np.dot(q_vec, vec) / denom)
        sims.append((c, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def top_synergies(query_champ, embed_dict, top_k=5, embed_dim=64,
                  primary_role=None):
    """
    Allies that the MF predicts boost `query_champ`'s score the most:
        score(ally) = U_syn[query] · V_syn[ally]

    If `primary_role` (champion -> role string) is provided, champions
    sharing the query's role are excluded — same-role pairs never
    actually appear as allies in real matches, and the MF's prediction
    for them reflects role-archetype similarity rather than synergy.
    """
    if query_champ not in embed_dict:
        return []
    q = embed_dim // 4
    u_syn_q = embed_dict[query_champ][0:q]
    q_role = primary_role.get(query_champ) if primary_role else None
    scored = []
    for c, vec in embed_dict.items():
        if c == query_champ:
            continue
        if q_role is not None and primary_role.get(c) == q_role:
            continue
        v_syn_c = vec[q:2*q]
        scored.append((c, float(np.dot(u_syn_q, v_syn_c))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def top_counters(query_champ, embed_dict, top_k=5, embed_dim=64):
    """
    Enemies the query is predicted to do best against:
        score(enemy) = U_match[query] · V_match[enemy]

    Cross-role here is fine — you face all 5 enemy roles, so no filter.
    """
    if query_champ not in embed_dict:
        return []
    q = embed_dim // 4
    u_m_q = embed_dict[query_champ][2*q:3*q]
    scored = []
    for c, vec in embed_dict.items():
        if c == query_champ:
            continue
        v_m_c = vec[3*q:4*q]
        scored.append((c, float(np.dot(u_m_q, v_m_c))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def compute_primary_roles(comp_df):
    """
    Map champion_name -> most-frequent position in the training data.
    Used to filter same-role pairs from synergy diagnostics.
    """
    counts = (comp_df.groupby(["champion_name", "position"])
                     .size().reset_index(name="n"))
    counts = counts.sort_values("n", ascending=False).drop_duplicates("champion_name")
    return counts.set_index("champion_name")["position"].to_dict()
