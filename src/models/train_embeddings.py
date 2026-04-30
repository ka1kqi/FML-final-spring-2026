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
    Pure-numpy MF with per-row and per-column bias terms:

        target[i, j] ~= mu + b_u[i] + b_v[j] + U[i] . V[j]

    Bias terms absorb each champion's overall strength as a row/column,
    so the latent vectors U and V are free to encode *relational*
    chemistry beyond baseline. This produces cleaner directional
    embeddings that survive L2 normalization without losing signal.
    """
    def __init__(self, n_items, n_components=32, lr=0.01, reg=0.02,
                 bias_reg=0.05, epochs=50):
        self.n_items = n_items
        self.n_components = n_components
        self.lr = lr
        self.reg = reg
        self.bias_reg = bias_reg
        self.epochs = epochs

        scale = 1.0 / np.sqrt(n_components)
        self.U = np.random.normal(0, scale, (n_items, n_components))
        self.V = np.random.normal(0, scale, (n_items, n_components))
        self.b_u = np.zeros(n_items)
        self.b_v = np.zeros(n_items)
        self.mu = 0.0

    def fit_transform(self, matrix, name=""):
        print(f"    Training Custom MF ({name}) for {self.epochs} epochs...")
        n_pairs = self.n_items * self.n_items
        self.mu = float(matrix.mean())

        for epoch in range(self.epochs):
            total_error = 0.0

            indices = np.random.permutation(n_pairs)
            for idx in indices:
                i = idx // self.n_items
                j = idx % self.n_items

                target = matrix[i, j]
                pred = self.mu + self.b_u[i] + self.b_v[j] + np.dot(self.U[i], self.V[j])
                err = target - pred
                total_error += err ** 2

                u_i = self.U[i].copy()
                v_j = self.V[j].copy()

                self.b_u[i] += self.lr * (err - self.bias_reg * self.b_u[i])
                self.b_v[j] += self.lr * (err - self.bias_reg * self.b_v[j])
                self.U[i] += self.lr * (err * v_j - self.reg * u_i)
                self.V[j] += self.lr * (err * u_i - self.reg * v_j)

            if (epoch + 1) % 10 == 0:
                rmse = np.sqrt(total_error / n_pairs)
                print(f"      Epoch {epoch+1}/{self.epochs} | RMSE: {rmse:.4f}")
        return self.U, self.V, self.b_u, self.b_v, self.mu

def _row_normalize(M, eps=1e-8):
    """Scale each row to unit L2 norm. Rows with near-zero norm stay as-is."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return M / norms


def build_champion_role_vocab(comp_df, min_games=20, min_role_share=0.10):
    """
    Build the (champion, role) vocabulary used by the role-aware embedding.
    Keys are "Champion|ROLE" strings. Filters to (champ, role) pairs with
    enough games to support a meaningful matchup estimate.
    """
    counts = (comp_df.groupby(["champion_name", "position"]).size()
                     .reset_index(name="games"))
    totals = (comp_df.groupby("champion_name").size()
                     .reset_index(name="total"))
    counts = counts.merge(totals, on="champion_name", how="left")
    counts["role_share"] = counts["games"] / counts["total"]
    counts = counts[(counts["games"] >= min_games)
                    & (counts["role_share"] >= min_role_share)]
    vocab = sorted(f"{c}|{r}" for c, r in zip(counts["champion_name"], counts["position"]))
    return vocab


def build_role_aware_matrices(comp_df, vocab):
    """
    Same logic as build_performance_matrices but the unit is (champion, role).

    For each match, each player contributes their (champion, position) key.
    Synergy: pairs on the same team in their respective lanes.
    Matchup: pairs across teams (any lane against any lane — captures the
    fact that Akali-MID's score is affected by every enemy, not only the
    enemy mid laner). For "lane-vs-lane" queries downstream, callers can
    intersect the matchup matrix to a single role pair.
    """
    n = len(vocab)
    key_to_idx = {key: i for i, key in enumerate(vocab)}

    syn_sum = np.full((n, n), 50.0 * 5)
    syn_count = np.full((n, n), 5.0)
    match_sum = np.full((n, n), 50.0 * 5)
    match_count = np.full((n, n), 5.0)

    for match_id, group in comp_df.groupby("match_id"):
        blue = group[group["team_id"] == 100]
        red = group[group["team_id"] == 200]
        if len(blue) != 5 or len(red) != 5:
            continue

        blue_keys = [f"{r['champion_name']}|{r['position']}" for _, r in blue.iterrows()]
        red_keys = [f"{r['champion_name']}|{r['position']}" for _, r in red.iterrows()]
        blue_scores = blue["champ_score"].tolist()
        red_scores = red["champ_score"].tolist()

        if not all(k in key_to_idx for k in blue_keys + red_keys):
            continue

        for i in range(5):
            for j in range(i + 1, 5):
                bi, bj = key_to_idx[blue_keys[i]], key_to_idx[blue_keys[j]]
                syn_sum[bi, bj] += blue_scores[i]
                syn_sum[bj, bi] += blue_scores[j]
                syn_count[bi, bj] += 1
                syn_count[bj, bi] += 1
                ri, rj = key_to_idx[red_keys[i]], key_to_idx[red_keys[j]]
                syn_sum[ri, rj] += red_scores[i]
                syn_sum[rj, ri] += red_scores[j]
                syn_count[ri, rj] += 1
                syn_count[rj, ri] += 1

        for i, b in enumerate(blue_keys):
            for j, r in enumerate(red_keys):
                ib, ir = key_to_idx[b], key_to_idx[r]
                match_sum[ib, ir] += blue_scores[i]
                match_count[ib, ir] += 1
                match_sum[ir, ib] += red_scores[j]
                match_count[ir, ib] += 1

    S = syn_sum / syn_count - 50.0
    M = match_sum / match_count - 50.0
    return S, M


def train_champion_role_2vec(comp_df, embed_dim=64,
                              min_games=20, min_role_share=0.10):
    """
    Role-aware embedding. Each (champion, role) is a separate unit.

    Returns (embed_dict, vocab, biases) just like train_champion2vec, but
    keys are "Champion|ROLE" strings. Use this for analysis queries like
    "Akali-MID vs Ahri-MID" or "Akali-TOP vs Sylas-TOP" — each role gets
    its own performance profile rather than averaging across roles.
    """
    vocab = build_champion_role_vocab(comp_df, min_games=min_games,
                                       min_role_share=min_role_share)
    print(f"  Role-aware vocab: {len(vocab)} (champion, role) units")

    print("  Building role-aware performance matrices...")
    S, M = build_role_aware_matrices(comp_df, vocab)

    print("  Running role-aware MF (with bias terms)...")
    quarter_dim = embed_dim // 4
    n = len(vocab)

    mf_syn = CustomMatrixFactorization(n, n_components=quarter_dim,
                                        lr=0.01, reg=0.02, bias_reg=0.05, epochs=50)
    u_syn, v_syn, b_syn_u, b_syn_v, mu_syn = mf_syn.fit_transform(S, name="Role-Synergy")

    mf_match = CustomMatrixFactorization(n, n_components=quarter_dim,
                                          lr=0.01, reg=0.02, bias_reg=0.05, epochs=50)
    u_match, v_match, b_match_u, b_match_v, mu_match = mf_match.fit_transform(M, name="Role-Matchup")

    embeddings = np.hstack([u_syn, v_syn, u_match, v_match])
    embed_dict = {key: embeddings[i] for i, key in enumerate(vocab)}

    biases = {
        "mu_syn": float(mu_syn),
        "b_syn_u": b_syn_u.astype(np.float32),
        "b_syn_v": b_syn_v.astype(np.float32),
        "mu_match": float(mu_match),
        "b_match_u": b_match_u.astype(np.float32),
        "b_match_v": b_match_v.astype(np.float32),
    }
    return embed_dict, vocab, biases


def train_champion2vec(comp_df, embed_dim=64):
    """
    Train MF with bias terms. Returns raw (unnormalized) latent vectors —
    the bias terms already separate global strength from relational
    signal, so the U/V vectors carry meaningful magnitudes that
    downstream features (booster, match LR) can exploit.

    Returns:
        embed_dict: dict[champion_name -> 64-d vector]
            Layout: [U_syn(16) | V_syn(16) | U_match(16) | V_match(16)]
            Cosine similarity over a block is comparable across pairs;
            raw dot product gives absolute relational deviation.
        vocab: list[str] champion names in row order.
        biases: dict with absolute-score reconstruction terms:
            'mu_syn', 'b_syn_u', 'b_syn_v',
            'mu_match', 'b_match_u', 'b_match_v'
        Full prediction:
            S[i,j] ~= mu_syn + b_syn_u[i] + b_syn_v[j] + U_syn[i] . V_syn[j]
    """
    vocab = sorted(comp_df["champion_name"].unique().tolist())

    print("  Building Performance Matrices...")
    S, M = build_performance_matrices(comp_df, vocab)

    print("  Running Custom Matrix Factorization with bias terms...")
    quarter_dim = embed_dim // 4
    n_champs = len(vocab)

    mf_syn = CustomMatrixFactorization(n_champs, n_components=quarter_dim,
                                        lr=0.01, reg=0.02, bias_reg=0.05, epochs=50)
    u_syn, v_syn, b_syn_u, b_syn_v, mu_syn = mf_syn.fit_transform(S, name="Synergy")

    mf_match = CustomMatrixFactorization(n_champs, n_components=quarter_dim,
                                          lr=0.01, reg=0.02, bias_reg=0.05, epochs=50)
    u_match, v_match, b_match_u, b_match_v, mu_match = mf_match.fit_transform(M, name="Matchup")

    embeddings = np.hstack([u_syn, v_syn, u_match, v_match])
    embed_dict = {champ: embeddings[i] for i, champ in enumerate(vocab)}

    biases = {
        "mu_syn": float(mu_syn),
        "b_syn_u": b_syn_u.astype(np.float32),
        "b_syn_v": b_syn_v.astype(np.float32),
        "mu_match": float(mu_match),
        "b_match_u": b_match_u.astype(np.float32),
        "b_match_v": b_match_v.astype(np.float32),
    }
    return embed_dict, vocab, biases




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
