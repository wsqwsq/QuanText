# chow_liu.py
import math
from collections import defaultdict, Counter, deque
import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import json

# -------- I/O (robust to encoding/TSV) --------
def read_csv_robust(path):
    # Try UTF-8 first, then fall back to cp1252/latin-1 if needed
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            # sep=None + engine='python' will auto-detect tabs vs commas
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, dtype=str)
        except Exception as e:
            last_err = e
    raise last_err

# -------- Discrete MI (no smoothing; observed pairs only) --------
def mutual_information(x, y):
    """x, y are iterables of discrete values (length N, no NaNs). Returns MI in nats."""
    N = len(x)
    assert N == len(y)
    # marginals
    cx = Counter(x)
    cy = Counter(y)
    # joint
    cxy = Counter(zip(x, y))

    mi = 0.0
    for (vx, vy), c in cxy.items():
        pxy = c / N
        px = cx[vx] / N
        py = cy[vy] / N
        mi += pxy * math.log(pxy / (px * py))
    return mi

# -------- Kruskal for Maximum Spanning Tree --------
class DSU:
    def __init__(self, items):
        self.parent = {i: i for i in items}
        self.rank = {i: 0 for i in items}
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

def max_spanning_tree(nodes, weighted_edges):
    """
    nodes: list of node names
    weighted_edges: list of (w, u, v) with w >= 0 (or any real), undirected
    returns set of undirected edges {(u,v), ...} with u<v by name order
    """
    dsu = DSU(nodes)
    # sort by weight DESC for maximum spanning tree
    edges = sorted(weighted_edges, key=lambda t: t[0], reverse=True)
    mst = set()
    for w, u, v in edges:
        if dsu.union(u, v):
            if u < v:
                mst.add((u, v))
            else:
                mst.add((v, u))
        if len(mst) == len(nodes) - 1:
            break
    return mst

# -------- Chow-Liu core --------
def chow_liu_tree(df, cols, root=None):
    """
    df: DataFrame with categorical columns
    cols: list of column names (e.g., ['Target','Stance','Sentiment'])
    root: optional root variable name; if None, choose the node with max degree in MST
    Returns:
      - directed_edges: list of (parent, child)
      - P: dict of marginals {var: {value: prob}}
      - CPT: dict {(child,parent): {parent_val: {child_val: prob}}}
    """
    data = df[cols].dropna()
    # compute pairwise MI
    edges_w = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            u, v = cols[i], cols[j]
            mi = mutual_information(list(data[u]), list(data[v]))
            edges_w.append((mi, u, v))

    # Build undirected MST
    mst = max_spanning_tree(cols, edges_w)

    # Choose root
    if root is None:
        deg = Counter()
        for u, v in mst:
            deg[u] += 1; deg[v] += 1
        root = max(cols, key=lambda c: deg[c])  # high-degree heuristic

    # Orient edges away from root
    adj = defaultdict(list)
    for u, v in mst:
        adj[u].append(v)
        adj[v].append(u)

    directed_edges = []
    seen = set([root])
    q = deque([root])
    parent_of = {root: None}
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                parent_of[v] = u
                directed_edges.append((u, v))
                q.append(v)

    # Estimate marginals P(X) and CPTs P(child|parent)
    N = len(data)
    # marginals
    P = {}
    for var in cols:
        counts = Counter(data[var])
        P[var] = {val: c / N for val, c in counts.items()}

    # conditionals
    CPT = {}
    for (pa, ch) in directed_edges:
        joint = Counter(zip(data[pa], data[ch]))
        pa_counts = Counter(data[pa])
        table = defaultdict(dict)
        for (pval, cval), c in joint.items():
            table[pval][cval] = c / pa_counts[pval]
        # ensure rows sum to 1 even if some child values are unseen under a parent value
        # (optional normalization over observed)
        CPT[(ch, pa)] = {pval: dict(cdict) for pval, cdict in table.items()}

    return root, directed_edges, P, CPT

# -------- Pretty printing & querying --------
def print_model(root, directed_edges, P, CPT):
    print(f"Root: {root}")
    print("Tree edges (parent -> child):")
    for u, v in directed_edges:
        print(f"  {u} -> {v}")
    print("\nMarginals P(X):")
    for var, dist in P.items():
        row = ", ".join(f"{val}: {prob:.4f}" for val, prob in sorted(dist.items()))
        print(f"  {var}: {row}")
    print("\nConditionals P(child | parent):")
    for (ch, pa), table in CPT.items():
        print(f"  {ch} | {pa}:")
        for pval, cdict in table.items():
            row = ", ".join(f"{cval}: {prob:.4f}" for cval, prob in sorted(cdict.items()))
            print(f"    {pa}={pval} -> {row}")

def query_logprob(assign, root, directed_edges, P, CPT):
    """
    assign: dict like {'Target':'Atheism','Stance':'AGAINST','Sentiment':'POSITIVE'}
    returns log P(assign) under the Chow-Liu tree.
    """
    if root not in assign:
        raise ValueError(f"Assignment must include the root '{root}'")
    lp = math.log(P[root].get(assign[root], 1e-30))  # small floor if unseen
    parents = {ch: pa for pa, ch in directed_edges}
    for var, val in assign.items():
        if var == root:
            continue
        pa = parents[var]
        pval = assign[pa]
        prob = CPT.get((var, pa), {}).get(pval, {}).get(val, 1e-30)
        lp += math.log(prob)
    return lp

## -------- Example usage --------
#if __name__ == "__main__":
#    # Replace 'data.tsv' with your actual file path.
#    # Your sample looks tab-separated; keep sep auto-detect for safety.
#    path = "sample.csv"
#
#    # If you already have a DataFrame (e.g., from a notebook), skip read_tsv_robust and pass it directly.
#    try:
#        df = read_csv_robust(path)
#    except Exception as e:
#        print(f"Failed to read '{path}': {e}")
#        print("Tip: ensure it's tab-separated; try saving as UTF-8 or use Excel -> CSV (UTF-8).")
#        raise
#
#    cols = ["Target", "Stance", "Sentiment"]
#    # Keep only the columns we need and drop rows with missing values there
#    df = df[cols].dropna()
#
#    # Learn Chow–Liu tree
#    root, directed_edges, P, CPT = chow_liu_tree(df, cols, root=None)
#
#    # Inspect
#    print_model(root, directed_edges, P, CPT)


# --- construct distribution bins ---

def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))

def value_spaces(df, cols):
    """Return sorted unique values for each categorical column (strings)."""
    spaces = {}
    for c in cols:
        spaces[c] = sorted(df[c].dropna().astype(str).unique().tolist())
    return spaces

# ---------- Diversity-first sampling on the simplex ----------
def _dirichlet_pool(m, pool_size=500, alpha=0.8, rng=None):
    """Draw a pool of Dirichlet(m, alpha) samples (rows sum to 1)."""
    rng = np.random.default_rng() if rng is None else rng
    alpha_vec = np.full(m, alpha, dtype=float)
    return rng.dirichlet(alpha_vec, size=pool_size)

def _l1_dist(a, b):
    return np.abs(a - b).sum(axis=-1)

def _greedy_maxmin_select(pool, K, metric="l1"):
    """
    Greedy farthest-point selection: pick K rows from 'pool' maximizing min distance.
    metric: 'l1' (default)
    """
    N, m = pool.shape
    if K >= N:
        return pool

    # Start with the sample farthest from the uniform vector (encourages extremal start)
    u = np.full(m, 1.0/m)
    d_to_u = _l1_dist(pool, u)
    chosen_idx = [int(np.argmax(d_to_u))]

    # Precompute pairwise distances lazily
    # Maintain min distance to the chosen set for each point
    min_d = _l1_dist(pool, pool[chosen_idx[0]])

    for _ in range(1, K):
        # pick the point with maximum min distance to the selected set
        next_idx = int(np.argmax(min_d))
        chosen_idx.append(next_idx)
        # update min distances using the newly added point
        d_new = _l1_dist(pool, pool[next_idx])
        min_d = np.minimum(min_d, d_new)

    return pool[chosen_idx, :]

def sample_diverse_simplex(m, K, pool_size=1000, alpha=0.2, rng=None):
    """
    Return K x m matrix where each row is a probability vector on the m-simplex.
    - Draw 'pool_size' Dirichlet(m, alpha) candidates (alpha<1 => spiky)
    - Select K that maximize diversity by greedy max-min (L1 distance)
    """
    if m <= 0:
        raise ValueError("m must be >= 1")
    if m == 1:
        return np.ones((K, 1))  # degenerate simplex
    if pool_size < K:
        pool_size = K * 5
    pool = _dirichlet_pool(m, pool_size=pool_size, alpha=alpha, rng=rng)
    return _greedy_maxmin_select(pool, K)

# ---------- Orchestrate for the Chow–Liu structure ----------
def construct_diverse_random_distributions(df, cols, root, directed_edges,
                                           K=100, alpha=0.2, pool_size=2000, outdir="random_chow_liu"):
    """
    Build K diverse random distributions for:
      - P(root)
      - each row of P(child | parent=v), for every parent value v
    *No dependence on empirical counts* (only category sets).
    Saves one CSV per factor row.
    """
    os.makedirs(outdir, exist_ok=True)
    df = df[cols].dropna().astype(str)
    spaces = value_spaces(df, cols)

    summary = {"root": {}, "conditionals": {}}

    # Root
    root_vals = spaces[root]
    root_samples = sample_diverse_simplex(m=len(root_vals), K=K, pool_size=pool_size, alpha=alpha)
    root_file = os.path.join(outdir, f"root_{_safe_name(root)}.csv")
    pd.DataFrame(root_samples, columns=root_vals).to_csv(root_file, index=False)
    summary["root"] = {"var": root, "values": root_vals, "file": root_file}

    # Conditionals
    for pa, ch in directed_edges:
        ch_vals = spaces[ch]
        pa_vals = spaces[pa]
        for pval in pa_vals:
            samples = sample_diverse_simplex(m=len(ch_vals), K=K, pool_size=pool_size, alpha=alpha)
            fname = os.path.join(outdir, f"{_safe_name(ch)}_|_{_safe_name(pa)}={_safe_name(pval)}.csv")
            pd.DataFrame(samples, columns=ch_vals).to_csv(fname, index=False)
            summary.setdefault("conditionals", {}).setdefault((ch, pa), {})[pval] = {
                "child": ch, "parent": pa, "parent_value": pval,
                "child_values": ch_vals, "file": fname
            }
    return summary


# ---------- Build & save a program-readable registry ----------
def build_registry(cols, spaces, root, directed_edges, sample_summary, outdir):
    """
    Create a JSON-serializable registry with:
      - variables (and their value order)
      - tree (root, directed edges)
      - factors, each with:
          id: stable machine id (e.g., "root:Target" or "cpt:Sentiment|Stance=AGAINST")
          name: human-friendly (e.g., "P(Target)" or "P(Sentiment | Stance=AGAINST)")
          scope: variables this distribution is over
          values: column order for the CSV
          file: path to CSV with K rows of samples
    """
    reg = {
        "version": "1.0",
        "columns": cols,
        "variables": {v: {"values": spaces[v]} for v in cols},
        "tree": {
            "root": root,
            "directed_edges": [{"parent": p, "child": c} for (p, c) in directed_edges]
        },
        "factors": []
    }

    def _rel(p):
        # Always store path relative to outdir so it composes correctly
        # with --base-dir in quantization.py.
        return os.path.relpath(p, outdir)

    # Root factor
    root_vals = spaces[root]
    reg["factors"].append({
        "id": f"root:{root}",
        "name": f"P({root})",
        "type": "root",
        "scope": [root],
        "values": root_vals,
        "file": _rel(sample_summary["root"]["file"])
    })

    # Conditional rows
    for (ch, pa), rows in sample_summary["conditionals"].items():
        ch_vals = spaces[ch]
        for pval, meta in rows.items():
            reg["factors"].append({
                "id": f"cpt:{ch}|{pa}={pval}",
                "name": f"P({ch} | {pa}={pval})",
                "type": "conditional_row",
                "scope": [ch],
                "condition": {"parent": pa, "value": pval},
                "values": ch_vals,
                "file": _rel(meta["file"])
            })

    return reg

def save_registry(registry, outdir, filename="model_registry.json"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    return path


# ---------- Driver ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tweet.csv",
                    help="CSV used to learn the Chow-Liu structure (only column categories matter).")
    ap.add_argument("--gamma", type=int, default=120,
                    help="Number of diverse candidate distributions per factor (K).")
    ap.add_argument("--alpha", type=float, default=0.8,
                    help="Dirichlet concentration (lower = spikier).")
    ap.add_argument("--pool-size", type=int, default=1200,
                    help="Dirichlet pool size before greedy max-min selection.")
    ap.add_argument("--outdir", default="tweet_bins",
                    help="Directory to store candidate-distribution CSVs and registry.")
    args = ap.parse_args()

    df = read_csv_robust(args.data)
    cols = ["Target", "Stance", "Sentiment"]
    df = df[cols].dropna()

    root, directed_edges, _, _ = chow_liu_tree(df, cols, root=None)
    print("Tree (parent -> child):", directed_edges, "\nRoot:", root)

    summary = construct_diverse_random_distributions(
        df, cols, root, directed_edges,
        K=args.gamma, alpha=args.alpha, pool_size=args.pool_size, outdir=args.outdir
    )
    print(f"Saved candidate distributions to {args.outdir}")

    spaces = {c: sorted(df[c].unique().tolist()) for c in cols}
    registry = build_registry(cols, spaces, root, directed_edges, summary, outdir=args.outdir)
    reg_path = save_registry(registry, args.outdir, filename="model_registry.json")
    print(f"Registry saved: {reg_path}")

