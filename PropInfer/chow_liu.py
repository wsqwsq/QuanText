"""Chow–Liu tree + candidate release distributions.

Specialized to PropInfer datasets: a single attribute per CSV (gender or
diagnosis). With one attribute the tree is trivially {root}, no edges —
the same code path that handles multi-attribute trees in the tweet pipeline.
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd


# ---------- I/O ----------
def read_csv_robust(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, dtype=str)
        except Exception:
            pass
    return pd.read_csv(path, dtype=str)


# ---------- MI / Chow-Liu ----------
def mutual_information(x, y):
    N = len(x)
    cx, cy, cxy = Counter(x), Counter(y), Counter(zip(x, y))
    mi = 0.0
    for (vx, vy), c in cxy.items():
        pxy, px, py = c / N, cx[vx] / N, cy[vy] / N
        mi += pxy * math.log(pxy / (px * py))
    return mi


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
        if ra == rb: return False
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1
        return True


def max_spanning_tree(nodes, weighted_edges):
    dsu = DSU(nodes)
    mst = set()
    for w, u, v in sorted(weighted_edges, key=lambda t: t[0], reverse=True):
        if dsu.union(u, v):
            mst.add((u, v) if u < v else (v, u))
        if len(mst) == len(nodes) - 1: break
    return mst


def chow_liu_tree(df, cols, root=None):
    data = df[cols].dropna()
    edges_w = [
        (mutual_information(list(data[cols[i]]), list(data[cols[j]])), cols[i], cols[j])
        for i in range(len(cols)) for j in range(i + 1, len(cols))
    ]
    mst = max_spanning_tree(cols, edges_w)
    if root is None:
        deg = Counter()
        for u, v in mst:
            deg[u] += 1; deg[v] += 1
        root = max(cols, key=lambda c: deg[c])

    adj = defaultdict(list)
    for u, v in mst:
        adj[u].append(v); adj[v].append(u)

    directed_edges, seen, q = [], {root}, deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                directed_edges.append((u, v))
                q.append(v)

    N = len(data)
    P = {c: {v: cnt / N for v, cnt in Counter(data[c]).items()} for c in cols}
    CPT = {}
    for pa, ch in directed_edges:
        joint = Counter(zip(data[pa], data[ch]))
        pa_counts = Counter(data[pa])
        table = defaultdict(dict)
        for (pval, cval), c in joint.items():
            table[pval][cval] = c / pa_counts[pval]
        CPT[(ch, pa)] = {pval: dict(d) for pval, d in table.items()}
    return root, directed_edges, P, CPT


# ---------- Diverse candidate distributions ----------
def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def _dirichlet_pool(m, pool_size, alpha, rng):
    return rng.dirichlet(np.full(m, alpha, dtype=float), size=pool_size)


def _l1_dist(a, b):
    return np.abs(a - b).sum(axis=-1)


def _greedy_maxmin_select(pool, K):
    N, m = pool.shape
    if K >= N: return pool
    u = np.full(m, 1.0 / m)
    chosen = [int(np.argmax(_l1_dist(pool, u)))]
    min_d = _l1_dist(pool, pool[chosen[0]])
    for _ in range(1, K):
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, _l1_dist(pool, pool[nxt]))
    return pool[chosen, :]


def sample_diverse_simplex(m, K, pool_size, alpha, rng):
    if m == 1: return np.ones((K, 1))
    if pool_size < K: pool_size = K * 5
    pool = _dirichlet_pool(m, pool_size, alpha, rng)
    return _greedy_maxmin_select(pool, K)


def value_spaces(df, cols):
    return {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in cols}


def construct_diverse_random_distributions(df, cols, root, directed_edges,
                                           K, alpha, pool_size, outdir, rng):
    os.makedirs(outdir, exist_ok=True)
    df = df[cols].dropna().astype(str)
    spaces = value_spaces(df, cols)
    summary = {"root": {}, "conditionals": {}}

    root_vals = spaces[root]
    root_samples = sample_diverse_simplex(len(root_vals), K, pool_size, alpha, rng)
    root_file = os.path.join(outdir, f"root_{_safe_name(root)}.csv")
    pd.DataFrame(root_samples, columns=root_vals).to_csv(root_file, index=False)
    summary["root"] = {"var": root, "values": root_vals, "file": root_file}

    for pa, ch in directed_edges:
        ch_vals, pa_vals = spaces[ch], spaces[pa]
        for pval in pa_vals:
            samples = sample_diverse_simplex(len(ch_vals), K, pool_size, alpha, rng)
            fname = os.path.join(outdir,
                f"{_safe_name(ch)}_|_{_safe_name(pa)}={_safe_name(pval)}.csv")
            pd.DataFrame(samples, columns=ch_vals).to_csv(fname, index=False)
            summary["conditionals"].setdefault((ch, pa), {})[pval] = {
                "child": ch, "parent": pa, "parent_value": pval,
                "child_values": ch_vals, "file": fname,
            }
    return summary


def build_registry(cols, spaces, root, directed_edges, sample_summary, outdir):
    reg = {
        "version": "1.0",
        "columns": cols,
        "variables": {v: {"values": spaces[v]} for v in cols},
        "tree": {"root": root,
                 "directed_edges": [{"parent": p, "child": c} for (p, c) in directed_edges]},
        "factors": []
    }
    rel = lambda p: os.path.relpath(p, outdir)
    reg["factors"].append({
        "id": f"root:{root}", "name": f"P({root})", "type": "root",
        "scope": [root], "values": spaces[root],
        "file": rel(sample_summary["root"]["file"])
    })
    for (ch, pa), rows in sample_summary["conditionals"].items():
        for pval, meta in rows.items():
            reg["factors"].append({
                "id": f"cpt:{ch}|{pa}={pval}",
                "name": f"P({ch} | {pa}={pval})",
                "type": "conditional_row",
                "scope": [ch],
                "condition": {"parent": pa, "value": pval},
                "values": spaces[ch],
                "file": rel(meta["file"]),
            })
    return reg


def save_registry(reg, outdir, filename="model_registry.json"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Private CSV.")
    ap.add_argument("--cols", default="gender",
                    help="Comma-separated attribute columns to model.")
    ap.add_argument("--gamma",     type=int, default=120)
    ap.add_argument("--alpha",     type=float, default=0.8)
    ap.add_argument("--pool-size", type=int, default=1200)
    ap.add_argument("--outdir",    required=True)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    df = read_csv_robust(args.data)
    df = df[cols].dropna()

    root, edges, _, _ = chow_liu_tree(df, cols, root=None)
    print("Tree (parent→child):", edges, "\nRoot:", root)

    rng = np.random.default_rng(args.seed)
    summary = construct_diverse_random_distributions(
        df, cols, root, edges, K=args.gamma, alpha=args.alpha,
        pool_size=args.pool_size, outdir=args.outdir, rng=rng,
    )
    spaces = {c: sorted(df[c].unique().tolist()) for c in cols}
    save_registry(build_registry(cols, spaces, root, edges, summary, outdir=args.outdir),
                  args.outdir)
    print(f"Registry saved under {args.outdir}/model_registry.json")
