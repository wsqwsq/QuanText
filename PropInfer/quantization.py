"""Randomized quantization step of the SML mechanism.

Reads private CSV + registry produced by chow_liu.py; for each factor
(root marginal and any conditional rows) picks one of the top-k closest
candidate distributions by TV distance; samples |D| property arrays from
the resulting product distribution and matches them to the originals
via Hungarian assignment on weighted Hamming.
"""

import argparse
import json
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def read_table_robust(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, dtype=str)
        except Exception:
            pass
    return pd.read_csv(path, dtype=str)


def load_registry(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_factors(reg, kind=None):
    for f in reg["factors"]:
        if kind is None or f.get("type") == kind:
            yield f


def load_factor_samples(factor, base_dir):
    values = factor["values"]
    fpath = factor["file"]
    if base_dir and not os.path.isabs(fpath):
        fpath = os.path.join(base_dir, fpath)
    df = pd.read_csv(fpath)
    df = df[[v for v in values if v in df.columns]]
    return df, values


def tv_distance(p, q, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, np.inf); p /= p.sum()
    q = np.clip(np.asarray(q, dtype=float), eps, np.inf); q /= q.sum()
    return 0.5 * np.abs(p - q).sum()


def empirical_distribution(series, values_order):
    counts = series.astype(str).value_counts()
    arr = np.array([counts.get(v, 0) for v in values_order], dtype=float)
    if arr.sum() == 0:
        return np.ones(len(values_order)) / len(values_order)
    return arr / arr.sum()


def empirical_conditional(df, child, parent, parent_value, child_values_order):
    mask = (df[parent].astype(str) == str(parent_value))
    if mask.sum() == 0:
        return np.ones(len(child_values_order)) / len(child_values_order)
    return empirical_distribution(df.loc[mask, child], child_values_order)


def choose_factor_row(samples_df, target_probs, k, rng):
    tvs = samples_df.apply(lambda r: tv_distance(target_probs, r.values), axis=1).values
    k = min(k, len(tvs))
    topk = np.argpartition(tvs, kth=k - 1)[:k]
    topk = topk[np.argsort(tvs[topk])]
    chosen = rng.choice(topk)
    return samples_df.iloc[chosen].values, int(chosen)


def topological_order(root, directed_edges):
    adj = defaultdict(list)
    for p, c in directed_edges:
        adj[p].append(c)
    order, parents = [], {root: None}
    seen, q = {root}, deque([root])
    while q:
        u = q.popleft(); order.append(u)
        for v in adj[u]:
            if v not in seen:
                seen.add(v); parents[v] = u; q.append(v)
    return order, parents


def sample_from_categorical(values, probs, rng):
    p = np.asarray(probs, dtype=float); p /= p.sum()
    return rng.choice(values, p=p)


def parse_weights(weights_arg, columns):
    if weights_arg is None:
        return {c: 1.0 for c in columns}
    if os.path.isfile(weights_arg):
        with open(weights_arg, "r", encoding="utf-8") as f:
            w = json.load(f)
    else:
        w = {}
        for tok in weights_arg.split(","):
            tok = tok.strip()
            if not tok: continue
            k, v = tok.split("=", 1)
            w[k.strip()] = float(v.strip())
    return {c: float(w.get(c, 1.0)) for c in columns}


def build_cost_matrix(df_orig, df_gen, columns, weights):
    n = len(df_orig)
    if n != len(df_gen):
        raise ValueError(f"Sizes must match: {n} vs {len(df_gen)}")
    W = np.array([weights[c] for c in columns], dtype=float)
    O, G = df_orig[columns].to_numpy(dtype=object), df_gen[columns].to_numpy(dtype=object)
    C = np.zeros((n, n), dtype=float)
    for k, _ in enumerate(columns):
        C += W[k] * (O[:, k][:, None] != G[:, k][None, :]).astype(float)
    return C / W.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out-matched", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    new_df = read_table_robust(args.data)
    reg = load_registry(args.registry)
    columns = reg["columns"]
    root = reg["tree"]["root"]
    directed_edges = [(e["parent"], e["child"]) for e in reg["tree"]["directed_edges"]]
    base_dir = args.base_dir or os.path.dirname(os.path.abspath(args.registry))
    spaces = {v: reg["variables"][v]["values"] for v in columns}

    missing = [c for c in columns if c not in new_df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    new_df = new_df[columns].dropna().astype(str)

    chosen = {"root": None, "conditionals": {}}

    root_factor = next(iter_factors(reg, "root"))
    root_samples_df, root_order = load_factor_samples(root_factor, base_dir)
    emp = empirical_distribution(new_df[root], root_order)
    probs, _ = choose_factor_row(root_samples_df, emp, args.k, rng)
    chosen["root"] = {"values": root_order, "probs": probs.tolist()}

    for f in iter_factors(reg, "conditional_row"):
        ch = f["name"].split("(")[1].split("|")[0].strip()
        pa = f["condition"]["parent"]
        pval = f["condition"]["value"]
        df_s, ch_order = load_factor_samples(f, base_dir)
        emp_c = empirical_conditional(new_df, ch, pa, pval, ch_order)
        probs, _ = choose_factor_row(df_s, emp_c, args.k, rng)
        chosen["conditionals"][(ch, pa, pval)] = {"values": ch_order, "probs": probs.tolist()}

    order, parents = topological_order(root, directed_edges)
    n = len(new_df)
    root_vals = chosen["root"]["values"]
    root_probs = np.array(chosen["root"]["probs"], dtype=float)

    def child_probs(ch, pa, pval):
        meta = chosen["conditionals"].get((ch, pa, pval))
        if meta is None:
            return spaces[ch], np.ones(len(spaces[ch])) / len(spaces[ch])
        return meta["values"], np.array(meta["probs"], dtype=float)

    rows = []
    for _ in range(n):
        a = {root: sample_from_categorical(root_vals, root_probs, rng)}
        for var in order:
            if var == root: continue
            pa = parents[var]; pval = a[pa]
            vals, p = child_probs(var, pa, pval)
            a[var] = sample_from_categorical(vals, p, rng)
        rows.append([a[c] for c in columns])
    out_df = pd.DataFrame(rows, columns=columns)

    weights = parse_weights(args.weights, columns)
    C = build_cost_matrix(new_df[columns].astype(str).reset_index(drop=True),
                          out_df[columns].astype(str).reset_index(drop=True),
                          columns, weights)
    row_ind, col_ind = linear_sum_assignment(C)
    reordered = out_df.iloc[col_ind].reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_matched)), exist_ok=True)
    reordered.to_csv(args.out_matched, index=False)
    print(f"Wrote {args.out_matched}  (avg cost {C[row_ind, col_ind].mean():.4f})")


if __name__ == "__main__":
    main()
