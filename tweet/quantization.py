#!/usr/bin/env python3
# analyze_generate_and_match.py
# End-to-end:
# 1) Read NEW dataset + model_registry.json
# 2) For each factor, pick a random distribution from the closest-k (by TV distance) to the new data
# 3) Sample n assignments from the Chow–Liu tree using those chosen factors
# 4) Reorder generated rows to best match original rows via weighted Hamming + Hungarian (SciPy)
#
# Example:
# python quantization.py \
#   --data tweet.csv \
#   --registry tweet_bins/model_registry.json \
#   --base-dir tweet_bins \
#   --k 30 \
#   --seed 42 \
#   --out synthetic_assignments.csv \
#   --match \
#   --weights Target=2,Stance=1,Sentiment=1 \
#   --normalize-cost \
#   --out-matched new_attributes.csv

import argparse
import json
import math
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ---------------- I/O helpers ----------------
def read_table_robust(path):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, dtype=str)
        except Exception as e:
            last_err = e
    raise last_err

def load_registry(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_factors(registry, kind=None):
    for f in registry["factors"]:
        if (kind is None) or (f.get("type") == kind):
            yield f

def load_factor_samples(factor, base_dir=None):
    values = factor["values"]
    fpath = factor["file"]
    if base_dir and not os.path.isabs(fpath):
        fpath = os.path.join(base_dir, fpath)
    df = pd.read_csv(fpath)
    # reorder columns to the values order stored in registry
    df = df[[v for v in values if v in df.columns]]
    return df, values

# ---------------- distances, empirical distributions ----------------
def tv_distance(p, q, eps=1e-12):
    """Total variation: TV(p,q) = 0.5 * ||p-q||_1."""
    p = np.clip(np.asarray(p, dtype=float), eps, np.inf)
    q = np.clip(np.asarray(q, dtype=float), eps, np.inf)
    p /= p.sum()
    q /= q.sum()
    return 0.5 * np.abs(p - q).sum()

def empirical_distribution(series, values_order):
    """Return probs aligned with values_order; uniform if no data found."""
    s = series.astype(str)
    counts = s.value_counts()
    arr = np.array([counts.get(v, 0) for v in values_order], dtype=float)
    if arr.sum() == 0:
        return np.ones(len(values_order)) / len(values_order)
    return arr / arr.sum()

def empirical_conditional(df, child, parent, parent_value, child_values_order):
    mask = (df[parent].astype(str) == str(parent_value))
    if mask.sum() == 0:
        # No rows for this parent value in the new dataset → uniform
        return np.ones(len(child_values_order)) / len(child_values_order)
    return empirical_distribution(df.loc[mask, child], child_values_order)

# ---------------- choose closest k then pick one at random ----------------
def choose_factor_row(samples_df, target_probs, k=5, rng=None):
    """
    samples_df: K x m DataFrame (each row sums to 1), columns ordered as in target_probs
    target_probs: 1D numpy array aligned to samples_df columns
    Returns (chosen_row_probs, chosen_index, topk_indices_sorted_by_tv, tv_values_for_topk)
    """
    rng = np.random.default_rng() if rng is None else rng
    tvs = samples_df.apply(lambda row: tv_distance(target_probs, row.values), axis=1).values
    k = min(k, len(tvs))
    topk_idx = np.argpartition(tvs, kth=k-1)[:k]
    topk_sorted = topk_idx[np.argsort(tvs[topk_idx])]
    chosen = rng.choice(topk_sorted)
    return samples_df.iloc[chosen].values, int(chosen), topk_sorted.tolist(), tvs[topk_sorted].tolist()

# ---------------- Chow–Liu sampling ----------------
def topological_order(root, directed_edges):
    """Return a topological order (BFS from root) and parents map."""
    adj = defaultdict(list)
    for (p, c) in directed_edges:
        adj[p].append(c)
    order = []
    q = deque([root])
    seen = {root}
    parents = {root: None}
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                parents[v] = u
                q.append(v)
    return order, parents

def sample_from_categorical(values, probs, rng):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return rng.choice(values, p=probs)

# ---------------- matching: weighted Hamming + Hungarian (SciPy) ----------------
def parse_weights(weights_arg, columns):
    """
    weights_arg:
      - None -> all 1.0
      - "A=2,B=1.5" inline string -> dict
      - path to JSON file -> dict
    Any missing column defaults to 1.0
    """
    if weights_arg is None:
        return {c: 1.0 for c in columns}
    if os.path.isfile(weights_arg):
        with open(weights_arg, "r", encoding="utf-8") as f:
            w = json.load(f)
    else:
        w = {}
        for tok in weights_arg.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "=" not in tok:
                raise ValueError(f"Bad weight token '{tok}', expected name=weight")
            k, v = tok.split("=", 1)
            w[k.strip()] = float(v.strip())
    return {c: float(w.get(c, 1.0)) for c in columns}

def build_cost_matrix_weighted_hamming(df_orig, df_gen, columns, weights, normalize=False):
    """
    cost[i,j] = sum_c w[c] * 1[orig[i,c] != gen[j,c]]
    If normalize=True, divide by sum_w so costs lie in [0,1].
    """
    n = len(df_orig)
    if n != len(df_gen):
        raise ValueError(f"Sizes must match for matching: {n} vs {len(df_gen)}")
    W = np.array([weights[c] for c in columns], dtype=float)
    denom = W.sum() if normalize else 1.0

    O = df_orig[columns].to_numpy(dtype=object)  # (n, d)
    G = df_gen[columns].to_numpy(dtype=object)   # (n, d)

    C = np.zeros((n, n), dtype=float)
    for k, _c in enumerate(columns):
        oc = O[:, k]                 # (n,)
        gc = G[:, k]                 # (n,)
        mism = (oc[:, None] != gc[None, :]).astype(float)  # (n, n)
        C += W[k] * mism
    C /= denom
    return C

def solve_assignment(cost):
    """Hungarian algorithm (SciPy). Returns row_ind, col_ind, total_cost."""
    row_ind, col_ind = linear_sum_assignment(cost)
    total = float(cost[row_ind, col_ind].sum())
    return row_ind, col_ind, total

# ---------------- main pipeline ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default='tweet.csv', help="Path to NEW dataset (CSV/TSV).")
    ap.add_argument("--registry", default='tweet_bins/model_registry.json', help="Path to model_registry.json.")
    ap.add_argument("--base-dir", default=None, help="Base dir where sample CSVs are stored (defaults to registry's parent).")
    ap.add_argument("--k", type=int, default=30, help="Choose randomly from the closest k distributions (TV).")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    #ap.add_argument("--out", default="new_attributes.csv", help="Output CSV for generated assignments.")

    # Matching options
    ap.add_argument("--match", action="store_true",
                    help="If set, reorder generated rows to best match originals by weighted distance.")
    ap.add_argument("--weights", default=None,
                    help='Per-attribute weights as "A=2,B=1.5" or path to JSON. Missing columns default to 1.0.')
    ap.add_argument("--normalize-cost", action="store_true",
                    help="Normalize pairwise costs by sum of weights to keep them in [0,1].")
    ap.add_argument("--out-matched", default="new_attributes.csv",
                    help="Output path for the reordered (matched) generated arrays.")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # 1) Read new dataset
    new_df = read_table_robust(args.data)

    # 2) Load registry (tree + metadata)
    reg = load_registry(args.registry)
    columns = reg["columns"]
    root = reg["tree"]["root"]
    directed_edges = [(e["parent"], e["child"]) for e in reg["tree"]["directed_edges"]]

    missing = [c for c in columns if c not in new_df.columns]
    if missing:
        raise ValueError(f"New dataset missing required columns: {missing}")
    new_df = new_df[columns].dropna().astype(str)

    # Find base_dir default if not provided
    base_dir = args.base_dir
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # Canonical value spaces from registry
    spaces = {v: reg["variables"][v]["values"] for v in columns}

    # 3) For each factor, compute empirical target on new data,
    #    compute TV to each constructed distribution, take closest k, pick 1 randomly.
    chosen = {"root": None, "conditionals": {}}

    # Root
    root_factor = next(iter_factors(reg, kind="root"))
    root_samples_df, root_order = load_factor_samples(root_factor, base_dir=base_dir)
    emp_root = empirical_distribution(new_df[root], root_order)
    root_probs, root_idx, root_topk_idx, root_topk_tvs = choose_factor_row(root_samples_df, emp_root, k=args.k, rng=rng)
    chosen["root"] = {
        "var": root,
        "values": root_order,
        "probs": root_probs.tolist(),
        "chosen_row_index": root_idx,
        "topk_indices": root_topk_idx,
        "topk_tvs": root_topk_tvs
    }

    # Conditionals
    for f in iter_factors(reg, kind="conditional_row"):
        ch = f["name"].split("(")[1].split("|")[0].strip()       # e.g., Sentiment
        pa = f["condition"]["parent"]
        pval = f["condition"]["value"]
        samples_df, ch_order = load_factor_samples(f, base_dir=base_dir)

        emp_ch_given = empirical_conditional(new_df, ch, pa, pval, ch_order)
        ch_probs, ch_idx, ch_topk_idx, ch_topk_tvs = choose_factor_row(samples_df, emp_ch_given, k=args.k, rng=rng)

        chosen["conditionals"][(ch, pa, pval)] = {
            "child": ch,
            "parent": pa,
            "parent_value": pval,
            "values": ch_order,
            "probs": ch_probs.tolist(),
            "chosen_row_index": ch_idx,
            "topk_indices": ch_topk_idx,
            "topk_tvs": ch_topk_tvs
        }

    # 4) Construct joint sampler per Chow–Liu and generate n assignments
    order, parents = topological_order(root, directed_edges)
    n = len(new_df)

    root_values = chosen["root"]["values"]
    root_probs = np.array(chosen["root"]["probs"], dtype=float)

    def get_child_probs(ch, pa, pval):
        key = (ch, pa, pval)
        meta = chosen["conditionals"].get(key)
        if meta is None:
            vals = spaces[ch]
            return vals, np.ones(len(vals)) / len(vals)
        return meta["values"], np.array(meta["probs"], dtype=float)

    out_rows = []
    for _ in range(n):
        assignment = {}
        # sample root
        r_val = sample_from_categorical(root_values, root_probs, rng)
        assignment[root] = r_val
        # sample others
        for var in order:
            if var == root:
                continue
            pa = parents[var]
            pval = assignment[pa]
            vals, probs = get_child_probs(var, pa, pval)
            c_val = sample_from_categorical(vals, probs, rng)
            assignment[var] = c_val
        out_rows.append([assignment[c] for c in columns])

    out_df = pd.DataFrame(out_rows, columns=columns)
    #out_df.to_csv(args.out, index=False)
    #print(f"Generated {len(out_df)} assignments → {args.out}")

    ## Also save which rows (indices) were chosen, for reproducibility/analysis
    #chosen_path = os.path.splitext(args.out)[0] + "_chosen.json"
    #cond_chosen = {f"{ch}|{pa}={pval}": v for (ch, pa, pval), v in chosen["conditionals"].items()}
    #with open(chosen_path, "w", encoding="utf-8") as f:
    #    json.dump({"root": chosen["root"], "conditionals": cond_chosen}, f, ensure_ascii=False, indent=2)
    #print(f"Saved chosen factor rows (metadata) → {chosen_path}")

    # ----- Reorder (match) generated rows to originals -----
    if args.match:
        orig_for_match = new_df[columns].astype(str).reset_index(drop=True)
        gen_for_match  = out_df[columns].astype(str).reset_index(drop=True)

        weights = parse_weights(args.weights, columns)
        C = build_cost_matrix_weighted_hamming(
            orig_for_match, gen_for_match, columns, weights, normalize=args.normalize_cost
        )

        row_ind, col_ind, total_cost = solve_assignment(C)
        avg_cost = total_cost / len(row_ind)

        # Reorder generated rows to align with original order
        gen_reordered = out_df.iloc[col_ind].reset_index(drop=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_matched)), exist_ok=True)
        gen_reordered.to_csv(args.out_matched, index=False)

        # Save mapping too
        map_path = os.path.splitext(args.out_matched)[0] + "_mapping.csv"
        pd.DataFrame({
            "original_index": row_ind,
            "generated_index": col_ind,
            "pair_cost": C[row_ind, col_ind]
        }).to_csv(map_path, index=False)

        print(f"Matched and saved reordered generated arrays → {args.out_matched}")
        print(f"Saved index mapping → {map_path}")
        print(f"Total matching cost = {total_cost:.6f}  |  Average cost = {avg_cost:.6f}")

if __name__ == "__main__":
    main()
