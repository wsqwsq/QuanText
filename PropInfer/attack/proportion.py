"""Proportion-inference attack report for a single attribute (gender or diagnosis)."""

import argparse
import json
import os
from collections import Counter

import pandas as pd


def proportions(df, col):
    vals = df[col].fillna("").astype(str).map(str.strip).tolist()
    n = len(vals)
    cnt = Counter(vals)
    return {k: cnt[k] / n for k in sorted(cnt)}


def diff_and_ratio(priv, syn):
    out = {}
    for cat in sorted(set(priv) | set(syn)):
        p, s = priv.get(cat, 0.0), syn.get(cat, 0.0)
        ad = abs(p - s)
        out[cat] = {
            "private": p, "synthetic": s,
            "abs_diff": ad,
            "diff_ratio": (ad / p) if p > 0 else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priv",          required=True, help="Private CSV (with true attribute).")
    ap.add_argument("--priv-labeled",  default=None,  help="Private CSV re-labeled by LLM (sanity).")
    ap.add_argument("--syn-labeled",   required=True, help="Synthetic CSV labeled by LLM (attack).")
    ap.add_argument("--attribute-col", required=True)
    ap.add_argument("--output",        required=True)
    args = ap.parse_args()

    priv_df = pd.read_csv(args.priv)
    priv_props = proportions(priv_df, args.attribute_col)

    output = {"private_proportions": priv_props, "comparisons": {}}
    sources = []
    if args.priv_labeled: sources.append(("private_labeled", args.priv_labeled))
    sources.append(("synthetic", args.syn_labeled))

    for name, path in sources:
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found"); continue
        df = pd.read_csv(path)
        sp = proportions(df, args.attribute_col)
        output["comparisons"][name] = diff_and_ratio(priv_props, sp)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
