import csv
import json
import os
import sys
from collections import Counter
from itertools import combinations

import pandas as pd

csv.field_size_limit(sys.maxsize)


current_folder = os.path.dirname(os.path.abspath(__file__))

priv_data_path     = os.path.join(current_folder, '../data/tweet.csv')
priv_labeled_path  = os.path.join(current_folder, '../res/tweet_labeled.csv')
syn_labeled_path   = os.path.join(current_folder, '../res/rewritten_tweets_labeled.csv')
output_path        = os.path.join(current_folder, '../res/attack/labeling_attack.json')

ATTRIBUTES = ['Target', 'Stance', 'Sentiment']


def _column_values(df, col):
    return df[col].fillna('').astype(str).map(lambda x: x.strip()).tolist()


def compute_proportions(df, attributes=ATTRIBUTES):
    """
    Compute proportions for every single attribute and every unordered pair
    of attributes. Pair keys look like 'Target&Sentiment', and category keys
    look like 'Atheism|POSITIVE'.
    """
    proportions = {}

    # singles
    for attr in attributes:
        vals = _column_values(df, attr)
        total = len(vals)
        counts = Counter(vals)
        proportions[attr] = {k: counts[k] / total for k in sorted(counts)}

    # pairs
    for a, b in combinations(attributes, 2):
        va, vb = _column_values(df, a), _column_values(df, b)
        pairs = list(zip(va, vb))
        total = len(pairs)
        counts = Counter(pairs)
        key = f'{a}&{b}'
        proportions[key] = {
            f'{ka}|{kb}': c / total for (ka, kb), c in sorted(counts.items())
        }

    return proportions


def diff_and_ratio(priv_props, syn_props):
    """For each group, report each category's abs diff and ratio vs private."""
    report = {}
    for group, priv_dist in priv_props.items():
        syn_dist = syn_props.get(group, {})
        all_cats = set(priv_dist) | set(syn_dist)
        group_report = {}
        for cat in sorted(all_cats):
            p = priv_dist.get(cat, 0.0)
            s = syn_dist.get(cat, 0.0)
            abs_diff = abs(s - p)
            diff_ratio = (abs_diff / p) if p > 0 else None
            group_report[cat] = {
                'private':    p,
                'synthetic':  s,
                'abs_diff':   abs_diff,
                'diff_ratio': diff_ratio,
            }
        report[group] = group_report
    return report


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--priv',         default=priv_data_path,    help='Private tweet CSV (with true labels).')
    ap.add_argument('--priv-labeled', default=priv_labeled_path, help='Private tweets re-labeled by LLM (sanity).')
    ap.add_argument('--syn-labeled',  default=syn_labeled_path,  help='Synthetic tweets labeled by LLM (attack).')
    ap.add_argument('--output',       default=output_path,       help='Output JSON path.')
    args = ap.parse_args()

    priv_df = pd.read_csv(args.priv)
    priv_props = compute_proportions(priv_df)

    output = {'private_proportions': priv_props, 'comparisons': {}}

    sources = [
        ('private_labeled', args.priv_labeled),
        ('synthetic',       args.syn_labeled),
    ]
    for name, path in sources:
        if not os.path.exists(path):
            print(f'Skipping {name}: {path} not found')
            continue
        df = pd.read_csv(path)
        syn_props = compute_proportions(df)
        output['comparisons'][name] = diff_and_ratio(priv_props, syn_props)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
