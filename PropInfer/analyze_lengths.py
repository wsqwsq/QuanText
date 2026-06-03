"""Token-length stats for the PropInfer CSVs, using the Llama tokenizer.

Use this to tune --max-completion-tokens / --max-len. As a rule of thumb,
pick the p90-p95 token count of the text column.
"""

import argparse
import os

import pandas as pd
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--model",    default="meta-llama/Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    for name in sorted(os.listdir(args.data_dir)):
        if not name.endswith(".csv"): continue
        path = os.path.join(args.data_dir, name)
        df = pd.read_csv(path)
        if args.text_col not in df.columns: continue
        lens = df[args.text_col].astype(str).map(
            lambda t: len(tok.encode(t, add_special_tokens=False)))
        q = lens.quantile
        print(f"{name:25s}  n={len(df):5d}  "
              f"mean={lens.mean():7.0f}  p50={q(0.5):.0f}  "
              f"p90={q(0.9):.0f}  p95={q(0.95):.0f}  "
              f"p99={q(0.99):.0f}  max={lens.max()}")


if __name__ == "__main__":
    main()
