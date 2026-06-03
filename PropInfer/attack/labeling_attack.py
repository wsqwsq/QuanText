"""LLM-based labeling attack: label each text in --input with a single
attribute (gender or diagnosis) and write a CSV with (text, attribute)."""

import argparse
import copy
import csv
import json
import os
import random
import sys

import pandas as pd

csv.field_size_limit(sys.maxsize)

from pe.api.api import API
from pe.llm import HuggingfaceLLM, Request


CATEGORIES = {
    "gender":    ["female", "male"],
    "diagnosis": ["Digestion", "mental disorder", "childbirth", "others"],
}


class _Labeler(API):
    def __init__(self, llm, prompt_file):
        super().__init__()
        self._llm = llm
        with open(prompt_file, "r") as f:
            self._prompt = json.load(f)

    def random_api(self): return
    def variation_api(self): return

    def _construct_prompt(self, variables):
        msgs = copy.deepcopy(self._prompt["message_template"])
        for m in msgs:
            m["content"] = m["content"].format(**variables)
        return msgs

    def label(self, texts):
        reqs = [Request(messages=self._construct_prompt({"text": t})) for t in texts]
        return self._llm.get_responses(reqs)


def normalize_label(raw, categories, default):
    s = str(raw).strip().lower()
    # Longest-match first to avoid 'mental' matching before 'mental disorder'.
    for cat in sorted(categories, key=len, reverse=True):
        if cat.lower() in s:
            return cat
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",         required=True, help="CSV with text column.")
    ap.add_argument("--output",        required=True, help="Output CSV with text + attribute.")
    ap.add_argument("--text-col",      default="text")
    ap.add_argument("--attribute-col", required=True, choices=list(CATEGORIES))
    ap.add_argument("--prompt",        required=True, help="Labeling prompt JSON.")
    ap.add_argument("--model",         default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-tokens",    type=int, default=20)
    ap.add_argument("--batch-size",    type=int, default=8)
    args = ap.parse_args()

    if os.path.exists(args.output):
        print(f"{args.output} already exists, skipping.")
        return

    cats = CATEGORIES[args.attribute_col]
    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"--text-col '{args.text_col}' missing from {args.input}")
    texts = df[args.text_col].fillna("").astype(str).map(str.strip).tolist()

    llm = HuggingfaceLLM(
        max_completion_tokens=args.max_tokens,
        batch_size=args.batch_size,
        model_name_or_path=args.model,
        temperature=0.5,
    )
    raw = _Labeler(llm, args.prompt).label(texts)
    labels = [normalize_label(r, cats, default=random.choice(cats)) for r in raw]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pd.DataFrame({args.text_col: texts, args.attribute_col: labels}).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
