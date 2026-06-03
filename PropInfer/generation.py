"""SML generation step for PropInfer: snippet extraction + rewrite + judge."""

import argparse
import copy
import csv
import json
import os
import random
import re
import sys

import pandas as pd

csv.field_size_limit(sys.maxsize)

from pe.api.api import API
from pe.llm import HuggingfaceLLM, Request


TEMP = 0.5


class _Worker(API):
    def __init__(self, llm, prompt_file):
        super().__init__()
        self._llm = llm
        with open(prompt_file, "r") as f:
            self._prompt = json.load(f)

    def random_api(self): return
    def variation_api(self): return

    def _construct_prompt(self, prompt_config, variables):
        if "replacement_rules" in prompt_config:
            for rule in prompt_config["replacement_rules"]:
                if all(variables.get(k) == v for k, v in rule["constraints"].items()):
                    for k, v in rule["replacements"].items():
                        if isinstance(v, list):
                            v = random.choice(v)
                        variables[k] = v
        msgs = copy.deepcopy(prompt_config["message_template"])
        for m in msgs:
            m["content"] = m["content"].format(**variables)
        return msgs

    def snippet_extraction(self, samples, attr_values):
        msgs = [
            self._construct_prompt(self._prompt,
                {"sample": samples[i], "attribute_value": attr_values[i]})
            for i in range(len(samples))
        ]
        reqs = [Request(messages=m) for m in msgs]
        return self._llm.get_responses(reqs)

    def rewrite(self, snippets, rewrite_attr_values, duplicate=1):
        n = len(snippets)
        msgs = [
            self._construct_prompt(self._prompt, {
                "snippet": snippets[i // duplicate],
                "rewrite_attribute_value": rewrite_attr_values[i // duplicate],
            })
            for i in range(n * duplicate)
        ]
        reqs = [Request(messages=m) for m in msgs]
        return self._llm.get_responses(reqs)

    def _construct_samples(self, samples):
        return "".join(f"Sample {i+1}:\n{s.strip()}\n\n" for i, s in enumerate(samples))

    def _select(self, samples, judgement):
        for i in range(len(samples)):
            if f"{i+1}" in judgement:
                return samples[i]
        return random.choice(samples)

    def judge(self, attr_values, samples, duplicate):
        n = len(samples) // duplicate
        msgs = [
            self._construct_prompt(self._prompt, {
                "attribute_value": attr_values[i],
                "sample": self._construct_samples(samples[i * duplicate:(i + 1) * duplicate]),
            })
            for i in range(n)
        ]
        reqs = [Request(messages=m) for m in msgs]
        judgements = self._llm.get_responses(reqs)
        return [self._select(samples[i * duplicate:(i + 1) * duplicate], judgements[i])
                for i in range(n)]


def _extract_after_step(text, step):
    m = re.search(rf"(?i)Step\s*{step}\s*:\s*(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _clean_lines(text):
    lines = [ln for ln in str(text).splitlines() if ln.strip() and ln.strip() != "###"]
    return re.sub(r"\s*#{3}\s*$", "", "\n".join(lines)).rstrip()


def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as wf:
        w = csv.writer(wf)
        w.writerow([header])
        for r in rows:
            w.writerow([r])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",           required=True, help="Private CSV (text + attribute column).")
    ap.add_argument("--rewrite-attrs",  required=True, help="CSV from quantization.py (attribute column only).")
    ap.add_argument("--snippets-out",   required=True)
    ap.add_argument("--save",           required=True, help="Final rewritten CSV.")
    ap.add_argument("--text-col",       default="text")
    ap.add_argument("--attribute-col",  required=True, help="e.g. 'gender' or 'diagnosis'.")
    ap.add_argument("--prompts-dir",    required=True, help="Directory with snippet_extraction.json, rewrite.json, judge.json.")
    ap.add_argument("--duplicate",      type=int, default=3)
    ap.add_argument("--model",          default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-tokens",     type=int, default=1024,
                    help="ChatDoctor dialogues are ~400-3000 tokens; 1024 covers ~p85.")
    ap.add_argument("--batch-size",     type=int, default=8)
    args = ap.parse_args()

    llm = HuggingfaceLLM(
        max_completion_tokens=args.max_tokens,
        batch_size=args.batch_size,
        model_name_or_path=args.model,
        temperature=TEMP,
    )

    # 1) Snippet extraction
    df = pd.read_csv(args.data, sep=None, engine="python", encoding="utf-8-sig")
    samples = df[args.text_col].astype(str).tolist()
    attr_values = df[args.attribute_col].astype(str).tolist()

    extractor = _Worker(llm, os.path.join(args.prompts_dir, "snippet_extraction.json"))
    raw_snippets = extractor.snippet_extraction(samples, attr_values)

    _write_csv(args.snippets_out.replace(".csv", "_whole.csv"), "Snippet", raw_snippets)
    snippets = [_extract_after_step(s, 3) for s in raw_snippets]
    snippets = [_clean_lines(s) for s in snippets]
    _write_csv(args.snippets_out, "Snippet", snippets)

    # 2) Rewrite — read with default comma sep (sep=None sniffer fails on
    # single-column CSVs since there is no delimiter to detect).
    rewrite_df = pd.read_csv(args.rewrite_attrs, encoding="utf-8-sig")
    if args.attribute_col not in rewrite_df.columns:
        raise ValueError(
            f"--rewrite-attrs is missing column '{args.attribute_col}' "
            f"(got columns: {list(rewrite_df.columns)})"
        )
    rewrite_attrs = rewrite_df[args.attribute_col].astype(str).tolist()

    rewriter = _Worker(llm, os.path.join(args.prompts_dir, "rewrite.json"))
    raw_rewrites = rewriter.rewrite(snippets, rewrite_attrs, duplicate=args.duplicate)

    interim_path = args.save if args.duplicate == 1 \
        else args.save.replace(".csv", f"_dup{args.duplicate}.csv")
    _write_csv(interim_path.replace(".csv", "_whole.csv"), "text", raw_rewrites)
    texts = [_clean_lines(_extract_after_step(s, 2)) for s in raw_rewrites]
    _write_csv(interim_path, "text", texts)

    # 3) Judge (only if duplicate > 1)
    if args.duplicate > 1:
        judge = _Worker(llm, os.path.join(args.prompts_dir, "judge.json"))
        judged = judge.judge(rewrite_attrs, texts, args.duplicate)
        _write_csv(args.save, "text", judged)
        texts = judged

    # Also pair generated text with the rewrite-attribute used for it.
    paired_path = args.save.replace(".csv", "_with_attr.csv")
    n = min(len(texts), len(rewrite_attrs))
    out_df = pd.DataFrame({"text": texts[:n], args.attribute_col: rewrite_attrs[:n]})
    out_df.to_csv(paired_path, index=False)
    print(f"Saved {args.save}  +  {paired_path}")


if __name__ == "__main__":
    main()
