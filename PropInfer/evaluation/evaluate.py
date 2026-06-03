"""Utility evaluation: length, kNN precision/recall, FID, attribute TV.

Generic over text + attribute columns.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from numpy import cov, iscomplexobj, trace
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from precision_recall import knn_precision_recall_features

csv.field_size_limit(sys.maxsize)


PRI_EMB, GEN_EMB = "pri_emb", "gen_emb"
_cache = {PRI_EMB: None, GEN_EMB: None}


def get_distance(a, b, tp="wasserstein"):
    a, b = list(a), list(b)
    if not a or not b:
        return float("inf")
    ca, cb = Counter(a), Counter(b)
    support = sorted(set(ca) | set(cb))
    pa = np.array([ca.get(k, 0) for k in support], dtype=float); pa /= pa.sum()
    pb = np.array([cb.get(k, 0) for k in support], dtype=float); pb /= pb.sum()
    if tp == "wasserstein":
        return float(wasserstein_distance(support, support, u_weights=pa, v_weights=pb))
    return float(0.5 * np.abs(pa - pb).sum())


def extract_features(data, batch_size=32, model_name="stsb-roberta-base-v2", cache=None):
    if cache and _cache[cache] is not None:
        return _cache[cache]
    model = SentenceTransformer(model_name); model.eval()
    parts = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(data) / batch_size)), desc="Embed"):
            emb = model.encode(data[i * batch_size:(i + 1) * batch_size])
            if len(emb): parts.append(emb)
    emb = np.concatenate(parts) if parts else np.array([])
    del model
    if cache: _cache[cache] = emb
    return emb


def calculate_fid(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0: return float("inf")
    if a.ndim == 1: a = a.reshape(1, -1)
    if b.ndim == 1: b = b.reshape(1, -1)
    mu1, mu2 = a.mean(axis=0), b.mean(axis=0)
    s1 = cov(a, rowvar=False) + np.eye(a.shape[1]) * 1e-6
    s2 = cov(b, rowvar=False) + np.eye(b.shape[1]) * 1e-6
    cm = sqrtm(s1.dot(s2))
    if iscomplexobj(cm): cm = cm.real
    return float(np.real_if_close(np.sum((mu1 - mu2) ** 2) + trace(s1 + s2 - 2.0 * cm)))


def evaluate_length(priv, gen):
    l1 = [len(t.split()) for t in priv]
    l2 = [len(t.split()) for t in gen]
    return {"length_wasserstein": get_distance(l1, l2, tp="wasserstein")}


def evaluate_precision_recall(priv, gen):
    fp = extract_features(priv, cache=PRI_EMB)
    fg = extract_features(gen,  cache=GEN_EMB)
    s = knn_precision_recall_features(fp, fg, nhood_sizes=[3])
    return {"precision": s["precision"], "recall": s["recall"]}


def evaluate_fid(priv, gen):
    fp = extract_features(priv, cache=PRI_EMB)
    fg = extract_features(gen,  cache=GEN_EMB)
    return {"fid": calculate_fid(fp, fg)}


def evaluate_attr(priv_df, gen_attr_df, attributes):
    res = {}
    for attr in attributes:
        if attr not in priv_df.columns or attr not in gen_attr_df.columns:
            continue
        pv = priv_df[attr].fillna("").astype(str).str.strip().tolist()
        gv = gen_attr_df[attr].fillna("").astype(str).str.strip().tolist()
        res[f"attr_tv_{attr}"] = get_distance(pv, gv, tp="tv")
    return res


def read_texts(path, col):
    return pd.read_csv(path)[col].fillna("").astype(str).str.strip().tolist()


def save_json(results, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path) as f:
                old = json.load(f)
            old.update(results); results = old
        except json.JSONDecodeError:
            pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priv",   required=True, help="Private CSV.")
    ap.add_argument("--gen",    required=True, help="Generated CSV (text column).")
    ap.add_argument("--attr",   required=True, help="Generated CSV with attribute column(s).")
    ap.add_argument("--text-col",       default="text")
    ap.add_argument("--attribute-cols", required=True,
                    help="Comma-separated, e.g. 'gender' or 'diagnosis'.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    attrs = [c.strip() for c in args.attribute_cols.split(",") if c.strip()]
    priv = read_texts(args.priv, args.text_col)
    gen  = read_texts(args.gen,  args.text_col)
    priv_df     = pd.read_csv(args.priv)
    gen_attr_df = pd.read_csv(args.attr)

    res = {}
    res.update(evaluate_length(priv, gen))
    res.update(evaluate_precision_recall(priv, gen))
    res.update(evaluate_fid(priv, gen))
    res.update(evaluate_attr(priv_df, gen_attr_df, attrs))

    save_json(res, args.output)
    print(json.dumps(res, indent=4))


if __name__ == "__main__":
    main()
