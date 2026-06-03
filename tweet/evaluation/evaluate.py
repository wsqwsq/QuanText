import json
import math
import os
import sys
import csv

import numpy as np
import pandas as pd
import torch
from collections import Counter
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm
from numpy import cov, trace, iscomplexobj
from sentence_transformers import SentenceTransformer
from precision_recall import knn_precision_recall_features

csv.field_size_limit(sys.maxsize)

current_folder = os.path.dirname(os.path.abspath(__file__))

gen_path  = current_folder + '/../res/rewritten_tweets.csv'
priv_path = current_folder + '/../data/tweet.csv'
attr_path = current_folder + '/../res/new_attributes.csv'

PRI_EMB = 'pri_emb'
GEN_EMB = 'gen_emb'

cached_embedding = {PRI_EMB: None, GEN_EMB: None}

ATTR_COLS = ['Stance', 'Target', 'Sentiment']


# ── helpers ──────────────────────────────────────────────────────────────────

def get_distance(a, b, tp='wasserstein'):
    """Wasserstein or TV distance between two sample sequences."""
    a_list, b_list = list(a), list(b)
    if not a_list or not b_list:
        return float('inf')

    count_a, count_b = Counter(a_list), Counter(b_list)
    support = sorted(set(count_a.keys()) | set(count_b.keys()))

    pmf_a = np.array([count_a.get(k, 0) for k in support], dtype=float)
    pmf_b = np.array([count_b.get(k, 0) for k in support], dtype=float)
    pmf_a /= pmf_a.sum()
    pmf_b /= pmf_b.sum()

    if tp == 'wasserstein':
        return float(wasserstein_distance(support, support,
                                          u_weights=pmf_a, v_weights=pmf_b))
    return float(0.5 * np.sum(np.abs(pmf_a - pmf_b)))


def extract_features(data, batch_size=100,
                     model_name='stsb-roberta-base-v2', cache=None):
    if cache and cached_embedding[cache] is not None:
        return cached_embedding[cache]

    model = SentenceTransformer(model_name)
    model.eval()

    with torch.no_grad():
        parts = []
        for i in tqdm(range(math.ceil(len(data) / batch_size)),
                      desc='Get Embedding'):
            emb = model.encode(data[i * batch_size:(i + 1) * batch_size])
            if len(emb) > 0:
                parts.append(emb)
    embeddings = np.concatenate(parts) if parts else np.array([])
    del model

    if cache:
        cached_embedding[cache] = embeddings
    return embeddings


def calculate_fid(act1, act2):
    act1 = np.asarray(act1, dtype=np.float64)
    act2 = np.asarray(act2, dtype=np.float64)
    if act1.size == 0 or act2.size == 0:
        return float('inf')
    if act1.ndim == 1:
        act1 = act1.reshape(1, -1)
    if act2.ndim == 1:
        act2 = act2.reshape(1, -1)

    mu1, mu2 = act1.mean(axis=0), act2.mean(axis=0)
    eps = 1e-6
    sigma1 = cov(act1, rowvar=False) + np.eye(act1.shape[1]) * eps
    sigma2 = cov(act2, rowvar=False) + np.eye(act2.shape[1]) * eps

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if iscomplexobj(covmean):
        covmean = covmean.real

    return float(np.real_if_close(ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)))


# ── evaluation functions ──────────────────────────────────────────────────────

def evaluate_length(priv_tweets, gen_tweets):
    """Wasserstein distance between word-count distributions."""
    len1 = [len(t.split()) for t in priv_tweets]
    len2 = [len(t.split()) for t in gen_tweets]
    return {'length_wasserstein': get_distance(len1, len2, tp='wasserstein')}


def evaluate_precision_recall(priv_tweets, gen_tweets):
    """kNN precision and recall on sentence embeddings."""
    fea_pri = extract_features(priv_tweets, cache=PRI_EMB)
    fea_syn = extract_features(gen_tweets,  cache=GEN_EMB)
    state = knn_precision_recall_features(fea_pri, fea_syn, nhood_sizes=[3])
    return {'precision': state['precision'], 'recall': state['recall']}


def evaluate_fid(priv_tweets, gen_tweets):
    """Frechet Inception Distance on sentence embeddings."""
    fea_pri = extract_features(priv_tweets, cache=PRI_EMB)
    fea_syn = extract_features(gen_tweets,  cache=GEN_EMB)
    return {'fid': calculate_fid(fea_pri, fea_syn)}


def evaluate_attr(priv_df, gen_attr_df, attributes=None):
    """
    TV distance between private and generated attribute distributions.

    priv_df     — DataFrame loaded from tweet.csv (has Stance, Opinion towards, Sentiment)
    gen_attr_df — DataFrame loaded from new_attributes.csv (has Target, Stance, Sentiment)
    Attributes missing from either DataFrame are skipped.
    """
    if attributes is None:
        attributes = ATTR_COLS
    results = {}
    for attr in attributes:
        if attr not in priv_df.columns or attr not in gen_attr_df.columns:
            continue
        pri_vals = priv_df[attr].fillna('').astype(str).str.strip().tolist()
        gen_vals = gen_attr_df[attr].fillna('').astype(str).str.strip().tolist()
        results[f'attr_tv_{attr}'] = get_distance(pri_vals, gen_vals, tp='tv')
    return results


# ── I/O ──────────────────────────────────────────────────────────────────────

def read_tweets(path, col='Tweet'):
    df = pd.read_csv(path)
    return df[col].fillna('').astype(str).str.strip().tolist()


def save_results_json(results, save_path):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
        existing.update(results)
        results = existing
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--priv',   default=priv_path, help='Private tweet CSV (with Tweet column).')
    ap.add_argument('--gen',    default=gen_path,  help='Generated tweet CSV (with Tweet column).')
    ap.add_argument('--attr',   default=attr_path, help='Generated attributes CSV (Target/Stance/Sentiment).')
    ap.add_argument('--output', default=os.path.join(current_folder, '../res/evaluation.json'))
    args = ap.parse_args()

    priv_tweets  = read_tweets(args.priv)
    gen_tweets   = read_tweets(args.gen)
    priv_df      = pd.read_csv(args.priv)
    gen_attr_df  = pd.read_csv(args.attr)

    results = {}
    results.update(evaluate_length(priv_tweets, gen_tweets))
    results.update(evaluate_precision_recall(priv_tweets, gen_tweets))
    results.update(evaluate_fid(priv_tweets, gen_tweets))
    results.update(evaluate_attr(priv_df, gen_attr_df))

    save_results_json(results, args.output)
    print(json.dumps(results, indent=4))
