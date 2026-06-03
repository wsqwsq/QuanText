#!/usr/bin/env bash
# End-to-end pipeline: chow-liu -> quantization -> tweet generation -> attack labeling
# -> attack proportion analysis -> evaluation, sweeping over the top-k hyperparameter.
#
# Results layout:
#   results/
#   ├── tweet_bins_gamma${GAMMA}/        # shared across all k (depends only on GAMMA)
#   ├── tweet_labeled.csv                # shared (LLM-relabeled private data)
#   └── gamma${GAMMA}_k${k}/
#       ├── new_attributes.csv
#       ├── snippets.csv  +  snippets_whole.csv
#       ├── rewritten_tweets.csv  (+ _dup3*.csv intermediates)
#       ├── rewritten_tweets_labeled.csv
#       ├── evaluation.json
#       └── attack/labeling_attack.json

set -euo pipefail

cd "$(dirname "$0")"

GAMMA=120
KS=(10 15 20 30 40 60)
DUPLICATE=3
WEIGHTS="Target=2,Stance=1,Sentiment=1"

DATA="data/tweet.csv"
RESULTS="results"
BINS="${RESULTS}/tweet_bins_gamma${GAMMA}"
PRIV_LABELED="${RESULTS}/tweet_labeled.csv"

mkdir -p "$RESULTS"

# ───────────────────────────────────────────────────────────────────────────
# Step 0: Chow-Liu candidate distributions (shared across all k)
# ───────────────────────────────────────────────────────────────────────────
if [ ! -f "${BINS}/model_registry.json" ]; then
    echo "===== [GAMMA=${GAMMA}] Building Chow-Liu candidates ====="
    python chow_liu.py --data "$DATA" --gamma "$GAMMA" --outdir "$BINS"
else
    echo "===== Chow-Liu bins already exist at ${BINS}, skipping ====="
fi

# ───────────────────────────────────────────────────────────────────────────
# Step 0b: Label the private dataset once (shared baseline for attack)
# ───────────────────────────────────────────────────────────────────────────
if [ ! -f "$PRIV_LABELED" ]; then
    echo "===== Labeling private dataset (shared) ====="
    python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED"
else
    echo "===== Private labeled file exists at ${PRIV_LABELED}, skipping ====="
fi

# ───────────────────────────────────────────────────────────────────────────
# Sweep over k
# ───────────────────────────────────────────────────────────────────────────
for k in "${KS[@]}"; do
    OUT="${RESULTS}/gamma${GAMMA}_k${k}"
    mkdir -p "${OUT}/attack"
    echo ""
    echo "##################################################################"
    echo "##  GAMMA=${GAMMA}  k=${k}   →  ${OUT}"
    echo "##################################################################"

    # 1) Randomized quantization → property arrays
    echo "----- [k=${k}] Quantization -----"
    python quantization.py \
        --data "$DATA" \
        --registry "${BINS}/model_registry.json" \
        --base-dir "$BINS" \
        --k "$k" \
        --seed 42 \
        --match \
        --weights "$WEIGHTS" \
        --normalize-cost \
        --out-matched "${OUT}/new_attributes.csv"

    # 2) Snippet extraction + LLM rewrite (with duplicate=3 + judge)
    echo "----- [k=${k}] Tweet generation -----"
    python tweet_generation.py \
        --ori "$DATA" \
        --snippets "${OUT}/snippets.csv" \
        --rewrite-attrs "${OUT}/new_attributes.csv" \
        --save "${OUT}/rewritten_tweets.csv" \
        --duplicate "$DUPLICATE"

    # 3) Attack: label the synthetic tweets
    echo "----- [k=${k}] Labeling attack on synthetic tweets -----"
    python attack/labeling_attack.py \
        --input "${OUT}/rewritten_tweets.csv" \
        --output "${OUT}/rewritten_tweets_labeled.csv"

    # 4) Attack: proportion / pair-proportion diff & ratio report
    echo "----- [k=${k}] Proportion analysis -----"
    python attack/proportion.py \
        --priv "$DATA" \
        --priv-labeled "$PRIV_LABELED" \
        --syn-labeled "${OUT}/rewritten_tweets_labeled.csv" \
        --output "${OUT}/attack/labeling_attack.json"

    # 5) Utility evaluation: length / precision-recall / FID / attr-TV
    echo "----- [k=${k}] Utility evaluation -----"
    python evaluation/evaluate.py \
        --priv "$DATA" \
        --gen  "${OUT}/rewritten_tweets.csv" \
        --attr "${OUT}/new_attributes.csv" \
        --output "${OUT}/evaluation.json"

    echo "===== [k=${k}] Done → ${OUT} ====="
done

echo ""
echo "All sweeps finished. Results in ${RESULTS}/"
