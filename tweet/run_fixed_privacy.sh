#!/usr/bin/env bash
# End-to-end pipeline sweeping (GAMMA, k) pairs with γ/k held constant
# (γ/k = 4 → fixed SML privacy bound), so each run varies utility under
# the same privacy guarantee.
#
# Results layout:
#   results/
#   ├── tweet_labeled.csv                          # shared (LLM-relabeled private data)
#   ├── tweet_bins_gamma${GAMMA}/                  # per-GAMMA candidates
#   └── gamma${GAMMA}_k${k}/
#       ├── new_attributes.csv
#       ├── snippets.csv  +  snippets_whole.csv
#       ├── rewritten_tweets.csv  (+ _dup3*.csv intermediates)
#       ├── rewritten_tweets_labeled.csv
#       ├── evaluation.json
#       └── attack/labeling_attack.json

set -euo pipefail

cd "$(dirname "$0")"

# (GAMMA, k) pairs — all share γ/k = 4
PAIRS=("40 10" "80 20" "120 30" "160 40" "200 50")

DUPLICATE=3
WEIGHTS="Target=2,Stance=1,Sentiment=1"

DATA="data/tweet.csv"
RESULTS="results"
PRIV_LABELED="${RESULTS}/tweet_labeled.csv"

mkdir -p "$RESULTS"

# ───────────────────────────────────────────────────────────────────────────
# Determine the run suffix so the script can be invoked multiple times.
# Usage:
#   ./run_fixed_privacy.sh           → auto-detect smallest free index
#   ./run_fixed_privacy.sh 3         → force suffix "_3" (no suffix if =1)
#
# Auto-detect rule: pick the smallest N ≥ 1 such that none of the per-pair
# output dirs gamma${G}_k${K}     (when N=1)
#  or         gamma${G}_k${K}_${N} (when N>1)
# exist yet. With N=1 the suffix is empty so the first run is backward-
# compatible with prior layouts.
# ───────────────────────────────────────────────────────────────────────────
suffix_for() {           # $1: index N → "" if 1, "_N" otherwise
    [ "$1" -eq 1 ] && echo "" || echo "_$1"
}

if [ "${1-}" != "" ]; then
    RUN_ID="$1"
else
    RUN_ID=1
    while :; do
        sx=$(suffix_for "$RUN_ID")
        free=1
        for pair in "${PAIRS[@]}"; do
            read -r G K <<< "$pair"
            if [ -d "${RESULTS}/gamma${G}_k${K}${sx}" ]; then
                free=0; break
            fi
        done
        [ "$free" -eq 1 ] && break
        RUN_ID=$((RUN_ID + 1))
    done
fi
RUN_SX=$(suffix_for "$RUN_ID")
echo "===== Run index: ${RUN_ID}   (folder suffix: '${RUN_SX}') ====="

# ───────────────────────────────────────────────────────────────────────────
# Step 0: Label the private dataset once (shared across all runs)
# ───────────────────────────────────────────────────────────────────────────
if [ ! -f "$PRIV_LABELED" ]; then
    echo "===== Labeling private dataset (shared) ====="
    python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED"
else
    echo "===== Private labeled file exists at ${PRIV_LABELED}, skipping ====="
fi

# ───────────────────────────────────────────────────────────────────────────
# Sweep over (GAMMA, k) pairs
# ───────────────────────────────────────────────────────────────────────────
for pair in "${PAIRS[@]}"; do
    read -r GAMMA K <<< "$pair"
    BINS="${RESULTS}/tweet_bins_gamma${GAMMA}"   # bins cached across runs (depend only on GAMMA)
    OUT="${RESULTS}/gamma${GAMMA}_k${K}${RUN_SX}"
    mkdir -p "${OUT}/attack"

    echo ""
    echo "##################################################################"
    echo "##  GAMMA=${GAMMA}  k=${K}   (γ/k = $((GAMMA / K)))   →  ${OUT}"
    echo "##################################################################"

    # 1) Chow-Liu candidate distributions for this GAMMA (cached across k if reused)
    if [ ! -f "${BINS}/model_registry.json" ]; then
        echo "----- [GAMMA=${GAMMA}] Chow-Liu candidates -----"
        python chow_liu.py --data "$DATA" --gamma "$GAMMA" --outdir "$BINS"
    else
        echo "----- Chow-Liu bins already exist at ${BINS}, skipping -----"
    fi

    # 2) Randomized quantization → property arrays
    echo "----- [k=${K}] Quantization -----"
    python quantization.py \
        --data "$DATA" \
        --registry "${BINS}/model_registry.json" \
        --base-dir "$BINS" \
        --k "$K" \
        --seed 42 \
        --match \
        --weights "$WEIGHTS" \
        --normalize-cost \
        --out-matched "${OUT}/new_attributes.csv"

    # 3) Snippet extraction + LLM rewrite (with duplicate=3 + judge)
    echo "----- [k=${K}] Tweet generation -----"
    python tweet_generation.py \
        --ori "$DATA" \
        --snippets "${OUT}/snippets.csv" \
        --rewrite-attrs "${OUT}/new_attributes.csv" \
        --save "${OUT}/rewritten_tweets.csv" \
        --duplicate "$DUPLICATE"

    # 4) Attack: label the synthetic tweets
    echo "----- [k=${K}] Labeling attack on synthetic tweets -----"
    python attack/labeling_attack.py \
        --input "${OUT}/rewritten_tweets.csv" \
        --output "${OUT}/rewritten_tweets_labeled.csv"

    # 5) Attack: proportion / pair-proportion diff & ratio report
    echo "----- [k=${K}] Proportion analysis -----"
    python attack/proportion.py \
        --priv "$DATA" \
        --priv-labeled "$PRIV_LABELED" \
        --syn-labeled "${OUT}/rewritten_tweets_labeled.csv" \
        --output "${OUT}/attack/labeling_attack.json"

    # 6) Utility evaluation: length / precision-recall / FID / attr-TV
    echo "----- [k=${K}] Utility evaluation -----"
    python evaluation/evaluate.py \
        --priv "$DATA" \
        --gen  "${OUT}/rewritten_tweets.csv" \
        --attr "${OUT}/new_attributes.csv" \
        --output "${OUT}/evaluation.json"

    echo "===== [GAMMA=${GAMMA}, k=${K}] Done → ${OUT} ====="
done

echo ""
echo "All (GAMMA, k) sweeps finished. Results in ${RESULTS}/"
