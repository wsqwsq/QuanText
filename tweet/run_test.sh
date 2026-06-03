#!/usr/bin/env bash
# Quick smoke test: runs the full pipeline on data/tweet_tmp.csv (~7 rows)
# with one hyperparameter setting. Output goes under results_test/.

set -euo pipefail

cd "$(dirname "$0")"

GAMMA=20         # small candidate pool — tiny dataset
K=5
DUPLICATE=1      # skip the judge step to keep the test fast
WEIGHTS="Target=2,Stance=1,Sentiment=1"

DATA="data/tweet_tmp.csv"
RESULTS="results_test"
BINS="${RESULTS}/tweet_bins_gamma${GAMMA}"
PRIV_LABELED="${RESULTS}/tweet_tmp_labeled.csv"
OUT="${RESULTS}/gamma${GAMMA}_k${K}"

rm -rf "$RESULTS"
mkdir -p "${OUT}/attack"

echo "===== [TEST] Chow-Liu (gamma=${GAMMA}) ====="
python chow_liu.py --data "$DATA" --gamma "$GAMMA" --pool-size 200 --outdir "$BINS"

echo "===== [TEST] Quantization (k=${K}) ====="
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

echo "===== [TEST] Tweet generation (duplicate=${DUPLICATE}) ====="
python tweet_generation.py \
    --ori "$DATA" \
    --snippets "${OUT}/snippets.csv" \
    --rewrite-attrs "${OUT}/new_attributes.csv" \
    --save "${OUT}/rewritten_tweets.csv" \
    --duplicate "$DUPLICATE"

echo "===== [TEST] Labeling attack (private) ====="
python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED"

echo "===== [TEST] Labeling attack (synthetic) ====="
python attack/labeling_attack.py \
    --input "${OUT}/rewritten_tweets.csv" \
    --output "${OUT}/rewritten_tweets_labeled.csv"

echo "===== [TEST] Proportion analysis ====="
python attack/proportion.py \
    --priv "$DATA" \
    --priv-labeled "$PRIV_LABELED" \
    --syn-labeled "${OUT}/rewritten_tweets_labeled.csv" \
    --output "${OUT}/attack/labeling_attack.json"

echo "===== [TEST] Evaluation ====="
python evaluation/evaluate.py \
    --priv "$DATA" \
    --gen  "${OUT}/rewritten_tweets.csv" \
    --attr "${OUT}/new_attributes.csv" \
    --output "${OUT}/evaluation.json"

echo ""
echo "===== Test pipeline done → ${OUT} ====="
ls -la "$OUT"
