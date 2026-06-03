#!/usr/bin/env bash
# Smoke test for the Private Evolution pipeline.
# Uses data/tweet_tmp.csv (~8 rows, 3 unique labels), 2 iterations, 9 samples.
# Output: results_test_pe/

set -euo pipefail

cd "$(dirname "$0")"

NUM_ITER=2
EPSILON=1.0
NUM_SAMPLES=6          # tweet_tmp.csv has 3 unique (Target,Stance,Sentiment) tuples → ≥3

DATA="data/tweet_tmp.csv"
RESULTS="results_test"
PE_OUT="${RESULTS}/pe"
PRIV_LABELED="${RESULTS}/tweet_tmp_labeled.csv"

FINAL_ITER_STR=$(printf "%09d" "$NUM_ITER")
SYN_CSV="${PE_OUT}/synthetic_text/${FINAL_ITER_STR}.csv"

rm -rf "$PE_OUT"
mkdir -p "$PE_OUT" "${PE_OUT}/attack"

# ───────────────────────────────────────────────────────────────────────────
# Step 1: PE generation
# ───────────────────────────────────────────────────────────────────────────
echo "===== [TEST PE] Generation (iter=${NUM_ITER}, eps=${EPSILON}, n=${NUM_SAMPLES}) ====="
python PE/main.py \
    --data "$DATA" \
    --exp-folder "$PE_OUT" \
    --num-iterations "$NUM_ITER" \
    --num-samples "$NUM_SAMPLES" \
    --epsilon "$EPSILON" \
    --delta 1e-5 \
    --max-completion-tokens 128 \
    --skip-fid

if [ ! -f "$SYN_CSV" ]; then
    echo "ERROR: expected synthetic CSV not found: $SYN_CSV" >&2
    ls -la "${PE_OUT}/synthetic_text" >&2 || true
    exit 1
fi
echo "----- Final synthetic CSV: ${SYN_CSV} -----"

# ───────────────────────────────────────────────────────────────────────────
# Step 2: Evaluation
# ───────────────────────────────────────────────────────────────────────────
echo "===== [TEST PE] Evaluation ====="
python evaluation/evaluate.py \
    --priv "$DATA" \
    --gen  "$SYN_CSV" \
    --attr "$SYN_CSV" \
    --output "${PE_OUT}/evaluation.json"

# ───────────────────────────────────────────────────────────────────────────
# Step 3: Attack
# ───────────────────────────────────────────────────────────────────────────
if [ ! -f "$PRIV_LABELED" ]; then
    echo "===== [TEST PE] Labeling attack (private) ====="
    python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED"
else
    echo "===== Private labeled file exists at ${PRIV_LABELED}, skipping ====="
fi

echo "===== [TEST PE] Labeling attack (synthetic) ====="
python attack/labeling_attack.py \
    --input  "$SYN_CSV" \
    --output "${PE_OUT}/rewritten_tweets_labeled.csv"

echo "===== [TEST PE] Proportion analysis ====="
python attack/proportion.py \
    --priv "$DATA" \
    --priv-labeled "$PRIV_LABELED" \
    --syn-labeled  "${PE_OUT}/rewritten_tweets_labeled.csv" \
    --output "${PE_OUT}/attack/labeling_attack.json"

echo ""
echo "===== Test PE pipeline done → ${PE_OUT} ====="
ls -la "$PE_OUT"
