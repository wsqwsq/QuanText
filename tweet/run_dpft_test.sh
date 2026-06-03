#!/usr/bin/env bash
# Smoke test for the DP-LoRA fine-tuning pipeline.
# Uses data/tweet_tmp.csv (8 rows), 1 epoch, generates 6 synthetic tweets.
# Output: results_test/dpft/  (shared with the other test runners)

set -euo pipefail

cd "$(dirname "$0")"

EPSILON=1.0
EPOCHS=1
NUM_SAMPLES=6
BATCH_SIZE=2          # tweet_tmp.csv has only 8 rows → small batch

DATA="data/tweet_tmp.csv"
RESULTS="results_test"
DPFT_OUT="${RESULTS}/dpft"
ADAPTER="${DPFT_OUT}/adapter"
SYN_CSV="${DPFT_OUT}/synthetic.csv"
PRIV_LABELED="${RESULTS}/tweet_tmp_labeled.csv"

rm -rf "$DPFT_OUT"
mkdir -p "${DPFT_OUT}/attack"

# ───────────────────────────────────────────────────────────────────────────
# Step 1: DP fine-tuning
# ───────────────────────────────────────────────────────────────────────────
echo "===== [TEST DPFT] Training (ε=${EPSILON}, epochs=${EPOCHS}) ====="
python DPFT/train.py \
    --data "$DATA" \
    --out  "$ADAPTER" \
    --epsilon "$EPSILON" \
    --delta 1e-5 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-len 128

# ───────────────────────────────────────────────────────────────────────────
# Step 2: Generation
# ───────────────────────────────────────────────────────────────────────────
echo "===== [TEST DPFT] Generating ${NUM_SAMPLES} synthetic tweets ====="
python DPFT/generate.py \
    --adapter "$ADAPTER" \
    --out "$SYN_CSV" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens 64

# ───────────────────────────────────────────────────────────────────────────
# Step 3: Evaluation
# ───────────────────────────────────────────────────────────────────────────
echo "===== [TEST DPFT] Evaluation ====="
python evaluation/evaluate.py \
    --priv "$DATA" \
    --gen  "$SYN_CSV" \
    --attr "$SYN_CSV" \
    --output "${DPFT_OUT}/evaluation.json"

# ───────────────────────────────────────────────────────────────────────────
# Step 4: Attack
# ───────────────────────────────────────────────────────────────────────────
if [ ! -f "$PRIV_LABELED" ]; then
    echo "===== [TEST DPFT] Labeling private dataset ====="
    python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED"
else
    echo "===== Private labeled file exists at ${PRIV_LABELED}, skipping ====="
fi

echo "===== [TEST DPFT] Labeling synthetic tweets ====="
python attack/labeling_attack.py \
    --input  "$SYN_CSV" \
    --output "${DPFT_OUT}/synthetic_labeled.csv"

echo "===== [TEST DPFT] Proportion analysis ====="
python attack/proportion.py \
    --priv "$DATA" \
    --priv-labeled "$PRIV_LABELED" \
    --syn-labeled  "${DPFT_OUT}/synthetic_labeled.csv" \
    --output "${DPFT_OUT}/attack/labeling_attack.json"

echo ""
echo "===== Test DPFT pipeline done → ${DPFT_OUT} ====="
ls -la "$DPFT_OUT"
