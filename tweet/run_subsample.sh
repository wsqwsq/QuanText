#!/usr/bin/env bash
# Subsampling baseline: randomly take half of the private samples
# (without replacement) → evaluation → attack.

set -euo pipefail
cd "$(dirname "$0")"

FRACTION=0.5
SEED=42

DATA="data/tweet.csv"
RESULTS="results"
SUB_OUT="${RESULTS}/subsample"
SYN_CSV="${SUB_OUT}/subsample.csv"
PRIV_LABELED="${RESULTS}/tweet_labeled.csv"

mkdir -p "${SUB_OUT}/attack"

# ── Step 1: random subsample without replacement ─────────────────────────
echo "===== [SUB] Subsampling ${FRACTION} of ${DATA} ====="
python -c "
import pandas as pd
df = pd.read_csv('${DATA}')
n = int(round(len(df) * ${FRACTION}))
df.sample(n=n, random_state=${SEED}, replace=False).to_csv('${SYN_CSV}', index=False)
print(f'Wrote {n} / {len(df)} rows → ${SYN_CSV}')"

# ── Step 2: utility evaluation ───────────────────────────────────────────
echo "===== [SUB] Evaluation ====="
python evaluation/evaluate.py \
    --priv "$DATA" \
    --gen  "$SYN_CSV" \
    --attr "$SYN_CSV" \
    --output "${SUB_OUT}/evaluation.json"

# ── Step 3: attack — label private (cached) and synthetic ────────────────
if [ ! -f "$PRIV_LABELED" ]; then
    echo "===== [SUB] Labeling private dataset ====="
    python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED"
else
    echo "===== Private labeled file exists at ${PRIV_LABELED}, skipping ====="
fi

echo "===== [SUB] Labeling subsampled tweets ====="
python attack/labeling_attack.py \
    --input  "$SYN_CSV" \
    --output "${SUB_OUT}/subsample_labeled.csv"

echo "===== [SUB] Proportion analysis ====="
python attack/proportion.py \
    --priv "$DATA" \
    --priv-labeled "$PRIV_LABELED" \
    --syn-labeled  "${SUB_OUT}/subsample_labeled.csv" \
    --output "${SUB_OUT}/attack/labeling_attack.json"

echo ""
echo "===== Subsampling pipeline done → ${SUB_OUT} ====="
ls -la "$SUB_OUT"
