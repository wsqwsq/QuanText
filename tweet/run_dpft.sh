#!/usr/bin/env bash
# DP-LoRA sweep on the tweet dataset.
# Runs ε ∈ {1, 2, 3, 4}. Output:
#   results/dpft/eps${eps}/{adapter, synthetic.csv, synthetic_labeled.csv,
#                           evaluation.json, attack/labeling_attack.json}
# Skip rule: if the per-eps attack JSON already exists, skip that eps.
# Legacy compat: if ε=1 results already exist at results/dpft/attack/labeling_attack.json,
# treat ε=1 as already done.

set -euo pipefail
cd "$(dirname "$0")"

EPOCHS=15
NUM_SAMPLES=2814
EPSILONS=(1 2 3 4)

DATA="data/tweet.csv"
RESULTS="results"
PRIV_LABELED="${RESULTS}/tweet_labeled.csv"

if [ ! -f "$PRIV_LABELED" ]; then
    echo "===== Labeling private dataset (shared) ====="
    python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED"
fi

for EPS in "${EPSILONS[@]}"; do
    OUT="${RESULTS}/dpft/eps${EPS}"
    ADAPTER="${OUT}/adapter"
    SYN_CSV="${OUT}/synthetic.csv"
    MARKER="${OUT}/attack/labeling_attack.json"
    LEGACY="${RESULTS}/dpft/attack/labeling_attack.json"

    if [ -f "$MARKER" ]; then
        echo "===== ε=${EPS} already done (${MARKER}), skipping ====="
        continue
    fi
    if [ "$EPS" = "1" ] && [ -f "$LEGACY" ]; then
        echo "===== ε=1 legacy results at ${LEGACY}, skipping ====="
        continue
    fi

    mkdir -p "${OUT}/attack"
    echo "===== DPFT ε=${EPS} → ${OUT} ====="

    if [ ! -f "${ADAPTER}/adapter_config.json" ]; then
        python DPFT/train.py \
            --data "$DATA" --out "$ADAPTER" \
            --epsilon "$EPS" --epochs "$EPOCHS"
    fi

    python DPFT/generate.py \
        --adapter "$ADAPTER" --out "$SYN_CSV" \
        --num-samples "$NUM_SAMPLES"

    python evaluation/evaluate.py \
        --priv "$DATA" --gen "$SYN_CSV" --attr "$SYN_CSV" \
        --output "${OUT}/evaluation.json"

    python attack/labeling_attack.py \
        --input  "$SYN_CSV" \
        --output "${OUT}/synthetic_labeled.csv"
    python attack/proportion.py \
        --priv "$DATA" --priv-labeled "$PRIV_LABELED" \
        --syn-labeled "${OUT}/synthetic_labeled.csv" \
        --output "$MARKER"
    echo "===== Done ε=${EPS} → ${OUT} ====="
done

echo ""
echo "DPFT sweep finished. Results under ${RESULTS}/dpft/eps*/"
