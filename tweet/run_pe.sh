#!/usr/bin/env bash
# Private Evolution sweep on the tweet dataset.
# Runs ε ∈ {1, 2, 3, 4}. Output:
#   results/pe/eps${eps}/{checkpoint, synthetic_text, evaluation.json,
#                         rewritten_tweets_labeled.csv, attack/labeling_attack.json}
# Skip rule: if the per-eps attack JSON already exists, skip that eps.
# Legacy compat: if ε=1 results already exist at the flat path
#   results/pe/attack/labeling_attack.json, treat ε=1 as already done.

set -euo pipefail
cd "$(dirname "$0")"

NUM_ITER=10
NUM_SAMPLES=2814
EPSILONS=(1 2 3 4)

DATA="data/tweet.csv"
RESULTS="results"
PRIV_LABELED="${RESULTS}/tweet_labeled.csv"
FINAL_ITER_STR=$(printf "%09d" "$NUM_ITER")

if [ ! -f "$PRIV_LABELED" ]; then
    echo "===== Labeling private dataset (shared) ====="
    python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED"
fi

for EPS in "${EPSILONS[@]}"; do
    OUT="${RESULTS}/pe/eps${EPS}"
    MARKER="${OUT}/attack/labeling_attack.json"
    LEGACY="${RESULTS}/pe/attack/labeling_attack.json"

    if [ -f "$MARKER" ]; then
        echo "===== ε=${EPS} already done (${MARKER}), skipping ====="
        continue
    fi
    if [ "$EPS" = "1" ] && [ -f "$LEGACY" ]; then
        echo "===== ε=1 legacy results at ${LEGACY}, skipping ====="
        continue
    fi

    SYN_CSV="${OUT}/synthetic_text/${FINAL_ITER_STR}.csv"
    mkdir -p "${OUT}/attack"

    echo "===== PE ε=${EPS} → ${OUT} ====="
    python PE/main.py \
        --data "$DATA" --exp-folder "$OUT" \
        --num-iterations "$NUM_ITER" --num-samples "$NUM_SAMPLES" \
        --epsilon "$EPS"

    if [ ! -f "$SYN_CSV" ]; then
        echo "ERROR: expected synthetic CSV not found: $SYN_CSV" >&2
        exit 1
    fi

    python evaluation/evaluate.py \
        --priv "$DATA" --gen "$SYN_CSV" --attr "$SYN_CSV" \
        --output "${OUT}/evaluation.json"

    python attack/labeling_attack.py \
        --input  "$SYN_CSV" \
        --output "${OUT}/rewritten_tweets_labeled.csv"
    python attack/proportion.py \
        --priv "$DATA" --priv-labeled "$PRIV_LABELED" \
        --syn-labeled "${OUT}/rewritten_tweets_labeled.csv" \
        --output "$MARKER"
    echo "===== Done ε=${EPS} → ${OUT} ====="
done

echo ""
echo "PE sweep finished. Results under ${RESULTS}/pe/eps*/"
