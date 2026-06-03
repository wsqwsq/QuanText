#!/usr/bin/env bash
# Private Evolution sweep on PropInfer: ε ∈ {1,2,3,4}, num-iterations=6.
# Output layout:
#   results/pe/${slug}/eps${eps}/...
# Legacy compat: if ε=1 results already exist at results/pe/${slug}/attack/labeling_attack.json
# (the old flat layout from before the eps subdir), treat ε=1 as already done.

set -euo pipefail
cd "$(dirname "$0")"

NUM_ITER=6
NUM_SAMPLES=500
EPSILONS=(1 2 3 4)

RESULTS="results"
mkdir -p "$RESULTS"

FINAL_ITER_STR=$(printf "%09d" "$NUM_ITER")

DATASETS=(
  "data/gender_0.3.csv|gender|gender|gender_0.3"
  "data/gender_0.5.csv|gender|gender|gender_0.5"
  "data/gender_0.7.csv|gender|gender|gender_0.7"
  "data/diagnosis.csv|diagnosis|diagnosis|diagnosis"
)

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r DATA ATTR PROMPT_SUB SLUG <<< "$entry"

  PRIV_LABELED="${RESULTS}/${SLUG}_labeled.csv"
  LEGACY_FLAT_MARKER="${RESULTS}/pe/${SLUG}/attack/labeling_attack.json"

  if [ ! -f "$PRIV_LABELED" ]; then
    python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED" \
        --attribute-col "$ATTR" --prompt "attack/prompts/${PROMPT_SUB}/labeling.json"
  fi

  for EPS in "${EPSILONS[@]}"; do
    OUT="${RESULTS}/pe/${SLUG}/eps${EPS}"
    SYN_CSV="${OUT}/synthetic_text/${FINAL_ITER_STR}.csv"
    MARKER="${OUT}/attack/labeling_attack.json"

    if [ -f "$MARKER" ]; then
      echo "===== ${SLUG} ε=${EPS} already done (${MARKER}), skipping ====="
      continue
    fi
    if [ "$EPS" = "1" ] && [ -f "$LEGACY_FLAT_MARKER" ]; then
      echo "===== ${SLUG} ε=1 legacy results at ${LEGACY_FLAT_MARKER}, skipping ====="
      continue
    fi

    mkdir -p "${OUT}/attack"
    echo ""
    echo "############### PE ${SLUG} ε=${EPS} → ${OUT} ###############"

    python PE/main.py \
        --data "$DATA" --exp-folder "$OUT" \
        --prompts-dir "PE/prompts/${PROMPT_SUB}" \
        --attribute-col "$ATTR" \
        --num-iterations "$NUM_ITER" --num-samples "$NUM_SAMPLES" \
        --epsilon "$EPS" --skip-fid

    if [ ! -f "$SYN_CSV" ]; then
      echo "ERROR: expected synthetic CSV not found: $SYN_CSV" >&2
      ls -la "${OUT}/synthetic_text" >&2 || true; exit 1
    fi

    python evaluation/evaluate.py \
        --priv "$DATA" --gen "$SYN_CSV" --attr "$SYN_CSV" \
        --attribute-cols "$ATTR" \
        --output "${OUT}/evaluation.json"

    python attack/labeling_attack.py \
        --input  "$SYN_CSV" \
        --output "${OUT}/rewritten_labeled.csv" \
        --attribute-col "$ATTR" \
        --prompt "attack/prompts/${PROMPT_SUB}/labeling.json"
    python attack/proportion.py \
        --priv "$DATA" --priv-labeled "$PRIV_LABELED" \
        --syn-labeled  "${OUT}/rewritten_labeled.csv" \
        --attribute-col "$ATTR" \
        --output "$MARKER"
    echo "===== Done ${SLUG} ε=${EPS} → ${OUT} ====="
  done
done

echo ""
echo "PE sweep finished. Results under ${RESULTS}/pe/*/eps*/"
