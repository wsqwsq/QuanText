#!/usr/bin/env bash
# Smoke test of the PE pipeline across all 4 datasets.

set -euo pipefail
cd "$(dirname "$0")"

NUM_ITER=2
EPSILON=1.0
NUM_SAMPLES=8
RESULTS="results_test"
TMP_DIR="${RESULTS}/data_tmp"
mkdir -p "$TMP_DIR"

FINAL_ITER_STR=$(printf "%09d" "$NUM_ITER")

DATASETS=(
  "data/gender_0.3.csv|gender|gender|gender_0.3"
  "data/gender_0.5.csv|gender|gender|gender_0.5"
  "data/gender_0.7.csv|gender|gender|gender_0.7"
  "data/diagnosis.csv|diagnosis|diagnosis|diagnosis"
)

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r DATA ATTR PROMPT_SUB SLUG <<< "$entry"
  TMP="${TMP_DIR}/${SLUG}_tmp.csv"
  # Ensure ≥1 sample per label class — sample 12 stratified
  python -c "
import pandas as pd
df = pd.read_csv('${DATA}')
df = df.groupby('${ATTR}', group_keys=False).apply(lambda g: g.sample(min(len(g), 4), random_state=0))
df.to_csv('${TMP}', index=False)"
  OUT="${RESULTS}/pe/${SLUG}"
  SYN_CSV="${OUT}/synthetic_text/${FINAL_ITER_STR}.csv"
  PRIV_LABELED="${RESULTS}/${SLUG}_tmp_labeled.csv"

  rm -rf "$OUT"
  mkdir -p "${OUT}/attack"
  echo ""
  echo "############### [TEST PE] ${SLUG} ###############"

  python PE/main.py --data "$TMP" --exp-folder "$OUT" \
      --prompts-dir "PE/prompts/${PROMPT_SUB}" --attribute-col "$ATTR" \
      --num-iterations "$NUM_ITER" --num-samples "$NUM_SAMPLES" \
      --epsilon "$EPSILON" --delta 1e-5 --skip-fid \
      --max-completion-tokens 128

  if [ ! -f "$SYN_CSV" ]; then echo "missing $SYN_CSV"; ls -la "${OUT}/synthetic_text"; exit 1; fi

  python evaluation/evaluate.py --priv "$TMP" --gen "$SYN_CSV" --attr "$SYN_CSV" \
      --attribute-cols "$ATTR" --output "${OUT}/evaluation.json"

  if [ ! -f "$PRIV_LABELED" ]; then
    python attack/labeling_attack.py --input "$TMP" --output "$PRIV_LABELED" \
        --attribute-col "$ATTR" --prompt "attack/prompts/${PROMPT_SUB}/labeling.json"
  fi
  python attack/labeling_attack.py --input "$SYN_CSV" \
      --output "${OUT}/rewritten_labeled.csv" --attribute-col "$ATTR" \
      --prompt "attack/prompts/${PROMPT_SUB}/labeling.json"
  python attack/proportion.py --priv "$TMP" --priv-labeled "$PRIV_LABELED" \
      --syn-labeled "${OUT}/rewritten_labeled.csv" --attribute-col "$ATTR" \
      --output "${OUT}/attack/labeling_attack.json"
  echo "===== Done ${SLUG} → ${OUT} ====="
done
