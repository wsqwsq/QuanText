#!/usr/bin/env bash
# Smoke test of the SML pipeline on tiny slices of each dataset.

set -euo pipefail
cd "$(dirname "$0")"

GAMMA=20
K=5
DUPLICATE=1
RESULTS="results_test"
TMP_DIR="${RESULTS}/data_tmp"
mkdir -p "$TMP_DIR"

# Source CSV, tmp slice, attribute, prompt subdir, slug
DATASETS=(
  "data/gender_0.3.csv|gender|gender|gender_0.3"
  "data/gender_0.5.csv|gender|gender|gender_0.5"
  "data/gender_0.7.csv|gender|gender|gender_0.7"
  "data/diagnosis.csv|diagnosis|diagnosis|diagnosis"
)

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r DATA ATTR PROMPT_SUB SLUG <<< "$entry"
  TMP="${TMP_DIR}/${SLUG}_tmp.csv"
  python -c "import pandas as pd; pd.read_csv('${DATA}').sample(n=8, random_state=0).to_csv('${TMP}', index=False)"
  OUT="${RESULTS}/sml/${SLUG}"
  BINS="${OUT}/bins"
  PRIV_LABELED="${RESULTS}/${SLUG}_tmp_labeled.csv"

  rm -rf "$OUT"
  mkdir -p "${OUT}/attack"
  echo ""
  echo "############### [TEST SML] ${SLUG} (attr=${ATTR}) ###############"

  python chow_liu.py --data "$TMP" --cols "$ATTR" --gamma "$GAMMA" --pool-size 100 --outdir "$BINS"
  python quantization.py --data "$TMP" --registry "${BINS}/model_registry.json" \
      --base-dir "$BINS" --k "$K" --out-matched "${OUT}/new_attributes.csv"
  python generation.py --data "$TMP" --rewrite-attrs "${OUT}/new_attributes.csv" \
      --snippets-out "${OUT}/snippets.csv" --save "${OUT}/rewritten.csv" \
      --attribute-col "$ATTR" --prompts-dir "prompts/${PROMPT_SUB}" --duplicate "$DUPLICATE"

  if [ ! -f "$PRIV_LABELED" ]; then
    python attack/labeling_attack.py --input "$TMP" --output "$PRIV_LABELED" \
        --attribute-col "$ATTR" --prompt "attack/prompts/${PROMPT_SUB}/labeling.json"
  fi
  python attack/labeling_attack.py --input "${OUT}/rewritten.csv" \
      --output "${OUT}/rewritten_labeled.csv" --attribute-col "$ATTR" \
      --prompt "attack/prompts/${PROMPT_SUB}/labeling.json"
  python attack/proportion.py --priv "$TMP" --priv-labeled "$PRIV_LABELED" \
      --syn-labeled "${OUT}/rewritten_labeled.csv" --attribute-col "$ATTR" \
      --output "${OUT}/attack/labeling_attack.json"

  python evaluation/evaluate.py --priv "$TMP" --gen "${OUT}/rewritten.csv" \
      --attr "${OUT}/rewritten_with_attr.csv" --attribute-cols "$ATTR" \
      --output "${OUT}/evaluation.json"
  echo "===== Done ${SLUG} → ${OUT} ====="
done
echo ""
echo "Test runs finished. Results under ${RESULTS}/sml/"
