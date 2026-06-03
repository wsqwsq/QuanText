#!/usr/bin/env bash
# Smoke test of DPFT across all 4 datasets.

set -euo pipefail
cd "$(dirname "$0")"

EPSILON=1.0
EPOCHS=1
NUM_SAMPLES=4
BATCH_SIZE=2
RESULTS="results_test"
TMP_DIR="${RESULTS}/data_tmp"
mkdir -p "$TMP_DIR"

INSTR="Below is a doctor-patient medical conversation. The patient describes their symptoms and the doctor responds. Write a complete two-line dialogue:\nPatient: <symptoms>\nDoctor: <response>"

DATASETS=(
  "gender_0.3|data/gender_0.3.csv|gender"
  "gender_0.5|data/gender_0.5.csv|gender"
  "gender_0.7|data/gender_0.7.csv|gender"
  "diagnosis|data/diagnosis.csv|diagnosis"
)

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r SLUG DATA ATTR <<< "$entry"
  TMP="${TMP_DIR}/${SLUG}_tmp.csv"
  python -c "import pandas as pd; pd.read_csv('${DATA}').sample(n=8, random_state=0).to_csv('${TMP}', index=False)"
  OUT="${RESULTS}/dpft/${SLUG}"
  ADAPTER="${OUT}/adapter"
  SYN_CSV="${OUT}/synthetic.csv"
  PRIV_LABELED="${RESULTS}/${SLUG}_tmp_labeled.csv"

  rm -rf "$OUT"
  mkdir -p "${OUT}/attack"
  echo ""
  echo "############### [TEST DPFT] ${SLUG} ###############"

  python DPFT/train.py --data "$TMP" --out "$ADAPTER" --instruction "$INSTR" \
      --epsilon "$EPSILON" --delta 1e-5 --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" --max-len 256

  python DPFT/generate.py --adapter "$ADAPTER" --out "$SYN_CSV" \
      --num-samples "$NUM_SAMPLES" --batch-size "$BATCH_SIZE" --max-new-tokens 128

  python evaluation/evaluate.py --priv "$TMP" --gen "$SYN_CSV" --attr "$SYN_CSV" \
      --attribute-cols "$ATTR" --output "${OUT}/evaluation.json"

  if [ ! -f "$PRIV_LABELED" ]; then
    python attack/labeling_attack.py --input "$TMP" --output "$PRIV_LABELED" \
        --attribute-col "$ATTR" --prompt "attack/prompts/${ATTR}/labeling.json"
  fi
  python attack/labeling_attack.py --input "$SYN_CSV" \
      --output "${OUT}/synthetic_labeled.csv" --attribute-col "$ATTR" \
      --prompt "attack/prompts/${ATTR}/labeling.json"
  python attack/proportion.py --priv "$TMP" --priv-labeled "$PRIV_LABELED" \
      --syn-labeled "${OUT}/synthetic_labeled.csv" --attribute-col "$ATTR" \
      --output "${OUT}/attack/labeling_attack.json"
  echo "===== Done ${SLUG} → ${OUT} ====="
done
