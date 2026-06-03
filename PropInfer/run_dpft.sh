#!/usr/bin/env bash
# DP-LoRA sweep on PropInfer: ε ∈ {1,2,3,4}.
# Output layout:
#   results/dpft/${slug}/eps${eps}/{adapter, synthetic.csv, ...}
# Legacy compat: if ε=1 results already exist at results/dpft/${slug}/attack/labeling_attack.json
# (the old flat layout), treat ε=1 as already done.

set -euo pipefail
cd "$(dirname "$0")"

EPOCHS=15
NUM_SAMPLES=500
EPSILONS=(1 2 3 4)

RESULTS="results"
mkdir -p "$RESULTS"

INSTR="Below is a doctor-patient medical conversation. The patient describes their symptoms and the doctor responds. Write a complete two-line dialogue in this format:\nPatient: <symptoms>\nDoctor: <response>"

# slug | csv | attribute | prompt_sub
DATASETS=(
  "gender_0.3|data/gender_0.3.csv|gender|gender"
  "gender_0.5|data/gender_0.5.csv|gender|gender"
  "gender_0.7|data/gender_0.7.csv|gender|gender"
  "diagnosis|data/diagnosis.csv|diagnosis|diagnosis"
)

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r SLUG DATA ATTR PROMPT_SUB <<< "$entry"

  PRIV_LABELED="${RESULTS}/${SLUG}_labeled.csv"
  LEGACY_FLAT_MARKER="${RESULTS}/dpft/${SLUG}/attack/labeling_attack.json"

  if [ ! -f "$PRIV_LABELED" ]; then
    python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED" \
        --attribute-col "$ATTR" --prompt "attack/prompts/${PROMPT_SUB}/labeling.json"
  fi

  for EPS in "${EPSILONS[@]}"; do
    OUT="${RESULTS}/dpft/${SLUG}/eps${EPS}"
    ADAPTER="${OUT}/adapter"
    SYN_CSV="${OUT}/synthetic.csv"
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
    echo "############### DPFT ${SLUG} ε=${EPS} → ${OUT} ###############"

    if [ ! -f "${ADAPTER}/adapter_config.json" ]; then
      python DPFT/train.py --data "$DATA" --out "$ADAPTER" \
          --instruction "$INSTR" --epsilon "$EPS" --epochs "$EPOCHS"
    fi

    python DPFT/generate.py --adapter "$ADAPTER" --out "$SYN_CSV" \
        --num-samples "$NUM_SAMPLES"

    python evaluation/evaluate.py --priv "$DATA" --gen "$SYN_CSV" --attr "$SYN_CSV" \
        --attribute-cols "$ATTR" --output "${OUT}/evaluation.json"

    python attack/labeling_attack.py --input "$SYN_CSV" \
        --output "${OUT}/synthetic_labeled.csv" --attribute-col "$ATTR" \
        --prompt "attack/prompts/${PROMPT_SUB}/labeling.json"
    python attack/proportion.py --priv "$DATA" --priv-labeled "$PRIV_LABELED" \
        --syn-labeled "${OUT}/synthetic_labeled.csv" --attribute-col "$ATTR" \
        --output "$MARKER"
    echo "===== Done ${SLUG} ε=${EPS} → ${OUT} ====="
  done
done

echo ""
echo "DPFT sweep finished. Results under ${RESULTS}/dpft/*/eps*/"
