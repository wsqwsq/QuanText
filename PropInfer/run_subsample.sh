#!/usr/bin/env bash
# Subsampling baseline across all 4 PropInfer datasets:
# randomly take half of the private samples (without replacement)
# → evaluation → attack.

set -euo pipefail
cd "$(dirname "$0")"

FRACTION=0.5
SEED=42
RESULTS="results"
mkdir -p "$RESULTS"

DATASETS=(
  "data/gender_0.3.csv|gender|gender|gender_0.3"
  "data/gender_0.5.csv|gender|gender|gender_0.5"
  "data/gender_0.7.csv|gender|gender|gender_0.7"
  "data/diagnosis.csv|diagnosis|diagnosis|diagnosis"
)

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r DATA ATTR PROMPT_SUB SLUG <<< "$entry"
  OUT="${RESULTS}/subsample/${SLUG}"
  SYN_CSV="${OUT}/subsample.csv"
  PRIV_LABELED="${RESULTS}/${SLUG}_labeled.csv"

  mkdir -p "${OUT}/attack"
  echo ""
  echo "############### Subsample on ${SLUG} (attr=${ATTR}) ###############"

  # 1) Random subsample without replacement
  python -c "
import pandas as pd
df = pd.read_csv('${DATA}')
n = int(round(len(df) * ${FRACTION}))
df.sample(n=n, random_state=${SEED}, replace=False).to_csv('${SYN_CSV}', index=False)
print(f'Wrote {n} / {len(df)} rows → ${SYN_CSV}')"

  # 2) Evaluation
  python evaluation/evaluate.py \
      --priv "$DATA" --gen "$SYN_CSV" --attr "$SYN_CSV" \
      --attribute-cols "$ATTR" --output "${OUT}/evaluation.json"

  # 3) Attack
  if [ ! -f "$PRIV_LABELED" ]; then
    python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED" \
        --attribute-col "$ATTR" --prompt "attack/prompts/${PROMPT_SUB}/labeling.json"
  fi
  python attack/labeling_attack.py --input "$SYN_CSV" \
      --output "${OUT}/subsample_labeled.csv" --attribute-col "$ATTR" \
      --prompt "attack/prompts/${PROMPT_SUB}/labeling.json"
  python attack/proportion.py --priv "$DATA" --priv-labeled "$PRIV_LABELED" \
      --syn-labeled "${OUT}/subsample_labeled.csv" --attribute-col "$ATTR" \
      --output "${OUT}/attack/labeling_attack.json"
  echo "===== Done ${SLUG} → ${OUT} ====="
done
echo ""
echo "All subsample runs finished. Results under ${RESULTS}/subsample/"
