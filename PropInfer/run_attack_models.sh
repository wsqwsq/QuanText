#!/usr/bin/env bash
# Re-run the labeling-based proportion attack on every (generation method,
# dataset) combination with two additional attacker LLMs:
#   - mistralai/Mistral-7B-Instruct-v0.3
#   - Qwen/Qwen2.5-7B-Instruct        (closest non-thinking Qwen ≈ "Qwen-3.5 9B")
#
# Output (lives next to the existing labeling_attack.json):
#   results/<method>/<slug>/attack/labeling_attack_mistral.json
#   results/<method>/<slug>/attack/labeling_attack_qwen.json

set -euo pipefail
cd "$(dirname "$0")"

RESULTS="results"

declare -A MODELS=(
  [mistral]="mistralai/Mistral-7B-Instruct-v0.3"
  [qwen]="Qwen/Qwen2.5-7B-Instruct"
)

DATASETS=(
  "data/gender_0.3.csv|gender|gender|gender_0.3"
  "data/gender_0.5.csv|gender|gender|gender_0.5"
  "data/gender_0.7.csv|gender|gender|gender_0.7"
  "data/diagnosis.csv|diagnosis|diagnosis|diagnosis"
)

# (method dir, function to produce synthetic CSV given slug)
sml_gen()       { echo "${RESULTS}/sml/$1/rewritten.csv"; }
pe_gen()        { echo "${RESULTS}/pe/$1/synthetic_text/000000010.csv"; }
dpft_gen()      { echo "${RESULTS}/dpft/$1/synthetic.csv"; }
subsample_gen() { echo "${RESULTS}/subsample/$1/subsample.csv"; }

METHODS=(sml pe dpft subsample)

for mslug in "${!MODELS[@]}"; do
  MODEL="${MODELS[$mslug]}"
  echo ""
  echo "##################################################################"
  echo "##  Attacker model: ${MODEL}  (slug=${mslug})"
  echo "##################################################################"

  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r DATA ATTR PROMPT_SUB SLUG <<< "$entry"
    PRIV_LABELED="${RESULTS}/${SLUG}_labeled_${mslug}.csv"
    PROMPT="attack/prompts/${PROMPT_SUB}/labeling.json"

    # 1) Private re-labeling once per (attacker model, dataset)
    if [ ! -f "$PRIV_LABELED" ]; then
      echo "----- [${mslug}/${SLUG}] Labeling private -----"
      python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED" \
          --attribute-col "$ATTR" --prompt "$PROMPT" --model "$MODEL"
    fi

    # 2) Per method
    for m in "${METHODS[@]}"; do
      GEN=$(${m}_gen "$SLUG")
      DIR=$(dirname $(dirname "$GEN"))   # parent of dirname(GEN) — e.g. results/pe/<slug>
      # For pe, GEN is .../<slug>/synthetic_text/000000010.csv so dirname/dirname = .../<slug>
      # For others, GEN is .../<slug>/x.csv so dirname/dirname = parent of <slug>; fix:
      if [ "$m" = "pe" ]; then
        DIR="$(dirname "$(dirname "$GEN")")"
      else
        DIR="$(dirname "$GEN")"
      fi

      if [ ! -f "$GEN" ]; then
        echo "----- [${mslug}/${m}/${SLUG}] synthetic CSV missing (${GEN}), skipping -----"
        continue
      fi
      SYN_LABELED="${DIR}/synthetic_labeled_${mslug}.csv"
      ATTACK_JSON="${DIR}/attack/labeling_attack_${mslug}.json"
      mkdir -p "${DIR}/attack"

      if [ ! -f "$SYN_LABELED" ]; then
        python attack/labeling_attack.py --input "$GEN" --output "$SYN_LABELED" \
            --attribute-col "$ATTR" --prompt "$PROMPT" --model "$MODEL"
      fi
      python attack/proportion.py \
          --priv "$DATA" \
          --priv-labeled "$PRIV_LABELED" \
          --syn-labeled  "$SYN_LABELED" \
          --attribute-col "$ATTR" \
          --output "$ATTACK_JSON"
      echo "----- ${m}/${SLUG}: wrote ${ATTACK_JSON} -----"
    done
  done
done

echo ""
echo "All attacker-model runs finished."
