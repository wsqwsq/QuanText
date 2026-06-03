#!/usr/bin/env bash
# Re-run the labeling-based proportion attack on every generation method
# with two additional LLMs as the attacker:
#   - mistralai/Mistral-7B-Instruct-v0.3
#   - Qwen/Qwen2.5-7B-Instruct        (closest non-thinking Qwen ≈ "Qwen-3.5 9B")
#
# Output (lives next to the existing Llama-based labeling_attack.json):
#   results/<method>/attack/labeling_attack_mistral.json
#   results/<method>/attack/labeling_attack_qwen.json

set -euo pipefail
cd "$(dirname "$0")"

RESULTS="results"
DATA="data/tweet.csv"

# Attacker model slug → HuggingFace id
declare -A MODELS=(
  [mistral]="mistralai/Mistral-7B-Instruct-v0.3"
  [qwen]="Qwen/Qwen2.5-7B-Instruct"
)

# Generation method → (synthetic CSV path, output dir for attack results)
# (used by the per-method loop below)
SML_GEN="${RESULTS}/gamma120_k30/rewritten_tweets.csv"
SML_DIR="${RESULTS}/gamma120_k30"
PE_GEN="${RESULTS}/pe/synthetic_text/000000010.csv"
PE_DIR="${RESULTS}/pe"
DPFT_GEN="${RESULTS}/dpft/synthetic.csv"
DPFT_DIR="${RESULTS}/dpft"
SUB_GEN="${RESULTS}/subsample/subsample.csv"
SUB_DIR="${RESULTS}/subsample"

METHODS=(
  "sml|${SML_GEN}|${SML_DIR}"
  "pe|${PE_GEN}|${PE_DIR}"
  "dpft|${DPFT_GEN}|${DPFT_DIR}"
  "subsample|${SUB_GEN}|${SUB_DIR}"
)

for slug in "${!MODELS[@]}"; do
  MODEL="${MODELS[$slug]}"
  PRIV_LABELED="${RESULTS}/tweet_labeled_${slug}.csv"
  echo ""
  echo "############### Attacker model: ${MODEL}  (slug=${slug}) ###############"

  # 1) Private re-labeling once per attacker model
  if [ ! -f "$PRIV_LABELED" ]; then
    echo "----- [${slug}] Labeling private dataset -----"
    python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED" --model "$MODEL"
  else
    echo "----- ${PRIV_LABELED} exists, skipping -----"
  fi

  # 2) Per method: re-label synthetic + run proportion attack
  for entry in "${METHODS[@]}"; do
    IFS='|' read -r mname GEN DIR <<< "$entry"
    if [ ! -f "$GEN" ]; then
      echo "----- [${slug}/${mname}] synthetic CSV missing (${GEN}), skipping -----"
      continue
    fi
    SYN_LABELED="${DIR}/synthetic_labeled_${slug}.csv"
    ATTACK_JSON="${DIR}/attack/labeling_attack_${slug}.json"
    mkdir -p "${DIR}/attack"

    if [ ! -f "$SYN_LABELED" ]; then
      python attack/labeling_attack.py --input "$GEN" --output "$SYN_LABELED" --model "$MODEL"
    fi
    python attack/proportion.py \
        --priv "$DATA" \
        --priv-labeled "$PRIV_LABELED" \
        --syn-labeled  "$SYN_LABELED" \
        --output "$ATTACK_JSON"
    echo "----- ${mname}: wrote ${ATTACK_JSON} -----"
  done
done

echo ""
echo "All attacker-model runs finished."
