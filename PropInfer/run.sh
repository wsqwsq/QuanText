#!/usr/bin/env bash
# SML pipeline sweep on PropInfer: γ=120, k ∈ {10,15,20,30,40,60}, dup=2.
# Output layout:
#   results/sml/${slug}/bins/         (per-slug, depends only on γ)
#   results/sml/${slug}/k${k}/...     (per-k results)
# Legacy compat: if k=30 results already exist at results/sml/${slug}/attack/labeling_attack.json
# (the old flat layout from before the k subdir was added), treat k=30 as already done.

set -euo pipefail
cd "$(dirname "$0")"

GAMMA=120
KS=(10 15 20 30 40 60)
DUPLICATE=2

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

  BINS="${RESULTS}/sml/${SLUG}/bins"
  PRIV_LABELED="${RESULTS}/${SLUG}_labeled.csv"
  LEGACY_FLAT_MARKER="${RESULTS}/sml/${SLUG}/attack/labeling_attack.json"

  # Per-slug Chow-Liu candidates (shared across k).
  if [ ! -f "${BINS}/model_registry.json" ]; then
    python chow_liu.py --data "$DATA" --cols "$ATTR" --gamma "$GAMMA" --outdir "$BINS"
  fi

  # Per-slug private labeling (shared across k).
  if [ ! -f "$PRIV_LABELED" ]; then
    python attack/labeling_attack.py --input "$DATA" --output "$PRIV_LABELED" \
        --attribute-col "$ATTR" --prompt "attack/prompts/${PROMPT_SUB}/labeling.json"
  fi

  for K in "${KS[@]}"; do
    OUT="${RESULTS}/sml/${SLUG}/k${K}"
    MARKER="${OUT}/attack/labeling_attack.json"

    if [ -f "$MARKER" ]; then
      echo "===== ${SLUG} k=${K} already done (${MARKER}), skipping ====="
      continue
    fi
    if [ "$K" = "30" ] && [ -f "$LEGACY_FLAT_MARKER" ]; then
      echo "===== ${SLUG} k=30 legacy results at ${LEGACY_FLAT_MARKER}, skipping ====="
      continue
    fi

    mkdir -p "${OUT}/attack"
    echo ""
    echo "############### SML ${SLUG} k=${K} → ${OUT} ###############"

    python quantization.py \
        --data "$DATA" --registry "${BINS}/model_registry.json" \
        --base-dir "$BINS" --k "$K" \
        --out-matched "${OUT}/new_attributes.csv"

    python generation.py \
        --data "$DATA" \
        --rewrite-attrs "${OUT}/new_attributes.csv" \
        --snippets-out  "${OUT}/snippets.csv" \
        --save          "${OUT}/rewritten.csv" \
        --attribute-col "$ATTR" \
        --prompts-dir   "prompts/${PROMPT_SUB}" \
        --duplicate "$DUPLICATE"

    python attack/labeling_attack.py \
        --input  "${OUT}/rewritten.csv" \
        --output "${OUT}/rewritten_labeled.csv" \
        --attribute-col "$ATTR" \
        --prompt "attack/prompts/${PROMPT_SUB}/labeling.json"
    python attack/proportion.py \
        --priv "$DATA" --priv-labeled "$PRIV_LABELED" \
        --syn-labeled  "${OUT}/rewritten_labeled.csv" \
        --attribute-col "$ATTR" \
        --output "$MARKER"

    python evaluation/evaluate.py \
        --priv "$DATA" --gen "${OUT}/rewritten.csv" \
        --attr "${OUT}/rewritten_with_attr.csv" \
        --attribute-cols "$ATTR" \
        --output "${OUT}/evaluation.json"
    echo "===== Done ${SLUG} k=${K} → ${OUT} ====="
  done
done

echo ""
echo "SML sweep finished. Results under ${RESULTS}/sml/*/k*/"
