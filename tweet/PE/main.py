"""
Private Evolution (DPSDA) on the Tweet Stance dataset.

Generates an (epsilon, delta)-DP synthetic tweet dataset using:
  - LLaMA-3.1-8B-Instruct as the foundation model
  - LLMAugPE with random + variation prompts conditioned on (Target, Stance, Sentiment)
  - SentenceTransformer embeddings + NearestNeighbors histogram

Hyperparameters: 10 PE iterations, epsilon = 1.
See https://microsoft.github.io/DPSDA/ for API details.
"""

import argparse
import os

import numpy as np
import pandas as pd

from pe.api.text import LLMAugPE
from pe.callback import ComputeFID, SaveCheckpoints, SaveTextToCSV
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME
from pe.data.text import TextCSV
from pe.embedding.text import SentenceTransformer
from pe.histogram import NearestNeighbors
from pe.llm import HuggingfaceLLM
from pe.logger import CSVPrint, LogPrint
from pe.logging import setup_logging
from pe.population import PEPopulation
from pe.runner import PE

pd.options.mode.copy_on_write = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "../data/tweet.csv"),
                    help="Private tweet CSV (must contain Tweet/Target/Stance/Sentiment columns).")
    ap.add_argument("--exp-folder", default=os.path.join(os.path.dirname(__file__), "../results/pe"),
                    help="Output folder for checkpoints, synthetic CSVs, and logs.")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="HuggingFace model id for the foundation LLM.")
    ap.add_argument("--num-iterations", type=int, default=10,
                    help="Number of PE iterations (T).")
    ap.add_argument("--num-samples", type=int, default=2814,
                    help="Synthetic samples per PE iteration.")
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="DP epsilon budget.")
    ap.add_argument("--delta", type=float, default=None,
                    help="DP delta. If unset, uses 1/N/log(N) where N is |private dataset|. "
                         "Override for small datasets where the formula yields a loose value "
                         "that the Gaussian-DP accountant cannot bracket.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-completion-tokens", type=int, default=128,
                    help="Tweets are short, so a small token cap is plenty.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-fid", action="store_true",
                    help="Skip the ComputeFID callback (cleanfid's covariance sqrt fails on "
                         "tiny test datasets).")
    args = ap.parse_args()

    exp_folder = os.path.abspath(args.exp_folder)
    os.makedirs(exp_folder, exist_ok=True)
    current_folder = os.path.dirname(os.path.abspath(__file__))

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    # ── Private data ────────────────────────────────────────────────────────
    data = TextCSV(
        csv_path=args.data,
        label_columns=["Target", "Stance", "Sentiment"],
        text_column="Tweet",
    )

    # ── LLM + PE APIs ───────────────────────────────────────────────────────
    llm = HuggingfaceLLM(
        max_completion_tokens=args.max_completion_tokens,
        batch_size=args.batch_size,
        model_name_or_path=args.model,
        temperature=args.temperature,
    )
    api = LLMAugPE(
        llm=llm,
        random_api_prompt_file=os.path.join(current_folder, "random_api_prompt.json"),
        variation_api_prompt_file=os.path.join(current_folder, "variation_api_prompt.json"),
    )

    # ── Embedding + histogram ───────────────────────────────────────────────
    embedding = SentenceTransformer(model="stsb-roberta-base-v2")
    histogram = NearestNeighbors(
        embedding=embedding,
        mode="L2",
        lookahead_degree=0,
    )

    # ── PE population ───────────────────────────────────────────────────────
    population = PEPopulation(
        api=api,
        initial_variation_api_fold=2,
        next_variation_api_fold=2,
        keep_selected=True,
        selection_mode="rank",
    )

    # ── Callbacks + loggers ─────────────────────────────────────────────────
    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    compute_fid = ComputeFID(
        priv_data=data,
        embedding=embedding,
        filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1},
    )
    save_text_to_csv = SaveTextToCSV(output_folder=os.path.join(exp_folder, "synthetic_text"))
    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    callbacks = [save_checkpoints, save_text_to_csv]
    if not args.skip_fid:
        callbacks.append(compute_fid)

    # ── DP accounting ───────────────────────────────────────────────────────
    num_private_samples = len(data.data_frame)
    if args.delta is not None:
        delta = args.delta
    else:
        delta = 1.0 / num_private_samples / np.log(num_private_samples)

    # ── Run PE ──────────────────────────────────────────────────────────────
    pe_runner = PE(
        priv_data=data,
        population=population,
        histogram=histogram,
        callbacks=callbacks,
        loggers=[csv_print, log_print],
    )
    # num_samples_schedule has (num_iterations + 1) entries: the initial
    # generation plus one per PE iteration.
    pe_runner.run(
        num_samples_schedule=[args.num_samples] * (args.num_iterations),
        delta=delta,
        epsilon=args.epsilon,
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )


if __name__ == "__main__":
    main()
