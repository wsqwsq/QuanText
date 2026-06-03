"""Private Evolution (DPSDA) on a PropInfer dataset (gender or diagnosis)."""

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
    ap.add_argument("--data",           required=True)
    ap.add_argument("--exp-folder",     required=True)
    ap.add_argument("--prompts-dir",    required=True,
                    help="Dir with random_api_prompt.json and variation_api_prompt.json.")
    ap.add_argument("--text-col",       default="text")
    ap.add_argument("--attribute-col",  required=True, help="e.g. 'gender' or 'diagnosis'.")
    ap.add_argument("--model",          default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--num-iterations", type=int, default=10)
    ap.add_argument("--num-samples",    type=int, default=500)
    ap.add_argument("--epsilon",        type=float, default=1.0)
    ap.add_argument("--delta",          type=float, default=None)
    ap.add_argument("--temperature",    type=float, default=1.0)
    ap.add_argument("--max-completion-tokens", type=int, default=1024,
                    help="ChatDoctor dialogues are ~400-3000 tokens; 1024 covers ~p85.")
    ap.add_argument("--batch-size",     type=int, default=8)
    ap.add_argument("--skip-fid",       action="store_true")
    args = ap.parse_args()

    exp_folder = os.path.abspath(args.exp_folder)
    os.makedirs(exp_folder, exist_ok=True)
    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    data = TextCSV(
        csv_path=args.data,
        label_columns=[args.attribute_col],
        text_column=args.text_col,
    )

    llm = HuggingfaceLLM(
        max_completion_tokens=args.max_completion_tokens,
        batch_size=args.batch_size,
        model_name_or_path=args.model,
        temperature=args.temperature,
    )
    api = LLMAugPE(
        llm=llm,
        random_api_prompt_file=os.path.join(args.prompts_dir, "random_api_prompt.json"),
        variation_api_prompt_file=os.path.join(args.prompts_dir, "variation_api_prompt.json"),
    )

    embedding = SentenceTransformer(model="stsb-roberta-base-v2")
    histogram = NearestNeighbors(embedding=embedding, mode="L2", lookahead_degree=0)

    population = PEPopulation(
        api=api, initial_variation_api_fold=2, next_variation_api_fold=2,
        keep_selected=True, selection_mode="rank",
    )

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    compute_fid = ComputeFID(
        priv_data=data, embedding=embedding,
        filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1},
    )
    save_text = SaveTextToCSV(output_folder=os.path.join(exp_folder, "synthetic_text"))
    csv_print, log_print = CSVPrint(output_folder=exp_folder), LogPrint()

    callbacks = [save_checkpoints, save_text]
    if not args.skip_fid:
        callbacks.append(compute_fid)

    N = len(data.data_frame)
    delta = args.delta if args.delta is not None else 1.0 / N / np.log(N)

    runner = PE(priv_data=data, population=population, histogram=histogram,
                callbacks=callbacks, loggers=[csv_print, log_print])
    runner.run(
        num_samples_schedule=[args.num_samples] * (args.num_iterations),
        delta=delta, epsilon=args.epsilon,
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )


if __name__ == "__main__":
    main()
