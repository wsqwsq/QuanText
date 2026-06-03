"""
Generate synthetic tweets from the DP-fine-tuned LoRA adapter, using the
same fixed instruction as in training.
"""

import argparse
import csv
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Re-use the instruction text from training so prompts match exactly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import INSTRUCTION  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--adapter",    default=os.path.join(os.path.dirname(__file__), "../results/dpft/adapter"))
    ap.add_argument("--out",        default=os.path.join(os.path.dirname(__file__), "../results/dpft/synthetic.csv"))
    ap.add_argument("--num-samples",    type=int,   default=2814)
    ap.add_argument("--batch-size",     type=int,   default=8)
    ap.add_argument("--max-new-tokens", type=int,   default=128)
    ap.add_argument("--temperature",    type=float, default=1.0)
    ap.add_argument("--top-p",          type=float, default=0.95)
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for batched generation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map={"": device},
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    prompt = f"Instruction: {INSTRUCTION}\nTweet: "

    with open(args.out, "w", newline="", encoding="utf-8") as wf:
        writer = csv.writer(wf)
        writer.writerow(["Tweet"])

        remaining = args.num_samples
        while remaining > 0:
            n = min(args.batch_size, remaining)
            inputs = tokenizer([prompt] * n, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            for j in range(n):
                gen_ids = out_ids[j, prompt_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                text = text.splitlines()[0].strip() if text else ""
                writer.writerow([text])
            remaining -= n
            done = args.num_samples - remaining
            if done % (args.batch_size * 10) == 0:
                print(f"  generated {done}/{args.num_samples}")

    print(f"Saved {args.num_samples} synthetic tweets → {args.out}")


if __name__ == "__main__":
    main()
