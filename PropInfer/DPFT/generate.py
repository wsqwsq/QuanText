"""Generate synthetic dialogues from a DP-fine-tuned LoRA adapter."""

import argparse
import csv
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--adapter",    required=True)
    ap.add_argument("--out",        required=True, help="Output CSV with one text column.")
    ap.add_argument("--text-col",   default="text")
    ap.add_argument("--instruction", default=None,
                    help="If unset, read from <adapter>/instruction.txt saved by train.py.")
    ap.add_argument("--num-samples",    type=int,   default=500)
    ap.add_argument("--batch-size",     type=int,   default=8)
    ap.add_argument("--max-new-tokens", type=int,   default=1024,
                    help="ChatDoctor dialogues are ~400-3000 tokens; 1024 covers ~p85.")
    ap.add_argument("--temperature",    type=float, default=1.0)
    ap.add_argument("--top-p",          type=float, default=0.95)
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.instruction is None:
        ipath = os.path.join(args.adapter, "instruction.txt")
        if not os.path.exists(ipath):
            raise FileNotFoundError(f"No --instruction passed and {ipath} not found.")
        with open(ipath, "r", encoding="utf-8") as f:
            args.instruction = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map={"": device},
    )
    model = PeftModel.from_pretrained(base, args.adapter); model.eval()

    prompt = f"Instruction: {args.instruction}\nDialogue: "

    with open(args.out, "w", newline="", encoding="utf-8") as wf:
        writer = csv.writer(wf)
        writer.writerow([args.text_col])
        remaining = args.num_samples
        while remaining > 0:
            n = min(args.batch_size, remaining)
            inputs = tokenizer([prompt] * n, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            for j in range(n):
                gen_ids = out_ids[j, prompt_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                writer.writerow([text])
            remaining -= n
            done = args.num_samples - remaining
            if done % (args.batch_size * 10) == 0:
                print(f"  generated {done}/{args.num_samples}")
    print(f"Saved {args.num_samples} synthetic dialogues → {args.out}")


if __name__ == "__main__":
    main()
