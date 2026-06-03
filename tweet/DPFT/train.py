"""
DP fine-tuning baseline: LoRA fine-tune Llama-3.1-8B-Instruct with DP-SGD.

The model is conditioned on a single fixed instruction. Only the LoRA
adapters are trained — the base model stays frozen, so Opacus only has
to track per-example gradients for the (small) LoRA matrices.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model
from opacus import PrivacyEngine


INSTRUCTION = (
    "Write a short Twitter post (under 280 characters) about a political or "
    "social topic such as climate change, abortion, atheism, the feminist "
    "movement, or a political figure. Output only the tweet text."
)


def format_example(tweet, tokenizer, max_len):
    """Build input_ids/labels for a single (instruction, tweet) example.

    Loss is masked over the prompt (instruction) tokens so the model only
    learns to produce the tweet continuation.
    """
    prompt = f"Instruction: {INSTRUCTION}\nTweet: "
    full = prompt + str(tweet).strip() + tokenizer.eos_token

    enc_full = tokenizer(full, truncation=True, max_length=max_len,
                         padding="max_length", return_tensors="pt")
    enc_prompt = tokenizer(prompt, truncation=True, max_length=max_len,
                           return_tensors="pt")

    input_ids = enc_full["input_ids"][0]
    attn = enc_full["attention_mask"][0]
    labels = input_ids.clone()
    prompt_len = enc_prompt["input_ids"].shape[1]
    labels[:prompt_len] = -100      # mask prompt
    labels[attn == 0] = -100        # mask padding

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


class TweetDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=256):
        df = pd.read_csv(csv_path)
        if "Tweet" not in df.columns:
            raise ValueError(f"{csv_path} must have a 'Tweet' column.")
        self.examples = [format_example(t, tokenizer, max_len)
                         for t in df["Tweet"].astype(str).tolist()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "../data/tweet.csv"))
    ap.add_argument("--out",  default=os.path.join(os.path.dirname(__file__), "../results/dpft/adapter"),
                    help="Where the LoRA adapter weights are saved.")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--epochs",          type=int,   default=3)
    ap.add_argument("--batch-size",      type=int,   default=8)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--max-len",         type=int,   default=256)
    ap.add_argument("--lora-r",          type=int,   default=8)
    ap.add_argument("--lora-alpha",      type=int,   default=16)
    ap.add_argument("--lora-dropout",    type=float, default=0.0)
    ap.add_argument("--epsilon",         type=float, default=1.0)
    ap.add_argument("--delta",           type=float, default=None,
                    help="If unset, defaults to 1/N/log(N).")
    ap.add_argument("--max-grad-norm",   type=float, default=1.0)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--bf16-base",       action="store_true", default=True,
                    help="Load the frozen base model in bf16 to save ~16 GB. "
                         "LoRA params are kept in fp32 (Opacus only injects noise into LoRA).")
    ap.add_argument("--no-bf16-base",    dest="bf16_base", action="store_false")
    ap.add_argument("--grad-checkpoint", action="store_true", default=True,
                    help="Enable activation checkpointing on the base model.")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    # ── Tokenizer + base model (frozen) ─────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_dtype = torch.bfloat16 if args.bf16_base else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=base_dtype)
    base.config.pad_token_id = tokenizer.pad_token_id
    for p in base.parameters():
        p.requires_grad = False

    # Activation checkpointing — slashes activation memory by ~4–8×.
    if args.grad_checkpoint:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        # cache must be off for checkpointing
        base.config.use_cache = False

    # ── LoRA ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)

    # Keep LoRA params in fp32 even when the base is bf16. Opacus' Gaussian
    # noise is unstable in low precision, and these layers are the only ones
    # it actually adds noise to.
    if args.bf16_base:
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # ── Data ────────────────────────────────────────────────────────────
    ds = TweetDataset(args.data, tokenizer, max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, collate_fn=collate)

    # ── DP accounting ───────────────────────────────────────────────────
    N = len(ds)
    delta = args.delta if args.delta is not None else 1.0 / N / np.log(N)
    print(f"N={N}, target ε={args.epsilon}, δ={delta:.2e}")

    # ── Opacus ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        target_epsilon=args.epsilon,
        target_delta=delta,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
    )

    # ── Train loop ──────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        running_loss, n_steps = 0.0, 0
        for step, batch in enumerate(loader):
            # Opacus replaces the DataLoader with a Poisson sampler, so the
            # actual batch size is Binomial(N, q) and is occasionally 0. Skip
            # those steps — they still count toward the privacy budget.
            if batch["input_ids"].shape[0] == 0:
                optimizer.zero_grad(set_to_none=True)
                optimizer.skip_step = True if hasattr(optimizer, "skip_step") else None
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            running_loss += float(out.loss.detach())
            n_steps += 1
            if step % 10 == 0:
                eps_now = privacy_engine.get_epsilon(delta)
                print(f"  epoch {epoch} step {step}: loss={out.loss.item():.4f} ε≈{eps_now:.3f}")
        print(f"epoch {epoch}: avg loss = {running_loss / max(n_steps, 1):.4f}")

    # ── Save LoRA adapter only (the base model is unchanged) ────────────
    # PEFT model is wrapped by Opacus GradSampleModule → unwrap to call save.
    inner = model._module if hasattr(model, "_module") else model
    inner.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    final_eps = privacy_engine.get_epsilon(delta)
    print(f"Saved LoRA adapter to {args.out} (final ε≈{final_eps:.3f}, δ={delta:.2e})")


if __name__ == "__main__":
    main()
