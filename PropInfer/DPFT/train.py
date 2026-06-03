"""DP-LoRA fine-tuning of Llama-3.1-8B-Instruct on PropInfer dialogues."""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model
from opacus import PrivacyEngine


def format_example(text, tokenizer, instruction, max_len):
    prompt = f"Instruction: {instruction}\nDialogue: "
    full = prompt + str(text).strip() + tokenizer.eos_token
    ef = tokenizer(full, truncation=True, max_length=max_len,
                   padding="max_length", return_tensors="pt")
    ep = tokenizer(prompt, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = ef["input_ids"][0]; attn = ef["attention_mask"][0]
    labels = input_ids.clone()
    labels[:ep["input_ids"].shape[1]] = -100
    labels[attn == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


class TextDataset(Dataset):
    def __init__(self, csv_path, text_col, tokenizer, instruction, max_len):
        df = pd.read_csv(csv_path)
        if text_col not in df.columns:
            raise ValueError(f"{csv_path} missing '{text_col}'")
        self.examples = [format_example(t, tokenizer, instruction, max_len)
                         for t in df[text_col].astype(str).tolist()]

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",     required=True)
    ap.add_argument("--out",      required=True, help="Adapter output dir.")
    ap.add_argument("--instruction", required=True,
                    help="Single fixed instruction the model is conditioned on.")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--model",    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--epochs",          type=int,   default=15)
    ap.add_argument("--batch-size",      type=int,   default=8)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--max-len",         type=int,   default=1024,
                    help="ChatDoctor dialogues are ~400-3000 tokens; 1024 covers ~p85.")
    ap.add_argument("--lora-r",          type=int,   default=8)
    ap.add_argument("--lora-alpha",      type=int,   default=16)
    ap.add_argument("--lora-dropout",    type=float, default=0.0)
    ap.add_argument("--epsilon",         type=float, default=1.0)
    ap.add_argument("--delta",           type=float, default=None)
    ap.add_argument("--max-grad-norm",   type=float, default=1.0)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--bf16-base",       action="store_true", default=True)
    ap.add_argument("--no-bf16-base",    dest="bf16_base", action="store_false")
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_dtype = torch.bfloat16 if args.bf16_base else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=base_dtype)
    base.config.pad_token_id = tokenizer.pad_token_id
    for p in base.parameters():
        p.requires_grad = False
    if args.grad_checkpoint:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        base.config.use_cache = False

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout,
                      target_modules=["q_proj", "v_proj"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    if args.bf16_base:
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
    model.print_trainable_parameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.train()

    ds = TextDataset(args.data, args.text_col, tokenizer, args.instruction, args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, collate_fn=collate)

    N = len(ds)
    delta = args.delta if args.delta is not None else 1.0 / N / np.log(N)
    print(f"N={N}, target ε={args.epsilon}, δ={delta:.2e}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    pe_engine = PrivacyEngine()
    model, optimizer, loader = pe_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=loader,
        target_epsilon=args.epsilon, target_delta=delta,
        epochs=args.epochs, max_grad_norm=args.max_grad_norm,
    )

    for epoch in range(args.epochs):
        run_loss, steps = 0.0, 0
        for step, batch in enumerate(loader):
            if batch["input_ids"].shape[0] == 0:
                optimizer.zero_grad(set_to_none=True)
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            run_loss += float(out.loss.detach()); steps += 1
            if step % 10 == 0:
                eps_now = pe_engine.get_epsilon(delta)
                print(f"  epoch {epoch} step {step}: loss={out.loss.item():.4f} ε≈{eps_now:.3f}")
        print(f"epoch {epoch}: avg loss = {run_loss / max(steps, 1):.4f}")

    inner = model._module if hasattr(model, "_module") else model
    inner.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    # Save the instruction used during training so generation can reuse it verbatim.
    with open(os.path.join(args.out, "instruction.txt"), "w", encoding="utf-8") as f:
        f.write(args.instruction)
    print(f"Saved adapter → {args.out} (final ε≈{pe_engine.get_epsilon(delta):.3f}, δ={delta:.2e})")


if __name__ == "__main__":
    main()
