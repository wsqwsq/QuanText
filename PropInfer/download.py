"""
Download and prepare the PropInfer datasets.

Source: https://huggingface.co/datasets/Pengrun/PropInfer_dataset

Produces 4 CSV files under ./data/:

  gender_0.3.csv     500 rows, 150 female / 350 male
  gender_0.5.csv     500 rows, 250 female / 250 male
  gender_0.7.csv     500 rows, 350 female / 150 male
  diagnosis.csv      500 random rows, diagnosis ∈
                     {Digestion, mental disorder, childbirth, others}

Each CSV has two columns:
  text       "Patient: {input}\nDoctor: {output}"
  gender     (for gender_*.csv)        ∈ {female, male}
  diagnosis  (for diagnosis.csv)
"""

import os
import random

import pandas as pd
from datasets import load_dataset


REPO = "Pengrun/PropInfer_dataset"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEED = 42
N = 500


def format_text(row):
    return f"Patient: {row['input']}\nDoctor: {row['output']}"


def _norm(v):
    """Strip optional 'N. ' numeric prefix and lowercase."""
    import re
    return re.sub(r"^\s*\d+\.\s*", "", str(v)).strip().lower()


def map_diagnosis(row):
    """Combine the three binary diagnosis columns into one categorical label
    with priority Digestion → mental disorder → childbirth → others."""
    if _norm(row.get("digestion")) == "digestion":
        return "Digestion"
    if _norm(row.get("mental")) == "mental disorder":
        return "mental disorder"
    if _norm(row.get("birth")) == "birth":
        return "childbirth"
    return "others"


def sample_gender(df_full, ratio, n=N, seed=SEED):
    n_f = int(round(n * ratio))
    n_m = n - n_f
    rng = random.Random(seed + int(ratio * 100))

    female = df_full[df_full["gender"] == "female"]
    male   = df_full[df_full["gender"] == "male"]
    if len(female) < n_f or len(male) < n_m:
        raise ValueError(
            f"Not enough samples: need {n_f}F+{n_m}M, have {len(female)}F+{len(male)}M"
        )

    fem_sample = female.sample(n=n_f, random_state=rng.randint(0, 2**31 - 1))
    mal_sample = male.sample(n=n_m, random_state=rng.randint(0, 2**31 - 1))
    out = pd.concat([fem_sample, mal_sample]).sample(
        frac=1.0, random_state=rng.randint(0, 2**31 - 1)
    ).reset_index(drop=True)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Gender subset ───────────────────────────────────────────────────────
    print(f"Loading {REPO} / gender ...")
    gender_ds = load_dataset(REPO, "gender", split="train")
    gender_df = gender_ds.to_pandas()
    # Values look like "1. female" / "2. male" — normalize to "female"/"male".
    gender_df["gender"] = (
        gender_df["gender"].astype(str)
        .str.lower()
        .str.replace(r"^\s*\d+\.\s*", "", regex=True)
        .str.strip()
    )
    print(f"  loaded {len(gender_df)} rows; gender counts:")
    print(gender_df["gender"].value_counts())

    for ratio in (0.3, 0.5, 0.7):
        sub = sample_gender(gender_df, ratio)
        out_df = pd.DataFrame({
            "text":   sub.apply(format_text, axis=1).tolist(),
            "gender": sub["gender"].tolist(),
        })
        path = os.path.join(OUT_DIR, f"gender_{ratio}.csv")
        out_df.to_csv(path, index=False)
        actual = (out_df["gender"] == "female").mean()
        print(f"  wrote {path}  (female ratio = {actual:.3f})")

    # ── Medical diagnosis subset ────────────────────────────────────────────
    print(f"Loading {REPO} / medical_diagnosis ...")
    diag_ds = load_dataset(REPO, "medical_diagnosis", split="train")
    diag_df = diag_ds.to_pandas()
    print(f"  loaded {len(diag_df)} rows")

    diag_sample = diag_df.sample(n=N, random_state=SEED).reset_index(drop=True)
    diag_sample["diagnosis"] = diag_sample.apply(map_diagnosis, axis=1)
    out_df = pd.DataFrame({
        "text":      diag_sample.apply(format_text, axis=1).tolist(),
        "diagnosis": diag_sample["diagnosis"].tolist(),
    })
    path = os.path.join(OUT_DIR, "diagnosis.csv")
    out_df.to_csv(path, index=False)
    print(f"  wrote {path}  (label counts:)")
    print(out_df["diagnosis"].value_counts())


if __name__ == "__main__":
    main()
