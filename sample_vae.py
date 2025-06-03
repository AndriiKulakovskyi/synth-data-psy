#!/usr/bin/env python3
import os
import argparse
import torch
import pandas as pd
import numpy as np

from src.data.clean import clean
from src.data.wrangle import wrangle
from src.data import split_numerical_categorical, reconstruct_decoded_dataframe
from src.trainer.vae_trainer import VAETrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic data using a trained VAE")
    parser.add_argument('--data_path', type=str, default='FACE/neuropsy_v0.csv',
                        help='Path to the raw CSV used during training')
    parser.add_argument('--checkpoint', type=str, default='ckpt/model.pt',
                        help='Path to the trained VAE checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of synthetic rows to generate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--n_cols', type=int, default=60,
                        help='Number of wrangled columns used during training')
    parser.add_argument('--output', type=str, default='generated_samples/generated_data.csv',
                        help='Where to save the generated data CSV')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ─── load and preprocess dataset identically to training ────────────────
    df_raw = pd.read_csv(args.data_path, sep=';', low_memory=False)
    # drop duplicated ID/visit columns if present
    cols_to_drop = [
        'usubjid_neuropsychologie',
        'visitnum_neuropsychologie',
        'visit_neuropsychologie',
        'visit_neuropsychologie',  # kept as in original script
    ]
    df_raw = df_raw.drop(columns=[c for c in cols_to_drop if c in df_raw.columns], errors='ignore')

    df_clean = clean(df_raw)
    df_wrangled, _params, wrangler = wrangle(
        df_clean,
        conversion_threshold=0.80,
        cardinality_threshold=10,
        scale_numeric=True,
    )

    df_partial = df_wrangled[df_wrangled.columns[: args.n_cols]]

    num_mat, cat_mat, mapping = split_numerical_categorical(df_partial, cardinality_threshold=10)
    num_mat = num_mat.astype(np.float32)
    cat_mat = cat_mat.astype(np.int64)

    # ─── load model ──────────────────────────────────────────────────────────
    model, _model_cfg = VAETrainer.load_model(args.checkpoint, device=args.device)

    # ─── generate synthetic samples ─────────────────────────────────────────
    with torch.no_grad():
        x_num, x_cat = model.sample(args.num_samples, current_device=args.device)

    # ─── reconstruct dataframe in original format ──────────────────────────
    df_generated = reconstruct_decoded_dataframe(
        numerical_matrix=x_num,
        categorical_matrix=x_cat,
        mapping=mapping,
        wrangler=wrangler,
        drop_scaled=True,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_generated.to_csv(args.output, index=False)
    print(f"Saved {len(df_generated)} rows to {args.output}")


if __name__ == '__main__':
    main()
