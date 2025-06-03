import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.clean import clean
from src.data.wrangle import wrangle
from src.data.dataset import split_numerical_categorical, reconstruct_dataframe
from src.data.dataset import TabularDataset, preprocess_data, PreprocessedData


# Load data
try:
    raw_data = pd.read_csv('FACE/neuropsy_v0.csv', sep=';', low_memory=False)
    raw_data = raw_data.drop(columns=['usubjid_neuropsychologie', 'visitnum_neuropsychologie', 'visit_neuropsychologie', 'visit_neuropsychologie'])
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Clean data
df_cleaned = clean(raw_data)
df_cleaned.to_csv('FACE/neuropsy_v0_cleaned.csv', index=False)

# Wrangle data
df_wrangled, params, wr = wrangle(df_cleaned, conversion_threshold=0.80, cardinality_threshold=10, scale_numeric=True)
df_wrangled.to_csv('FACE/neuropsy_v0_wrangled.csv', index=False)

N_COLS = 60
df_partial = df_wrangled[df_wrangled.columns[:N_COLS]]
df_partial.to_csv(f"FACE/neuropsy_v0_partial_n_cols_{N_COLS}.csv", index=False)

# Split numerical and categorical features
num_mat, cat_mat, mapping = split_numerical_categorical(df_partial, cardinality_threshold=10)

# Convert to correct data types
num_mat = num_mat.astype(np.float32)
cat_mat = cat_mat.astype(np.int64)

# Convert to dataframe to insure that the data is correct
df_partial_converted = reconstruct_dataframe(num_mat, cat_mat, mapping)

num_mat = num_mat.astype(np.float32)
cat_mat = cat_mat.astype(np.int64)

preprocessed_data = preprocess_data(num_mat=num_mat, cat_mat=cat_mat, y=None, mapping=mapping,
    test_size=0.25, transform=False, scaling_strategy=None, cat_encoding=None)

# Generate new samples

import os
import sys
import torch
import numpy as np
import argparse
from typing import List, Tuple, Dict, Any
from src.trainer.vae_trainer import VAETrainer

device = "cpu"
checkpoint = "ckpt/model.pt"
num_samples = 128

model, model_config = VAETrainer.load_model(checkpoint, device=device)

d_token = model_config['model_params']['d_token']
categories = model_config['model_params']['categories']
d_numerical = model_config['model_params']['d_numerical']
print(f"Model configuration:")
print(f"- Latent dimension: {d_token}")
print(f"- Number of numerical features: {d_numerical}")
print(f"- Number of categorical features: {len(categories)}")
print(f"- Categories: {categories}")

num, cat = model.sample(num_samples=128, current_device="cpu")