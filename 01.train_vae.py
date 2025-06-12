#!/usr/bin/env python3
"""
Refactored VAE Training Script - Integrated with New Preprocessing Pipeline

This script trains a VAE using the preprocessed data from the new pipeline:
- Uses numerical_data_imputed.csv and categorical_data_imputed.csv
- Integrates with MLDataTransformer for consistent data handling
- Streamlined training process with modern PyTorch practices
"""

import os
import yaml
import argparse
import warnings
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

from src.ldm.vae.model import VAE
from src.trainer.vae import VAETrainer
from src.data.dataset import IntegratedTabularDataset

warnings.filterwarnings('ignore')



def load_config(config_path: str) -> Dict:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def main():
    parser = argparse.ArgumentParser(description='Refactored VAE Training with Integrated Preprocessing')
    parser.add_argument('--data_folder', type=str, default='FACE/processed',
                        help='Path to folder with preprocessed data')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # Initialize trainer
    trainer = VAETrainer(config, args.data_folder, args.device)
    
    # Load preprocessed data
    numerical_df, categorical_df, categorical_info = trainer.load_preprocessed_data()
    
    # Create dataloaders
    trainer.create_dataloaders(
        numerical_df, categorical_df,
        train_split=0.8,
        batch_size=config['training']['batch_size']
    )
    
    # Setup model
    trainer.setup_model(numerical_df.shape[1], categorical_info)
    
    # Train
    history = trainer.train()
    
    print(f"Training completed. Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main() 