# Tabular VAE for Psychiatric Data

A Variational Autoencoder (VAE) implementation for tabular psychiatric data. This repository contains code for training a VAE on mixed numerical and categorical data.

## Features

- Handles mixed numerical and categorical features
- Configurable VAE architecture with attention mechanism
- Separate train and validation steps with metrics tracking
- Automatic model checkpointing and early stopping
- KL divergence annealing for better convergence
- Extracts and saves latent embeddings
- Hardware acceleration support (CUDA for NVIDIA GPUs and MPS for Apple Silicon)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synth-data-psy.git
cd synth-data-psy

# Install dependencies (requires Python 3.7+)
pip install -r requirements.txt
```

## Usage

### Configuration

The training parameters are stored in the YAML configuration file at `config/vae_config.yaml`. You can customize these settings to adjust the model architecture, training parameters, and data processing steps.

### Training the VAE

```bash
# Basic usage with default configuration
python train_vae.py

# Using a custom configuration file
python train_vae.py --config path/to/custom_config.yaml

# Override specific configuration settings
python train_vae.py --data_path data/your_data.csv --device cuda:0 --batch_size 128 --epochs 1000

# Using Apple Silicon GPU acceleration
python train_vae.py --device mps
```

### Command Line Arguments

- `--config`: Path to the configuration file (default: `config/vae_config.yaml`)
- `--data_path`: Override the data path specified in the config file
- `--device`: Override the device (e.g., 'cpu', 'cuda:0', 'mps' for Apple Silicon)
- `--checkpoint_dir`: Override the checkpoint directory
- `--batch_size`: Override the training batch size
- `--epochs`: Override the number of training epochs
- `--seed`: Override the random seed for reproducibility

## Project Structure

```
├── config/
│   └── vae_config.yaml   # Training configuration
├── logs/                 # Training logs
├── ckpt/                 # Model checkpoints
├── data/                 # Data files
├── src/
│   ├── data/             # Data processing modules
│   ├── ldm/              # Model architecture
│   ├── trainer/          # Training loop implementation
│   └── utils/            # Utility functions
├── train_vae.py          # Main training script
└── README.md             # This file
```

## Output

After training, the following files will be saved:

- `ckpt/model.pt`: Complete VAE model
- `ckpt/encoder.pt`: Encoder part for generating embeddings
- `ckpt/decoder.pt`: Decoder part for generating synthetic data
- `ckpt/train_z.npy`: Latent embeddings of the training data
- Logs will be saved to the `logs/` directory

## License

[Your License] 