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

## Training the Model

### Starting the Training Process

To start training the VAE model, use the following command:

```bash
# Basic training with default configuration
python train_vae.py

# Training with custom configuration
python train_vae.py --config path/to/custom_config.yaml

# Training with specific hardware
python train_vae.py --device cuda:0  # For NVIDIA GPU
python train_vae.py --device mps     # For Apple Silicon GPU
python train_vae.py --device cpu     # For CPU training
```

### Monitoring Training with TensorBoard

Training progress is automatically logged to TensorBoard. To monitor the training:

1. Install TensorBoard if you haven't already:
   ```bash
   pip install tensorboard
   ```

2. In a separate terminal, start TensorBoard:
   ```bash
   tensorboard --logdir=./runs
   ```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:6006/)

#### Key Metrics to Monitor

- **Losses**:
  - `train/loss`: Total training loss (MSE + CE + β*KL)
  - `val/loss`: Total validation loss
  - `train/mse_loss`: Mean Squared Error for numerical features
  - `train/ce_loss`: Cross-Entropy loss for categorical features
  - `train/kl_loss`: KL Divergence between latent distribution and standard normal

- **Hyperparameters**:
  - `hyperparams/lr`: Learning rate over time
  - `hyperparams/beta`: KL weight (β) value during training

- **Model Statistics**:
  - Parameter histograms (updated every 5 epochs)
  - Gradient histograms (updated every 5 epochs)
  - Model graph (available after first forward pass)

### Sampling from the Trained Model

After training, you can generate synthetic samples using the trained model:

1. Load the trained decoder:
   ```python
   from src.ldm.vae.model import Decoder_model
   import torch
   
   # Initialize decoder with the same architecture as training
   decoder = Decoder_model(
       num_layers=config.model.num_layers,
       d_numerical=your_numerical_dim,
       categories=your_categories_list,
       d_token=config.model.d_token,
       n_head=config.model.n_head,
       factor=config.model.factor
   ).to(device)
   
   # Load trained weights
   decoder.load_state_dict(torch.load('ckpt/decoder.pt'))
   decoder.eval()
   ```

2. Generate samples:
   ```python
   def generate_samples(decoder, num_samples, latent_dim, device='cuda'):
       with torch.no_grad():
           # Sample from standard normal distribution
           z = torch.randn(num_samples, latent_dim).to(device)
           
           # Generate samples
           num_recon, cat_recon = decoder(z)
           
           # Convert to numpy if needed
           num_samples = num_recon.cpu().numpy()
           cat_samples = [t.argmax(dim=-1).cpu().numpy() for t in cat_recon]
           
           return num_samples, cat_samples
   
   # Generate 10 samples
   num_samples, cat_samples = generate_samples(decoder, num_samples=10, latent_dim=config.model.d_token)
   ```

## Output Files

After training, the following files will be saved:

- `ckpt/model.pt`: Complete VAE model
- `ckpt/encoder.pt`: Encoder part for generating embeddings
- `ckpt/decoder.pt`: Decoder part for generating synthetic data
- `ckpt/train_z.npy`: Latent embeddings of the training data
- `runs/`: Directory containing TensorBoard logs
- `logs/`: Directory containing training logs

## License

[Your License] 