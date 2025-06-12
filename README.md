# Tabular VAE for Psychiatric Data

A modern Variational Autoencoder (VAE) implementation designed for mixed tabular data containing both numerical and categorical features. This project provides a complete pipeline from data preprocessing to VAE training with comprehensive monitoring and evaluation capabilities.

## Architecture Overview

The project is structured around three main components:

### 1. **Data Preprocessing Pipeline** (`preprocessing.py`)
- **MLDataTransformer**: Intelligent data type detection and transformation
- **Missing Data Analysis**: Comprehensive missing value detection and visualization
- **Imputation**: Multiple strategies for numerical (median, mean, KNN) and categorical (mode, KNN) data
- **Output**: Produces `numerical_data_imputed.csv`, `categorical_data_imputed.csv`, and `categorical_info.csv`

### 2. **VAE Model Architecture** (`src/ldm/vae/`)
- **Encoder** (`encoder.py`): Transformer-based encoder with attention mechanism
- **Decoder** (`decoder.py`): Transformer-based decoder for reconstruction
- **Core** (`core.py`): Transformer blocks with multi-head attention and feed-forward layers
- **Model** (`model.py`): Complete VAE implementation with reparameterization trick

### 3. **Training Framework** (`src/trainer/vae.py`)
- **VAETrainer**: Comprehensive training loop with modern PyTorch practices
- **Beta Scheduling**: Linear annealing of KL divergence weight over epochs
- **Correlation Analysis**: Automated analysis comparing real vs reconstructed data correlations
- **Checkpointing**: Automatic model saving with best model selection

## Key Features

### Data Handling
- **Mixed Data Types**: Handles both numerical and categorical features seamlessly
- **Intelligent Imputation**: KNN and statistical imputation methods
- **Data Validation**: Comprehensive checks for data integrity and missing values
- **Reproducible Preprocessing**: Saves transformation parameters for consistent application

### Model Architecture
- **Transformer-Based VAE**: Uses attention mechanisms for better feature relationships
- **Token-Based Representation**: Each feature is represented as a token with learned embeddings
- **Configurable Architecture**: Easily adjustable layers, heads, dimensions, and factors
- **Hardware Optimization**: Supports CUDA, MPS (Apple Silicon), and CPU training

### Training Features
- **Beta Annealing**: Gradual increase of KL weight for stable training
- **Early Stopping**: Prevents overfitting with validation-based stopping
- **Learning Rate Scheduling**: Adaptive learning rate based on validation loss
- **Comprehensive Logging**: TensorBoard integration with detailed metrics

### Monitoring & Analysis
- **Real-time Metrics**: Loss components (MSE, Cross-entropy, KL divergence)
- **Correlation Analysis**: Periodic comparison of data correlations (real vs reconstructed)
- **TensorBoard Integration**: Visual monitoring of training progress
- **Model Checkpointing**: Automatic saving of best models and training states

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd synth-data-psy

# Install dependencies (Python 3.8+)
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preprocessing

First, prepare your data using the preprocessing pipeline:

```python
from preprocessing import MLDataTransformer, impute_missing_data, split_ml_data

# Load and preprocess data
transformer = MLDataTransformer(categorical_threshold=10, save_folder='DATA/processed')
df_processed = transformer.fit_transform(raw_data)

# Split into numerical and categorical
numerical_df, categorical_df, categorical_info_df = split_ml_data(df_processed, transformer)

# Impute missing values
numerical_imputed, categorical_imputed = impute_missing_data(
    numerical_df, categorical_df, 
    numerical_strategy='knn', 
    categorical_strategy='most_frequent',
    save_folder='DATA/processed'
)
```

### 2. VAE Training

Train the VAE model using the preprocessed data:

```bash
# Basic training with default configuration
python train_vae.py --data_folder DATA/processed

# Custom configuration
python train_vae.py --config config/custom_config.yaml --data_folder DATA/processed

# Hardware-specific training
python train_vae.py --device cuda:0    # NVIDIA GPU
python train_vae.py --device mps       # Apple Silicon
python train_vae.py --device cpu       # CPU only
```

### 3. Monitor Training

Track training progress with TensorBoard:

```bash
tensorboard --logdir=runs
```

## Project Structure

```
synth-data-psy/
├── src/
│   ├── data/
│   │   └── dataset.py              # PyTorch dataset for mixed tabular data
│   ├── ldm/vae/
│   │   ├── core.py                 # Transformer building blocks
│   │   ├── encoder.py              # VAE encoder with attention
│   │   ├── decoder.py              # VAE decoder for reconstruction
│   │   └── model.py                # Complete VAE model
│   └── trainer/
│       └── vae.py                  # Training loop and utilities
├── config/
│   └── vae_config.yaml             # Training configuration
├── preprocessing.py                # Data preprocessing pipeline
├── train_vae.py                    # Main training script
├── generate_synthetic_data.py      # Synthetic data generation script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Configuration

The training configuration is managed through YAML files in the `config/` directory:

```yaml
model:
  num_layers: 2        # Number of transformer layers
  d_token: 16          # Token dimension
  n_head: 1            # Number of attention heads
  factor: 32           # Feed-forward expansion factor
  token_bias: true     # Use bias in token embeddings

training:
  batch_size: 512      # Training batch size
  num_epochs: 400      # Maximum epochs
  learning_rate: 0.001 # Initial learning rate
  weight_decay: 0      # L2 regularization
  beta:
    min: 0.001         # Initial KL weight
    max: 0.15          # Final KL weight
  early_stopping_patience: 400  # Early stopping patience
  scheduler_patience: 10        # LR scheduler patience
  scheduler_factor: 0.95        # LR reduction factor

correlation_freq: 5    # Correlation analysis frequency (epochs)
checkpoint_dir: ckpt   # Checkpoint directory
seed: 42              # Random seed
```

## How It Works

### Data Flow

1. **Raw Data** → **Preprocessing Pipeline**:
   - Automatic column type detection (numerical vs categorical)
   - Missing data analysis and visualization
   - Intelligent imputation based on data characteristics
   - Data validation and integrity checks

2. **Preprocessed Data** → **Dataset Creation**:
   - Separate numerical and categorical tensors
   - Categorical encoding information preservation
   - PyTorch dataset wrapper for efficient loading

3. **Dataset** → **VAE Training**:
   - Token-based representation of features
   - Attention mechanism for feature relationships
   - Beta-scheduled KL divergence annealing
   - Comprehensive loss tracking and validation

### VAE Architecture Details

The VAE uses a transformer-based architecture where:

- **Input Processing**: Each feature (numerical or categorical) becomes a token
- **Encoder**: Multi-layer transformer that outputs latent mean (μ) and log-variance (σ²)
- **Reparameterization**: Samples latent vectors using μ + σ × ε (ε ~ N(0,1))
- **Decoder**: Multi-layer transformer that reconstructs original features from latent vectors

### Loss Function

The total loss combines three components:

```
L_total = L_reconstruction + β × L_KL

where:
- L_reconstruction = MSE(numerical) + CrossEntropy(categorical)
- L_KL = KL_divergence(latent_distribution || N(0,1))
- β = linearly annealed weight (min → max over epochs)
```

### Training Process

1. **Initialization**: Model, optimizer, scheduler, and data loaders
2. **Training Loop**: 
   - Forward pass through encoder-decoder
   - Loss computation with current β value
   - Backpropagation with gradient clipping
   - Learning rate scheduling based on validation loss
3. **Validation**: Model evaluation without gradient updates
4. **Monitoring**: 
   - TensorBoard logging of all metrics
   - Periodic correlation analysis between real and reconstructed data
   - Automatic checkpointing of best models

### Outputs

#### Training Outputs

After training, the following artifacts are generated:

- **Model Checkpoints**: Complete model state for resuming training (`ckpt/best_model.pt`)
- **Best Model**: Best performing model based on validation loss
- **TensorBoard Logs**: Comprehensive training metrics and visualizations (`runs/`)
- **Correlation Analysis**: Heatmaps comparing real vs reconstructed data correlations

#### Synthetic Data Generation Outputs

When using `generate_synthetic_data.py`, the following files are created:

- **`synthetic_data_original_scale.csv`**: Synthetic data transformed back to original scales and units
- **`synthetic_data_transformed_scale.csv`**: Synthetic data in the same scale as training (normalized/encoded)
- **`correlation_comparison.png`**: Side-by-side correlation heatmaps (original vs synthetic vs difference)
- **`generation_metadata.json`**: Complete metadata including model config, feature info, and generation parameters

## Advanced Usage

### Custom Data Preprocessing

```python
# Custom preprocessing with specific parameters
transformer = MLDataTransformer(
    categorical_threshold=5,  # Treat features with ≤5 unique values as categorical
    save_transforms=True,     # Save transformation parameters
    save_folder='custom_processed'
)

# Apply custom imputation strategies
numerical_imputed, categorical_imputed = impute_missing_data(
    numerical_df, categorical_df,
    numerical_strategy='knn',           # KNN imputation for numerical
    categorical_strategy='most_frequent', # Mode imputation for categorical
    knn_neighbors=10,                   # Number of neighbors for KNN
    save_folder='custom_processed'
)
```

### Synthetic Data Generation

After training, use the dedicated script to generate synthetic data with comprehensive analysis:

```bash
# Basic synthetic data generation
python generate_synthetic_data.py --checkpoint ckpt/best_model.pt --data_folder FACE/processed --num_samples 1000

# Custom output folder and device
python generate_synthetic_data.py \
    --checkpoint ckpt/best_model.pt \
    --data_folder DATA/processed \
    --num_samples 5000 \
    --output_folder my_synthetic_data \
    --device cuda:0
```

This script provides:
- **Automatic model and data loading** from preprocessing pipeline
- **Synthetic sample generation** using the trained VAE
- **Correlation analysis** with side-by-side heatmaps comparing real vs synthetic data
- **Inverse transformation** back to original data scales
- **Proper column ordering** matching the original dataset
- **Multiple output formats** (original scale, transformed scale, metadata)

#### Manual Sampling (Advanced)

For custom applications, generate samples programmatically:

```python
from src.ldm.vae.model import VAE

# Load trained model
model = VAE(...)
model.load_state_dict(torch.load('ckpt/best_model.pt')['model_state_dict'])
model.eval()

# Generate samples
num_samples = 1000
synthetic_numerical, synthetic_categorical = model.sample(num_samples, device)
```

### Custom Training Configuration

Override configuration parameters via command line:

```bash
python train_vae.py \
    --data_folder custom_data/ \
    --batch_size 256 \
    --epochs 500 \
    --correlation_freq 10 \
    --device cuda:1
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` in configuration
2. **Slow Training**: Use GPU acceleration (`--device cuda:0` or `--device mps`)
3. **Poor Reconstruction**: Increase model capacity (`d_token`, `num_layers`) or training epochs
4. **Missing Data Files**: Ensure preprocessing pipeline completed successfully

### Monitoring Training Health

- **KL Loss**: Should gradually increase as β increases
- **Reconstruction Loss**: Should steadily decrease
- **Validation Loss**: Should improve without diverging from training loss
- **Correlation Analysis**: Reconstructed correlations should approach real data correlations

## License

[Specify your license here] 