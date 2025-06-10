#!/usr/bin/env python3
"""
VAE Sanity Check Script

This script performs an empirical check of the trained VAE model by:
1. Loading the trained model
2. Encoding the whole training set
3. Plotting the latent distribution 
4. Showing statistics about the latent space

The latent distribution should ideally look like a unit sphere (standard normal),
but in practice it often appears as a tight, shifted, possibly correlated blob.

EXPECTED RESULTS:
================

Well-trained VAE:
- μ values centered around 0 (small absolute values)
- σ² values close to 1 
- Low correlation between latent dimensions
- Approximately normal distribution in each dimension
- KL divergence around 0.5-2.0 per dimension

Poorly-trained VAE (posterior collapse):
- σ² values much smaller than 1 (e.g., < 0.1)
- μ values may be shifted away from 0
- Very tight distribution (samples clustered)
- Generated samples will lack diversity

Over-regularized VAE:
- σ² values much larger than 1 (e.g., > 5)
- Poor reconstruction quality
- Too much emphasis on matching prior

USAGE EXAMPLES:
===============

Basic usage:
    python vae_sanity.py

With custom checkpoint:
    python vae_sanity.py --checkpoint ckpt/my_model.pt

Analyze fewer samples for speed:
    python vae_sanity.py --max_samples 1000

Use specific device:
    python vae_sanity.py --device cuda
"""

import os
import sys
import torch
from src.utils import load_config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import argparse
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trainer.vae_trainer import VAETrainer
from src.data.data_module import VAEDataModule
from src.utils.config import Config


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_trained_model(checkpoint_path: str, device: str = 'auto'):
    """
    Load the trained VAE model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded VAE model
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    model, model_config = VAETrainer.load_model(checkpoint_path, device)
    return model, device


def check_latent_distribution(model, data_loader, device, logger, max_samples=5000):
    """
    Check the latent distribution by encoding the training data.
    
    Args:
        model: Trained VAE model
        data_loader: DataLoader for training data
        device: Device to run inference on
        logger: Logger instance
        max_samples: Maximum number of samples to analyze (for memory efficiency)
    """
    logger.info("Encoding training data to check latent distribution...")
    
    model.eval()
    mus, logvars = [], []
    total_samples = 0
    
    with torch.no_grad():
        for batch_num, batch_cat in data_loader:
            if total_samples >= max_samples:
                break
                
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            
            # Get latent distribution parameters
            mu, logvar = model.encoder(batch_num, batch_cat)
            
            # Take only the CLS token (first token) which represents the whole sample
            mu_cls = mu[:, 0, :]  # Shape: (batch_size, d_token)
            logvar_cls = logvar[:, 0, :]  # Shape: (batch_size, d_token)
            
            mus.append(mu_cls.cpu())
            logvars.append(logvar_cls.cpu())
            
            total_samples += batch_num.shape[0]
    
    # Concatenate all batches
    mu = torch.cat(mus, 0).numpy()
    logvar = torch.cat(logvars, 0).numpy()
    sigma = np.exp(0.5 * logvar)
    
    logger.info(f"Analyzed {mu.shape[0]} samples with latent dimension {mu.shape[1]}")
    
    # Plot latent distribution
    plot_latent_distribution(mu, sigma, logger)
    
    # Print statistics
    print_latent_statistics(mu, sigma, logger)
    
    return mu, sigma


def plot_latent_distribution(mu, sigma, logger):
    """
    Plot the latent distribution.
    
    Args:
        mu: Mean values of latent distribution (n_samples, latent_dim)
        sigma: Standard deviation values (n_samples, latent_dim)
        logger: Logger instance
    """
    latent_dim = mu.shape[1]
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    if latent_dim >= 2:
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Latent Space Analysis', fontsize=16)
        
        # 2D scatter plot of first two dimensions
        sns.scatterplot(x=mu[:, 0], y=mu[:, 1], alpha=0.6, ax=axes[0, 0])
        axes[0, 0].set_title('μ₁ vs μ₂ (should be centered on (0,0))')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('μ₁')
        axes[0, 0].set_ylabel('μ₂')
        
        # Histogram of first dimension
        axes[0, 1].hist(mu[:, 0], bins=50, alpha=0.7, density=True, label='μ₁')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Target mean (0)')
        axes[0, 1].set_title('Distribution of μ₁')
        axes[0, 1].set_xlabel('μ₁')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        
        # Standard deviation plot
        mean_sigma = sigma.mean(axis=0)
        axes[1, 0].bar(range(min(latent_dim, 10)), mean_sigma[:10])
        axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Target σ (1)')
        axes[1, 0].set_title('Mean Standard Deviation by Dimension')
        axes[1, 0].set_xlabel('Latent Dimension')
        axes[1, 0].set_ylabel('Mean σ')
        axes[1, 0].legend()
        
        # Correlation matrix of mu values
        if latent_dim <= 10:  # Only show correlation for small dimensions
            corr_matrix = np.corrcoef(mu.T)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=axes[1, 1], cbar_kws={'shrink': 0.8})
            axes[1, 1].set_title('Correlation Matrix of μ')
        else:
            # For high dimensions, show correlation of first few dimensions
            corr_matrix = np.corrcoef(mu[:, :6].T)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=axes[1, 1], cbar_kws={'shrink': 0.8})
            axes[1, 1].set_title('Correlation Matrix of μ (first 6 dims)')
        
        plt.tight_layout()
        plt.show()
        
    else:
        # For 1D latent space
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        ax1.hist(mu[:, 0], bins=50, alpha=0.7, density=True)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Target mean (0)')
        ax1.set_title('Distribution of μ₁')
        ax1.set_xlabel('μ₁')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # Standard deviation
        ax2.hist(sigma[:, 0], bins=50, alpha=0.7, density=True)
        ax2.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Target σ (1)')
        ax2.set_title('Distribution of σ₁')
        ax2.set_xlabel('σ₁')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


def print_latent_statistics(mu, sigma, logger):
    """
    Print statistics about the latent distribution.
    
    Args:
        mu: Mean values of latent distribution
        sigma: Standard deviation values 
        logger: Logger instance
    """
    print("\n" + "="*60)
    print("LATENT SPACE STATISTICS")
    print("="*60)
    
    print(f"\nDataset size: {mu.shape[0]} samples")
    print(f"Latent dimension: {mu.shape[1]}")
    
    print(f"\nMean of μ (should be close to 0):")
    mu_mean = mu.mean(axis=0)
    for i, val in enumerate(mu_mean[:10]):  # Show first 10 dimensions
        print(f"  Dimension {i}: {val:.6f}")
    if mu.shape[1] > 10:
        print(f"  ... and {mu.shape[1] - 10} more dimensions")
    
    print(f"\nStd of μ (indicates spread):")
    mu_std = mu.std(axis=0)
    for i, val in enumerate(mu_std[:10]):
        print(f"  Dimension {i}: {val:.6f}")
    if mu.shape[1] > 10:
        print(f"  ... and {mu.shape[1] - 10} more dimensions")
    
    print(f"\nMean of σ² (should be close to 1 for good regularization):")
    sigma_sq_mean = (sigma**2).mean(axis=0)
    for i, val in enumerate(sigma_sq_mean[:10]):
        print(f"  Dimension {i}: {val:.6f}")
    if mu.shape[1] > 10:
        print(f"  ... and {mu.shape[1] - 10} more dimensions")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Mean |μ|: {np.linalg.norm(mu_mean):.6f} (should be close to 0)")
    print(f"  Mean σ²: {sigma_sq_mean.mean():.6f} (should be close to 1)")
    print(f"  Max |μ|: {np.abs(mu).max():.6f}")
    print(f"  Min σ²: {(sigma**2).min():.6f}")
    print(f"  Max σ²: {(sigma**2).max():.6f}")
    
    # KL divergence estimate (approximate)
    kl_div = 0.5 * np.mean(mu**2 + sigma**2 - np.log(sigma**2) - 1)
    print(f"  Approximate KL divergence: {kl_div:.6f}")
    
    print("\n" + "="*60)
    
    # Interpretation guidance
    print("\nINTERPRETation GUIDE:")
    print("- μ should be centered around 0 (standard normal prior)")
    print("- σ² should be close to 1 (standard normal prior)")
    print("- If μ is far from 0, the model hasn't learned to use the prior well")
    print("- If σ² is much less than 1, the latent space is 'collapsed'")
    print("- If σ² is much greater than 1, the latent space is 'expanded'")
    print("- High correlation between dimensions indicates redundancy")
    print("="*60)


def main():
    """Main function to run the VAE sanity check."""
    parser = argparse.ArgumentParser(description='VAE Sanity Check')
    parser.add_argument('--config', type=str, default='config/vae_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='ckpt/model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum number of samples to analyze')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting VAE sanity check...")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.info("Please train a model first or specify the correct checkpoint path")
        return
    
    # Load config
    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
    
    # Load trained model
    try:
        model, device = load_trained_model(args.checkpoint, args.device)
        logger.info(f"Loaded model from {args.checkpoint} on device {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Setup data module
    try:
        data_module = VAEDataModule(config, logger)
        data_module.setup()
        train_loader, _ = data_module.get_dataloaders()
        logger.info("Loaded training data")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Perform sanity check
    try:
        mu, sigma = check_latent_distribution(model, train_loader, device, logger, args.max_samples)
        logger.info("Sanity check completed successfully!")
    except Exception as e:
        logger.error(f"Failed during sanity check: {e}")
        return


if __name__ == "__main__":
    main() 