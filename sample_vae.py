import os
import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any, List
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer.vae_trainer import VAETrainer
from src.data.dataset import reconstruct_dataframe

def generate_samples(
    model: torch.nn.Module,
    num_samples: int,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Generate samples using the trained VAE model.
    
    Args:
        model: Trained VAE model
        num_samples: Number of samples to generate
        device: Device to run the model on
        
    Returns:
        Tuple containing:
            - Generated numerical features
            - List of generated categorical features (as probabilities)
    """
    with torch.no_grad():
        model.eval()
        x_num, x_cat = model.sample(num_samples, device)
        return x_num, x_cat

def plot_embeddings(embeddings: np.ndarray, output_dir: str = 'results') -> None:
    """
    Plot 2D visualizations of the embeddings using t-SNE and PCA.
    
    Args:
        embeddings: Embeddings to visualize (n_samples, n_features)
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Reduce dimensions
    tsne = TSNE(n_components=2, random_state=42)
    pca = PCA(n_components=2, random_state=42)
    
    embeddings_tsne = tsne.fit_transform(embeddings)
    embeddings_pca = pca.fit_transform(embeddings)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # t-SNE plot
    sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1], 
                   alpha=0.6, ax=ax1)
    ax1.set_title('t-SNE Visualization of Embeddings')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    
    # PCA plot
    sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], 
                   alpha=0.6, ax=ax2)
    ax2.set_title('PCA Visualization of Embeddings')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embeddings_visualization.png'))
    plt.close()

def save_samples(
    x_num: torch.Tensor,
    x_cat: List[torch.Tensor],
    mapping: Dict[str, Any] = None,
    output_dir: str = 'generated_samples'
) -> None:
    """
    Save generated samples to disk as CSV files.
    
    Args:
        x_num: Generated numerical features
        x_cat: List of generated categorical features (as probabilities)
        mapping: Optional mapping dictionary from original data
        output_dir: Directory to save samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy
    x_num_np = x_num.cpu().numpy()
    x_cat_np = [cat.cpu().numpy() for cat in x_cat]
    
    # Convert categorical probabilities to class indices
    x_cat_indices = [np.argmax(cat, axis=1) for cat in x_cat_np]
    x_cat_stacked = np.column_stack(x_cat_indices)
    
    if mapping is not None:
        # Reconstruct the DataFrame using the mapping if available
        df = reconstruct_dataframe(
            numerical_matrix=x_num_np,
            categorical_matrix=x_cat_stacked,
            mapping=mapping
        )
    else:
        # Fallback: Create a DataFrame with generic column names
        num_cols = [f"num_{i}" for i in range(x_num_np.shape[1])]
        cat_cols = [f"cat_{i}" for i in range(x_cat_stacked.shape[1])]
        
        df = pd.DataFrame(
            np.hstack([x_num_np, x_cat_stacked]),
            columns=num_cols + cat_cols
        )
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'generated_samples.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")

def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model and config
    print(f"Loading model from {args.checkpoint}")
    model, model_config = VAETrainer.load_model(args.checkpoint, device=device)
    
    # Print model info
    d_token = model_config['model_params']['d_token']
    categories = model_config['model_params']['categories']
    d_numerical = model_config['model_params']['d_numerical']
    print(f"Model configuration:")
    print(f"- Latent dimension: {d_token}")
    print(f"- Number of numerical features: {d_numerical}")
    print(f"- Number of categorical features: {len(categories)}")
    print(f"- Categories: {categories}")
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    x_num, x_cat = generate_samples(model, args.num_samples, device)
    
    # Get mapping if available
    mapping = model_config.get('mapping')
    if mapping is None:
        print("\nWarning: No mapping found in model config. Using generic column names.")
    
    # Save samples
    save_samples(
        x_num=x_num,
        x_cat=x_cat,
        mapping=mapping,  # Can be None
        output_dir=args.output_dir
    )
    
    # Generate and plot embeddings
    print("\nGenerating embeddings visualization...")
    with torch.no_grad():
        # Sample some points from the latent space
        z = torch.randn(1000, d_token).to(device)
        embeddings = model.encoder.mu_head(z).cpu().numpy()
    
    plot_embeddings(embeddings, output_dir=args.output_dir)
    print(f"Saved embeddings visualization to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate samples from a trained VAE model')
    parser.add_argument('--checkpoint', type=str, default='ckpt/model.pt',
                       help='Path to the model checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='generated_samples',
                       help='Directory to save generated samples')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run the model on')
    
    args = parser.parse_args()
    main(args)