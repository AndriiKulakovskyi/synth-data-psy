import os
import torch
import numpy as np
import argparse
from typing import List, Tuple, Dict, Any
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer.vae_trainer import VAETrainer

def generate_samples(
    model: torch.nn.Module,
    num_samples: int,
    latent_dim: int,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Generate samples using the trained VAE model.
    
    Args:
        model: Trained VAE model
        num_samples: Number of samples to generate
        latent_dim: Dimension of the latent space
        device: Device to run the model on
        
    Returns:
        Tuple containing:
            - Generated numerical features
            - List of generated categorical features
    """
    # Generate random latent vectors
    z = torch.randn(num_samples, latent_dim).to(device)
    
    # Add sequence dimension expected by the decoder
    z = z.unsqueeze(1)  # Shape: [batch_size, 1, latent_dim]
    
    # Generate samples
    with torch.no_grad():
        # Get decoder outputs
        decoder_output = model.VAE.decoder(z)
        
        # The first output is the numerical reconstruction
        x_num_recon = decoder_output[0]
        
        # The remaining outputs are categorical logits
        x_cat_recon = decoder_output[1:]
        
        # Convert logits to probabilities for categorical variables
        x_cat_probs = [torch.softmax(cat, dim=1) for cat in x_cat_recon]
        
        # Sample from categorical distributions
        x_cat_samples = [torch.multinomial(probs, 1).squeeze() for probs in x_cat_probs]
    
    return x_num_recon, x_cat_samples

def save_samples(
    x_num: torch.Tensor,
    x_cat: List[torch.Tensor],
    output_dir: str = 'generated_samples',
    model_config: Dict[str, Any] = None
) -> None:
    """
    Save generated samples to disk.
    
    Args:
        x_num: Generated numerical features
        x_cat: List of generated categorical features
        output_dir: Directory to save samples
        model_config: Model configuration dictionary
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model configuration if provided
    if model_config is not None:
        import yaml
        with open(os.path.join(output_dir, 'model_config.yaml'), 'w') as f:
            yaml.dump(model_config, f)
    
    # Save numerical features
    num_path = os.path.join(output_dir, 'numerical_samples.npy')
    np.save(num_path, x_num.cpu().numpy())
    print(f"Saved numerical samples to {num_path}")
    
    # Save categorical features
    for i, cat in enumerate(x_cat):
        cat_path = os.path.join(output_dir, f'categorical_{i:02d}_samples.npy')
        np.save(cat_path, cat.cpu().numpy())
        print(f"Saved categorical {i} samples to {cat_path}")
    
    print(f"\nAll generated samples saved to {os.path.abspath(output_dir)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate samples from a trained VAE model')
    parser.add_argument('--checkpoint', type=str, default='ckpt/model.pt',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--num-samples', type=int, default=128,
                       help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default='generated_samples',
                       help='Directory to save generated samples')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for generation (cuda/cpu)')
    args = parser.parse_args()
    
    # Set device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    print(f"Loading model from {args.checkpoint}...")
    model, model_config = VAETrainer.load_model(args.checkpoint, device=device)
    print("Successfully loaded trained model")
    
    # Extract model parameters
    latent_dim = model_config['model_params']['d_token']
    categories = model_config['model_params']['categories']
    print(f"Model configuration:")
    print(f"- Latent dimension: {latent_dim}")
    print(f"- Number of numerical features: {model_config['model_params']['d_numerical']}")
    print(f"- Number of categorical features: {len(categories)}")
    print(f"- Categories: {categories}")
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    x_num, x_cat = generate_samples(model, args.num_samples, latent_dim, device)
    
    # Save generated samples
    save_samples(x_num, x_cat, args.output_dir, model_config)

if __name__ == "__main__":
    main()