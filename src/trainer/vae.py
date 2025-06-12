import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

from src.ldm.vae.model import VAE
from src.data.dataset import IntegratedTabularDataset


class VAETrainer:
    """Streamlined VAE trainer integrated with the new preprocessing pipeline."""
    
    def __init__(self, config: Dict, data_folder: str, device: str = 'auto'):
        self.config = config
        self.data_folder = Path(data_folder)
        
        # Setup device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Set random seed
        torch.manual_seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        
        # Initialize training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Correlation logging configuration
        self.correlation_log_freq = config.get('logging', {}).get('correlation_freq', 10)
        self.categorical_info = None  # Will be set during setup
        
        # Pre-compute beta schedule for homogeneous distribution over all epochs
        self.beta_schedule = self._compute_beta_schedule()
        
        # Setup paths
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'ckpt_refactored'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup TensorBoard
        log_dir = Path('runs') / f'vae_refactored_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f'TensorBoard logs: {log_dir}')
        print(f'Correlation analysis will be logged every {self.correlation_log_freq} epochs')
        print(f'Beta schedule computed: {len(self.beta_schedule)} values from {self.beta_schedule[0]:.6f} to {self.beta_schedule[-1]:.6f}')
        
    def _compute_beta_schedule(self) -> List[float]:
        """Compute beta values homogeneously distributed over all epochs."""
        training_config = self.config.get('training', {})
        beta_config = training_config.get('beta', {})
        
        beta_min = beta_config.get('min', 1e-4)
        beta_max = beta_config.get('max', 1e-2)
        num_epochs = training_config.get('num_epochs', 100)
        
        # Linear interpolation from beta_min to beta_max over all epochs
        beta_schedule = []
        for epoch in range(num_epochs):
            beta = beta_min + (beta_max - beta_min) * (epoch / max(num_epochs - 1, 1))
            beta_schedule.append(beta)
        
        return beta_schedule
    
    def load_preprocessed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Load preprocessed data from the new pipeline."""
        # Load imputed data
        numerical_path = self.data_folder / 'numerical_data_imputed.csv'
        categorical_path = self.data_folder / 'categorical_data_imputed.csv'
        categorical_info_path = self.data_folder / 'categorical_info.csv'
        
        if not all(p.exists() for p in [numerical_path, categorical_path, categorical_info_path]):
            raise FileNotFoundError(
                f"Missing preprocessed data files in {self.data_folder}. "
                "Run preprocessing pipeline first."
            )
        
        # Load dataframes
        numerical_df = pd.read_csv(numerical_path, sep=';')
        categorical_df = pd.read_csv(categorical_path, sep=';')
        categorical_info_df = pd.read_csv(categorical_info_path, sep=';')
        
        print(f"Loaded data shapes:")
        print(f"  Numerical: {numerical_df.shape}")
        print(f"  Categorical: {categorical_df.shape}")
        
        # Verify no missing values
        num_missing = numerical_df.isnull().sum().sum()
        cat_missing = categorical_df.isnull().sum().sum()
        
        if num_missing > 0 or cat_missing > 0:
            raise ValueError(f"Found missing values: numerical={num_missing}, categorical={cat_missing}")
        
        # Extract category information
        categorical_info = {}
        for _, row in categorical_info_df.iterrows():
            categorical_info[row['column_name']] = {
                'num_categories': row['num_categories'],
                'encoded_values': eval(row['encoded_values']) if isinstance(row['encoded_values'], str) else row['encoded_values']
            }
        
        print(f"Categorical features: {len(categorical_info)}")
        for col, info in categorical_info.items():
            print(f"  {col}: {info['num_categories']} categories")
        
        return numerical_df, categorical_df, categorical_info
    
    def create_dataloaders(self, numerical_df: pd.DataFrame, categorical_df: pd.DataFrame,
                          train_split: float = 0.8, batch_size: int = 512) -> None:
        """Create train and validation dataloaders."""
        # Create dataset
        dataset = IntegratedTabularDataset(numerical_df, categorical_df)
        
        # Split into train/val
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.get('seed', 42))
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    def setup_model(self, num_features: int, categorical_info: Dict) -> None:
        """Initialize VAE model, optimizer, and scheduler."""
        # Extract category cardinalities
        categories = [info['num_categories'] for info in categorical_info.values()]
        
        # Initialize model
        model_config = self.config.get('model', {})
        self.model = VAE(
            num_layers=model_config.get('num_layers', 2),
            d_numerical=num_features,
            categories=categories,
            d_token=model_config.get('d_token', 4),
            n_head=model_config.get('n_head', 1),
            factor=model_config.get('factor', 32),
            bias=model_config.get('token_bias', True)
        ).to(self.device)
        
        # Initialize optimizer
        training_config = self.config.get('training', {})
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 1e-3),
            weight_decay=training_config.get('weight_decay', 0)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=training_config.get('scheduler_factor', 0.95),
            patience=training_config.get('scheduler_patience', 10),
        )
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized with {total_params:,} trainable parameters")
        print(f"Architecture: {num_features} numerical + {len(categories)} categorical features")
        
        self.categorical_info = categorical_info
    
    def compute_vae_loss(self, x_num: torch.Tensor, x_cat: torch.Tensor,
                        recon_num: torch.Tensor, recon_cat: List[torch.Tensor],
                        mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """Compute VAE loss components."""
        # Reconstruction loss for numerical features (MSE)
        mse_loss = nn.functional.mse_loss(recon_num, x_num, reduction='mean')
        
        # Reconstruction loss for categorical features (Cross-entropy)
        ce_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for i, cat_logits in enumerate(recon_cat):
            if cat_logits is not None:
                ce_loss += nn.functional.cross_entropy(cat_logits, x_cat[:, i], reduction='mean')
                predicted = cat_logits.argmax(dim=-1)
                correct_predictions += (predicted == x_cat[:, i]).sum().item()
                total_predictions += x_cat.size(0)
        
        if len(recon_cat) > 0:
            ce_loss /= len(recon_cat)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        else:
            accuracy = 0.0
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = mse_loss + ce_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss,
            'accuracy': torch.tensor(accuracy)
        }
    
    def generate_synthetic_data_and_log_correlation(self, epoch: int, num_samples: int = 1000) -> None:
        """Generate synthetic data from latent space and log correlation matrix."""
        self.model.eval()
        
        with torch.no_grad():
            # Generate synthetic data using the model's sample method
            synthetic_num, synthetic_cat = self.model.sample(num_samples, self.device)
            
            # Convert to numpy
            synthetic_num_np = synthetic_num.cpu().numpy()
            
            # Convert categorical logits to predictions
            synthetic_cat_predictions = []
            for cat_logits in synthetic_cat:
                if cat_logits is not None:
                    pred = cat_logits.argmax(dim=-1).cpu().numpy()
                    synthetic_cat_predictions.append(pred)
            
            # Combine numerical and categorical data
            if synthetic_cat_predictions:
                synthetic_cat_np = np.column_stack(synthetic_cat_predictions)
                synthetic_combined = np.hstack([synthetic_num_np, synthetic_cat_np])
                
                # Create column names
                num_cols = [f'num_{i}' for i in range(synthetic_num_np.shape[1])]
                cat_cols = [f'cat_{i}' for i in range(synthetic_cat_np.shape[1])]
                all_cols = num_cols + cat_cols
            else:
                synthetic_combined = synthetic_num_np
                all_cols = [f'num_{i}' for i in range(synthetic_num_np.shape[1])]
            
            # Create DataFrame and compute correlation
            synthetic_df = pd.DataFrame(synthetic_combined, columns=all_cols)
            synthetic_corr = synthetic_df.corr(method='pearson')
            
            # Create correlation plot
            plt.figure(figsize=(12, 10))
            sns.heatmap(synthetic_corr, annot=False, cmap='coolwarm', center=0, 
                       square=True, cbar_kws={'shrink': 0.8})
            plt.title(f'Synthetic Data Correlation Matrix (Epoch {epoch+1})')
            plt.xlabel('Features')
            plt.ylabel('Features')
            plt.tight_layout()
            
            # Log to TensorBoard
            self.writer.add_figure('Synthetic_Data_Correlation', plt.gcf(), epoch)
            
            plt.close()
            
            # Log some statistics about the synthetic data
            synthetic_stats = {
                'mean': synthetic_combined.mean(axis=0).mean(),
                'std': synthetic_combined.std(axis=0).mean(),
                'min': synthetic_combined.min(axis=0).mean(),
                'max': synthetic_combined.max(axis=0).mean()
            }
            
            for stat_name, stat_value in synthetic_stats.items():
                self.writer.add_scalar(f'Synthetic_Stats/{stat_name}', stat_value, epoch)
            
            print(f"Synthetic data correlation matrix logged for epoch {epoch+1}")
    
    def log_correlation_analysis(self, epoch: int) -> None:
        """Compute and log correlation analysis between real and reconstructed data."""
        if self.categorical_info is None:
            print("Warning: Categorical info not available for correlation analysis")
            return
            
        self.model.eval()
        
        # Collect all data for correlation analysis
        real_num_data = []
        real_cat_data = []
        recon_num_data = []
        recon_cat_data = []
        
        with torch.no_grad():
            # Use a subset of validation data for correlation analysis
            num_batches_to_use = min(5, len(self.val_loader))  # Limit to avoid memory issues
            
            for batch_idx, (x_num, x_cat) in enumerate(self.val_loader):
                if batch_idx >= num_batches_to_use:
                    break
                    
                x_num, x_cat = x_num.to(self.device), x_cat.to(self.device)
                
                # Forward pass
                recon_num, recon_cat, mu, logvar = self.model(x_num, x_cat)
                
                # Collect real data
                real_num_data.append(x_num.cpu().numpy())
                real_cat_data.append(x_cat.cpu().numpy())
                
                # Collect reconstructed data
                recon_num_data.append(recon_num.cpu().numpy())
                
                # Convert categorical reconstructions to predictions
                cat_predictions = []
                for cat_logits in recon_cat:
                    if cat_logits is not None:
                        pred = cat_logits.argmax(dim=-1).cpu().numpy()
                        cat_predictions.append(pred)
                
                if cat_predictions:
                    cat_predictions = np.column_stack(cat_predictions)
                    recon_cat_data.append(cat_predictions)
        
        # Concatenate all data
        real_num = np.vstack(real_num_data)
        real_cat = np.vstack(real_cat_data) if real_cat_data else np.empty((real_num.shape[0], 0))
        recon_num = np.vstack(recon_num_data)
        recon_cat = np.vstack(recon_cat_data) if recon_cat_data else np.empty((recon_num.shape[0], 0))
        
        # Create combined datasets (numerical + categorical)
        real_combined = np.hstack([real_num, real_cat]) if real_cat.shape[1] > 0 else real_num
        recon_combined = np.hstack([recon_num, recon_cat]) if recon_cat.shape[1] > 0 else recon_num
        
        # Create column names
        num_cols = [f'num_{i}' for i in range(real_num.shape[1])]
        cat_cols = [f'cat_{i}' for i in range(real_cat.shape[1])] if real_cat.shape[1] > 0 else []
        all_cols = num_cols + cat_cols
        
        # Convert to DataFrames for correlation computation
        real_df = pd.DataFrame(real_combined, columns=all_cols)
        recon_df = pd.DataFrame(recon_combined, columns=all_cols)
        
        # Compute correlations
        real_corr = real_df.corr(method='pearson')
        recon_corr = recon_df.corr(method='pearson')
        
        # Create correlation plots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 1: Real data correlation
        sns.heatmap(real_corr, annot=False, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
        axes[0].set_title(f'Real Data Correlation (Epoch {epoch+1})')
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Features')
        
        # Plot 2: Reconstructed data correlation
        sns.heatmap(recon_corr, annot=False, cmap='coolwarm', center=0,
                   square=True, ax=axes[1], cbar_kws={'shrink': 0.8})
        axes[1].set_title(f'Reconstructed Data Correlation (Epoch {epoch+1})')
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Features')
        
        # Plot 3: Difference between correlations
        corr_diff = recon_corr - real_corr
        sns.heatmap(corr_diff, annot=False, cmap='RdBu_r', center=0,
                   square=True, ax=axes[2], cbar_kws={'shrink': 0.8})
        axes[2].set_title(f'Correlation Difference (Recon - Real)')
        axes[2].set_xlabel('Features')
        axes[2].set_ylabel('Features')
        
        plt.tight_layout()
        
        # Log to TensorBoard
        self.writer.add_figure('Correlation_Analysis', fig, epoch)
        
        plt.close(fig)
        
        print("Correlation analysis logged - heatmaps saved to TensorBoard")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_losses = {'total_loss': 0, 'mse_loss': 0, 'ce_loss': 0, 'kl_loss': 0, 'accuracy': 0}
        num_batches = 0
        
        # Use pre-computed beta value
        beta = self.beta_schedule[epoch]
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (x_num, x_cat) in enumerate(pbar):
            x_num, x_cat = x_num.to(self.device), x_cat.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_num, recon_cat, mu, logvar = self.model(x_num, x_cat)
            
            # Compute loss
            losses = self.compute_vae_loss(x_num, x_cat, recon_num, recon_cat, mu, logvar, beta)
            
            # Backward pass
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                total_losses[key] += value.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'MSE': f"{losses['mse_loss'].item():.4f}",
                'CE': f"{losses['ce_loss'].item():.4f}",
                'KL': f"{losses['kl_loss'].item():.4f}",
                'Beta': f"{beta:.6f}",
                'Acc': f"{losses['accuracy'].item():.3f}"
            })
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        # Log to TensorBoard
        for key, value in avg_losses.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        self.writer.add_scalar('Train/beta', beta, epoch)
        
        return avg_losses
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_losses = {'total_loss': 0, 'mse_loss': 0, 'ce_loss': 0, 'kl_loss': 0, 'accuracy': 0}
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for x_num, x_cat in pbar:
                x_num, x_cat = x_num.to(self.device), x_cat.to(self.device)
                
                # Forward pass
                recon_num, recon_cat, mu, logvar = self.model(x_num, x_cat)
                
                # Compute loss (use fixed beta for validation)
                losses = self.compute_vae_loss(x_num, x_cat, recon_num, recon_cat, mu, logvar, beta=1.0)
                
                # Accumulate losses
                for key, value in losses.items():
                    total_losses[key] += value.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.4f}",
                    'Acc': f"{losses['accuracy'].item():.3f}"
                })
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        # Log to TensorBoard
        for key, value in avg_losses.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)

        # Generate synthetic data and log correlation at each epoch
        print(f"Generating synthetic data correlation analysis at epoch {epoch+1}...")
        self.generate_synthetic_data_and_log_correlation(epoch)

        # Correlation analysis logging (real vs reconstructed)
        if (epoch + 1) % self.correlation_log_freq == 0:
            print(f"Computing real vs reconstructed correlation analysis at epoch {epoch+1}...")
            self.log_correlation_analysis(epoch)
        
        return avg_losses
    
    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        num_epochs = self.config.get('training', {}).get('num_epochs', 100)
        patience = self.config.get('training', {}).get('early_stopping_patience', 20)
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)
            self.train_losses.append(train_losses['total_loss'])
            
            # Validate
            val_losses = self.validate_epoch(epoch)
            self.val_losses.append(val_losses['total_loss'])
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total_loss'])
            
            # Early stopping check
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
            

            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_losses['total_loss']:.4f}, "
                  f"Val Loss: {val_losses['total_loss']:.4f}, "
                  f"Val Acc: {val_losses['accuracy']:.3f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.writer.close()
        print("Training completed!")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path}")
