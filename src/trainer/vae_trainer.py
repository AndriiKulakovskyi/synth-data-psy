import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from typing import Tuple, Dict, Optional, List, Any
from datetime import datetime

from src.ldm.vae.model import Model_VAE, Encoder_model, Decoder_model
from src.utils.config import Config


class VAETrainer:
    """
    Trainer class for VAE model.
    
    Implements training and evaluation loops, model saving, and early stopping.
    """
    
    def __init__(self, config: Config, logger: logging.Logger):
        """
        Initialize VAE trainer.
        
        Args:
            config: Configuration object with training parameters
            logger: Logger instance for logging training progress
        """
        self.config = config
        self.logger = logger
        
        # Set device
        if config.device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(config.device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # Setup paths
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                           self.config.paths.checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        self.model_path = os.path.join(self.checkpoint_dir, config.paths.model_filename)
        self.encoder_path = os.path.join(self.checkpoint_dir, config.paths.encoder_filename)
        self.decoder_path = os.path.join(self.checkpoint_dir, config.paths.decoder_filename)
        self.embeddings_path = os.path.join(self.checkpoint_dir, config.paths.embeddings_filename)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Beta for KL annealing
        self.beta = config.training.beta_max
        
        # Model, optimizer and scheduler will be initialized in setup_model method
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Initialize TensorBoard
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'runs',
            f'vae_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logger.info(f'TensorBoard logs will be saved to: {log_dir}')
        self.global_step = 0
        
    def setup_model(self, d_numerical: int, categories: List[int]) -> None:
        """
        Initialize the VAE model, optimizer and scheduler.
        
        Args:
            d_numerical: Number of numerical features
            categories: List of category cardinalities for categorical features
        """
        self.model = Model_VAE(
            num_layers=self.config.model.num_layers,
            d_numerical=d_numerical,
            categories=categories,
            d_token=self.config.model.d_token,
            n_head=self.config.model.n_head,
            factor=self.config.model.factor,
            bias=self.config.model.token_bias
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.training.scheduler_factor,
            patience=self.config.training.scheduler_patience
        )
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model initialized with {total_params} trainable parameters")
        
        # Log model architecture to TensorBoard
        if hasattr(self, 'writer') and self.writer is not None:
            # Create dummy input for graph visualization
            dummy_num = torch.randn(1, d_numerical, device=self.device)
            dummy_cat = torch.zeros(1, len(categories), dtype=torch.long, device=self.device)
            try:
                self.writer.add_graph(self.model, (dummy_num, dummy_cat))
            except Exception as e:
                self.logger.warning(f"Failed to add model graph to TensorBoard: {e}")
        
    def compute_loss(self, X_num: torch.Tensor, X_cat: torch.Tensor, 
                   Recon_X_num: torch.Tensor, Recon_X_cat: List[torch.Tensor], 
                   mu_z: torch.Tensor, logvar_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss components: MSE loss, cross-entropy loss, KL divergence, and accuracy.
        
        Args:
            X_num: Input numerical features
            X_cat: Input categorical features
            Recon_X_num: Reconstructed numerical features
            Recon_X_cat: Reconstructed categorical features (list of tensors)
            mu_z: Mean of latent distribution
            logvar_z: Log variance of latent distribution
            
        Returns:
            Tuple of (MSE loss, CE loss, KL divergence, accuracy)
        """
        ce_loss_fn = nn.CrossEntropyLoss()
        mse_loss = (X_num - Recon_X_num).pow(2).mean()
        ce_loss = 0
        acc = 0
        total_num = 0

        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
                x_hat = x_cat.argmax(dim=-1)
                acc += (x_hat == X_cat[:, idx]).float().sum()
                total_num += x_hat.shape[0]
        
        ce_loss /= (idx + 1) if idx >= 0 else 1
        acc /= total_num if total_num > 0 else 1

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
        
        return mse_loss, ce_loss, kl_loss, acc
        
    def train_step(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Execute one training epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_acc = 0.0
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {self.current_epoch+1}/{self.config.training.num_epochs}")
        
        for batch_num, batch_cat in pbar:
            self.optimizer.zero_grad()
            
            batch_num = batch_num.to(self.device)
            batch_cat = batch_cat.to(self.device)
            
            # Forward pass
            Recon_X_num, Recon_X_cat, mu_z, logvar_z = self.model(batch_num, batch_cat)
            
            # Compute loss
            mse_loss, ce_loss, kl_loss, acc = self.compute_loss(
                batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z
            )
            
            # Total loss
            loss = mse_loss + ce_loss + self.beta * kl_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            batch_size = batch_num.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_mse_loss += mse_loss.item() * batch_size
            epoch_ce_loss += ce_loss.item() * batch_size
            epoch_kl_loss += kl_loss.item() * batch_size
            epoch_acc += acc.item() * batch_size
            
            # Log batch metrics to TensorBoard
            if hasattr(self, 'writer') and self.writer is not None:
                self.writer.add_scalar('train/batch/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/batch/mse_loss', mse_loss.item(), self.global_step)
                self.writer.add_scalar('train/batch/ce_loss', ce_loss.item(), self.global_step)
                self.writer.add_scalar('train/batch/kl_loss', kl_loss.item(), self.global_step)
                self.writer.add_scalar('train/batch/accuracy', acc.item(), self.global_step)
                self.writer.add_scalar('train/batch/beta', self.beta, self.global_step)
                self.writer.add_scalar('train/batch/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                # Log histograms of parameters and gradients
                if self.global_step % 100 == 0:  # Don't log too frequently to save space
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            self.writer.add_histogram(f'params/{name}', param, self.global_step)
                            self.writer.add_histogram(f'grads/{name}', param.grad, self.global_step)
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'mse': mse_loss.item(),
                'ce': ce_loss.item(),
                'kl': kl_loss.item(),
                'acc': acc.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
        # Normalize metrics by dataset size
        num_samples = len(train_loader.dataset)
        epoch_loss /= num_samples
        epoch_mse_loss /= num_samples
        epoch_ce_loss /= num_samples
        epoch_kl_loss /= num_samples
        epoch_acc /= num_samples
        
        return {
            'loss': epoch_loss,
            'mse_loss': epoch_mse_loss,
            'ce_loss': epoch_ce_loss,
            'kl_loss': epoch_kl_loss,
            'accuracy': epoch_acc
        }
        
    def val_step(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Execute one validation epoch.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_acc = 0.0
        
        with torch.no_grad():
            for batch_num, batch_cat in val_loader:
                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)
                
                # Forward pass
                Recon_X_num, Recon_X_cat, mu_z, logvar_z = self.model(batch_num, batch_cat)
                
                # Compute loss
                mse_loss, ce_loss, kl_loss, acc = self.compute_loss(
                    batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z
                )
                
                # Total loss
                loss = mse_loss + ce_loss + self.beta * kl_loss
                
                # Update metrics
                batch_size = batch_num.shape[0]
                epoch_loss += loss.item() * batch_size
                epoch_mse_loss += mse_loss.item() * batch_size
                epoch_ce_loss += ce_loss.item() * batch_size
                epoch_kl_loss += kl_loss.item() * batch_size
                epoch_acc += acc.item() * batch_size
                
        # Normalize metrics by dataset size
        num_samples = len(val_loader.dataset)
        epoch_loss /= num_samples
        epoch_mse_loss /= num_samples
        epoch_ce_loss /= num_samples
        epoch_kl_loss /= num_samples
        epoch_acc /= num_samples
        
        return {
            'loss': epoch_loss,
            'mse_loss': epoch_mse_loss,
            'ce_loss': epoch_ce_loss,
            'kl_loss': epoch_kl_loss,
            'accuracy': epoch_acc
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the VAE model for the specified number of epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary with training history
        """
        self.logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        start_time = time.time()
        
        # Log hyperparameters to TensorBoard
        if hasattr(self, 'writer') and self.writer is not None:
            hparams = {
                'learning_rate': self.config.training.learning_rate,
                'weight_decay': self.config.training.weight_decay,
                'batch_size': self.config.training.batch_size,
                'num_epochs': self.config.training.num_epochs,
                'beta_max': self.config.training.beta_max,
                'd_token': self.config.model.d_token,
                'n_head': self.config.model.n_head,
                'num_layers': self.config.model.num_layers,
            }
            self.writer.add_hparams(hparams, {})
        
        history = {
            'train_loss': [],
            'train_mse_loss': [],
            'train_ce_loss': [],
            'train_kl_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_mse_loss': [],
            'val_ce_loss': [],
            'val_kl_loss': [],
            'val_accuracy': [],
            'beta': []
        }
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Training step
            train_metrics = self.train_step(train_loader)
            
            # Validation step
            val_metrics = self.val_step(val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_mse_loss'].append(train_metrics['mse_loss'])
            history['train_ce_loss'].append(train_metrics['ce_loss'])
            history['train_kl_loss'].append(train_metrics['kl_loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            history['val_loss'].append(val_metrics['loss'])
            history['val_mse_loss'].append(val_metrics['mse_loss'])
            history['val_ce_loss'].append(val_metrics['ce_loss'])
            history['val_kl_loss'].append(val_metrics['kl_loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            history['beta'].append(self.beta)
            
            # Log metrics to console
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.training.num_epochs} - "
                f"beta={self.beta:.6f} - "
                f"train_loss={train_metrics['loss']:.6f} - "
                f"train_mse={train_metrics['mse_loss']:.6f} - "
                f"train_ce={train_metrics['ce_loss']:.6f} - "
                f"train_kl={train_metrics['kl_loss']:.6f} - "
                f"train_acc={train_metrics['accuracy']:.6f} - "
                f"val_loss={val_metrics['loss']:.6f} - "
                f"val_mse={val_metrics['mse_loss']:.6f} - "
                f"val_ce={val_metrics['ce_loss']:.6f} - "
                f"val_kl={val_metrics['kl_loss']:.6f} - "
                f"val_acc={val_metrics['accuracy']:.6f}"
            )
            
            # Log metrics to TensorBoard
            if hasattr(self, 'writer') and self.writer is not None:
                # Log training metrics
                self.writer.add_scalar('epoch/train/loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('epoch/train/mse_loss', train_metrics['mse_loss'], epoch)
                self.writer.add_scalar('epoch/train/ce_loss', train_metrics['ce_loss'], epoch)
                self.writer.add_scalar('epoch/train/kl_loss', train_metrics['kl_loss'], epoch)
                self.writer.add_scalar('epoch/train/accuracy', train_metrics['accuracy'], epoch)
                
                # Log validation metrics
                self.writer.add_scalar('epoch/val/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('epoch/val/mse_loss', val_metrics['mse_loss'], epoch)
                self.writer.add_scalar('epoch/val/ce_loss', val_metrics['ce_loss'], epoch)
                self.writer.add_scalar('epoch/val/kl_loss', val_metrics['kl_loss'], epoch)
                self.writer.add_scalar('epoch/val/accuracy', val_metrics['accuracy'], epoch)
                
                # Log learning rate and beta
                self.writer.add_scalar('hyperparams/lr', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('hyperparams/beta', self.beta, epoch)
                
                # Log histograms of model parameters
                if epoch % 5 == 0:  # Don't log too frequently to save space
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            self.writer.add_histogram(f'params/{name}', param, epoch)
                            self.writer.add_histogram(f'grads/{name}', param.grad, epoch)
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_metrics['loss'])
            
            # Check for early stopping and model improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_model()
                self.logger.info(f"Model improved, saved checkpoint at epoch {epoch+1}")
                
                # Log best model metrics to TensorBoard
                if hasattr(self, 'writer') and self.writer is not None:
                    self.writer.add_scalar('best/val_loss', self.best_val_loss, epoch)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    # Adjust beta for KL annealing
                    if self.beta > self.config.training.beta_min:
                        self.beta *= self.config.training.beta_decay_factor
                        self.logger.info(f"Adjusted beta to {self.beta:.6f}")
                        self.patience_counter = 0
        
        training_time = (time.time() - start_time) / 60
        self.logger.info(f"Training completed in {training_time:.2f} minutes")
        
        # Save embeddings after training
        self.save_embeddings(train_loader)
        
        # Close TensorBoard writer
        if hasattr(self, 'writer') and self.writer is not None:
            # Log final metrics
            self.writer.add_hparams(
                {
                    'learning_rate': self.config.training.learning_rate,
                    'weight_decay': self.config.training.weight_decay,
                    'batch_size': self.config.training.batch_size,
                    'num_epochs': self.config.training.num_epochs,
                    'beta_max': self.config.training.beta_max,
                },
                {
                    'hparam/best_val_loss': self.best_val_loss,
                    'hparam/final_train_loss': history['train_loss'][-1],
                    'hparam/final_val_loss': history['val_loss'][-1],
                    'hparam/training_time': training_time * 60,  # Convert to seconds
                },
                run_name='.'  # This ensures the hparams are logged to the current run
            )
            self.writer.flush()
            self.writer.close()
        
        return history
    
    def save_model(self) -> None:
        """Save the current model, encoder and decoder"""
        torch.save(self.model.state_dict(), self.model_path)
        
        # Create and save encoder and decoder models
        pre_encoder = Encoder_model(
            num_layers=self.config.model.num_layers,
            d_numerical=self.model.VAE.d_numerical,
            categories=self.model.VAE.categories,
            d_token=self.config.model.d_token,
            n_head=self.config.model.n_head,
            factor=self.config.model.factor
        ).to(self.device)
        
        pre_decoder = Decoder_model(
            num_layers=self.config.model.num_layers,
            d_numerical=self.model.VAE.d_numerical,
            categories=self.model.VAE.categories,
            d_token=self.config.model.d_token,
            n_head=self.config.model.n_head,
            factor=self.config.model.factor
        ).to(self.device)
        
        pre_encoder.load_weights(self.model)
        pre_decoder.load_weights(self.model)
        
        torch.save(pre_encoder.state_dict(), self.encoder_path)
        torch.save(pre_decoder.state_dict(), self.decoder_path)
        
        self.logger.info(f"Saved model to {self.model_path}")
        self.logger.info(f"Saved encoder to {self.encoder_path}")
        self.logger.info(f"Saved decoder to {self.decoder_path}")
    
    def save_embeddings(self, data_loader: DataLoader) -> None:
        """
        Generate and save latent embeddings.
        
        Args:
            data_loader: DataLoader for data to encode
        """
        self.logger.info("Generating and saving latent embeddings")
        
        # Create encoder model
        pre_encoder = Encoder_model(
            num_layers=self.config.model.num_layers,
            d_numerical=self.model.VAE.d_numerical,
            categories=self.model.VAE.categories,
            d_token=self.config.model.d_token,
            n_head=self.config.model.n_head,
            factor=self.config.model.factor
        ).to(self.device)
        
        pre_encoder.load_weights(self.model)
        
        # Generate embeddings
        all_embeddings = []
        pre_encoder.eval()
        
        with torch.no_grad():
            for batch_num, batch_cat in data_loader:
                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)
                
                embeddings = pre_encoder(batch_num, batch_cat).cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Concatenate and save embeddings
        all_embeddings = np.vstack(all_embeddings)
        np.save(self.embeddings_path, all_embeddings)
        
        self.logger.info(f"Saved embeddings with shape {all_embeddings.shape} to {self.embeddings_path}") 