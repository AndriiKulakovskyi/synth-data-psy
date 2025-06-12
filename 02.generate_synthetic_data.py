#!/usr/bin/env python3
"""
Synthetic Data Generation Script

This script:
1. Loads a trained VAE model
2. Generates synthetic data samples
3. Creates correlation heatmaps comparing original vs synthetic data
4. Transforms synthetic data back to original scales
5. Saves results with proper column ordering

Usage:
    python generate_synthetic_data.py --checkpoint ckpt/best_model.pt --data_folder FACE/processed --num_samples 1000
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.ldm.vae.model import VAE
from src.data.dataset import IntegratedTabularDataset
from src.data.processing import DataTransformer

warnings.filterwarnings('ignore')

class SyntheticDataGenerator:
    """Generator for synthetic tabular data using trained VAE."""
    
    def __init__(self, checkpoint_path: str, data_folder: str, device: str = 'auto'):
        self.checkpoint_path = Path(checkpoint_path)
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
        
        # Initialize containers
        self.model = None
        self.config = None
        self.categorical_info = None
        self.ml_transformer = None
        self.original_numerical_df = None
        self.original_categorical_df = None
        self.transformed_numerical_df = None
        self.transformed_categorical_df = None
        
    def load_model_and_data(self) -> None:
        """Load the trained VAE model and associated data."""
        # Load checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Load preprocessed data that was used for training
        print("Loading preprocessed data...")
        numerical_path = self.data_folder / 'numerical_data_imputed.csv'
        categorical_path = self.data_folder / 'categorical_data_imputed.csv'
        categorical_info_path = self.data_folder / 'categorical_info.csv'
        
        if not all(p.exists() for p in [numerical_path, categorical_path, categorical_info_path]):
            raise FileNotFoundError(
                f"Missing preprocessed data files in {self.data_folder}. "
                "Required: numerical_data_imputed.csv, categorical_data_imputed.csv, categorical_info.csv"
            )
        
        # Load transformed data (this is what goes to the VAE)
        self.transformed_numerical_df = pd.read_csv(numerical_path, sep=';')
        self.transformed_categorical_df = pd.read_csv(categorical_path, sep=';')
        categorical_info_df = pd.read_csv(categorical_info_path, sep=';')
        
        print(f"Loaded transformed data shapes:")
        print(f"  Numerical: {self.transformed_numerical_df.shape}")
        print(f"  Categorical: {self.transformed_categorical_df.shape}")
        
        # Extract categorical information
        self.categorical_info = {}
        for _, row in categorical_info_df.iterrows():
            self.categorical_info[row['column_name']] = {
                'num_categories': row['num_categories'],
                'encoded_values': eval(row['encoded_values']) if isinstance(row['encoded_values'], str) else row['encoded_values']
            }
        
        # Load MLDataTransformer for inverse transformation
        transformer_path = self.data_folder / 'ml_transformer.pkl'
        if transformer_path.exists() and transformer_path.stat().st_size > 0:
            try:
                with open(transformer_path, 'rb') as f:
                    self.ml_transformer = pickle.load(f)
                print("Loaded MLDataTransformer for inverse transformation")
            except (pickle.UnpicklingError, EOFError, Exception) as e:
                print(f"Warning: Could not load MLDataTransformer ({e}). Will create a basic fallback.")
                self.ml_transformer = None
                self._create_fallback_transformer()
        else:
            print("Warning: MLDataTransformer not found or empty. Will create a basic fallback.")
            self.ml_transformer = None
            self._create_fallback_transformer()
        
        # Initialize and load VAE model
        categories = [info['num_categories'] for info in self.categorical_info.values()]
        
        self.model = VAE(
            num_layers=self.config['model']['num_layers'],
            d_numerical=self.transformed_numerical_df.shape[1],
            categories=categories,
            d_token=self.config['model']['d_token'],
            n_head=self.config['model']['n_head'],
            factor=self.config['model']['factor'],
            bias=self.config['model']['token_bias']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Loaded VAE model with {total_params:,} parameters")
    
    def _create_fallback_transformer(self) -> None:
        """Create a basic fallback transformer when the original can't be loaded."""
        try:
            # Try to load transform_info.json if available
            transform_info_path = self.data_folder / 'transform_info.json'
            if transform_info_path.exists():
                with open(transform_info_path, 'r') as f:
                    transform_info = json.load(f)
                
                print("Creating fallback transformer from transform_info.json")
                
                # Create a minimal transformer-like object
                class FallbackTransformer:
                    def __init__(self, transform_info, numerical_cols, categorical_cols):
                        self.transform_info = transform_info
                        self.numerical_columns = numerical_cols
                        self.categorical_columns = categorical_cols
                        self.is_fitted = True
                    
                    def inverse_transform(self, df):
                        """Basic inverse transformation without scalers/encoders."""
                        # For now, just return the data as-is with a warning
                        print("Warning: Using fallback transformer - data may not be in original scale")
                        return df
                
                # Extract column information
                numerical_cols = list(self.transformed_numerical_df.columns)
                categorical_cols = list(self.transformed_categorical_df.columns)
                
                self.ml_transformer = FallbackTransformer(transform_info, numerical_cols, categorical_cols)
                print("Created fallback transformer")
            else:
                print("Warning: No transformation info available. Synthetic data will be in transformed scale.")
                
        except Exception as e:
            print(f"Could not create fallback transformer: {e}")
        
    def generate_synthetic_data(self, num_samples: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate synthetic data samples from the trained VAE."""
        print(f"Generating {num_samples} synthetic samples...")
        
        with torch.no_grad():
            # Generate samples using the VAE's sample method
            synthetic_numerical, synthetic_categorical = self.model.sample(num_samples, self.device)
            
        print(f"Generated synthetic data shapes:")
        print(f"  Numerical: {synthetic_numerical.shape}")
        print(f"  Categorical: {len(synthetic_categorical)} features")
        
        return synthetic_numerical, synthetic_categorical
    
    def create_correlation_comparison(self, synthetic_numerical: torch.Tensor, 
                                    synthetic_categorical: List[torch.Tensor],
                                    save_path: str = None) -> None:
        """Create correlation heatmaps comparing original vs synthetic data."""
        print("Creating correlation comparison heatmaps...")
        
        # Prepare original data (transformed - what goes to encoder)
        original_num = self.transformed_numerical_df.values
        original_cat = self.transformed_categorical_df.values
        
        # Prepare synthetic data (from decoder before scaling back)
        synthetic_num = synthetic_numerical.cpu().numpy()
        
        # Convert categorical predictions to encoded values
        synthetic_cat_list = []
        for cat_logits in synthetic_categorical:
            if cat_logits is not None:
                pred = cat_logits.argmax(dim=-1).cpu().numpy()
                synthetic_cat_list.append(pred)
        
        synthetic_cat = np.column_stack(synthetic_cat_list) if synthetic_cat_list else np.empty((synthetic_num.shape[0], 0))
        
        # Combine numerical and categorical data
        original_combined = np.hstack([original_num, original_cat]) if original_cat.shape[1] > 0 else original_num
        synthetic_combined = np.hstack([synthetic_num, synthetic_cat]) if synthetic_cat.shape[1] > 0 else synthetic_num
        
        # Create column names
        num_cols = [f'num_{i}' for i in range(original_num.shape[1])]
        cat_cols = [f'cat_{i}' for i in range(original_cat.shape[1])] if original_cat.shape[1] > 0 else []
        all_cols = num_cols + cat_cols
        
        # Convert to DataFrames
        original_df = pd.DataFrame(original_combined, columns=all_cols)
        synthetic_df = pd.DataFrame(synthetic_combined, columns=all_cols)
        
        # Compute correlations
        original_corr = original_df.corr(method='pearson')
        synthetic_corr = synthetic_df.corr(method='pearson')
        
        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        # Plot 1: Original data correlation
        sns.heatmap(original_corr, annot=False, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
        axes[0].set_title('Original Data Correlation (Transformed)')
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Features')
        
        # Plot 2: Synthetic data correlation
        sns.heatmap(synthetic_corr, annot=False, cmap='coolwarm', center=0,
                   square=True, ax=axes[1], cbar_kws={'shrink': 0.8})
        axes[1].set_title('Synthetic Data Correlation')
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Features')
        
        # Plot 3: Difference between correlations
        corr_diff = synthetic_corr - original_corr
        sns.heatmap(corr_diff, annot=False, cmap='RdBu_r', center=0,
                   square=True, ax=axes[2], cbar_kws={'shrink': 0.8})
        axes[2].set_title('Correlation Difference (Synthetic - Original)')
        axes[2].set_xlabel('Features')
        axes[2].set_ylabel('Features')
        
        plt.tight_layout()
        
        if save_path:
            # Ensure the directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation comparison saved to: {save_path}")
        
        plt.show()
        
        # Compute and print correlation statistics
        print("\nCorrelation Analysis Results:")
        print(f"Original data correlation matrix shape: {original_corr.shape}")
        print(f"Synthetic data correlation matrix shape: {synthetic_corr.shape}")
        
        # Calculate correlation similarity metrics
        corr_diff_abs = np.abs(corr_diff.values)
        mean_abs_diff = np.nanmean(corr_diff_abs)
        max_abs_diff = np.nanmax(corr_diff_abs)
        
        print(f"Mean absolute correlation difference: {mean_abs_diff:.4f}")
        print(f"Maximum absolute correlation difference: {max_abs_diff:.4f}")
        
        # Calculate correlation between correlation matrices
        orig_corr_flat = original_corr.values.flatten()
        synth_corr_flat = synthetic_corr.values.flatten()
        
        # Remove NaN values and diagonal elements for correlation calculation
        mask = ~(np.isnan(orig_corr_flat) | np.isnan(synth_corr_flat))
        if mask.sum() > 1:
            corr_similarity = np.corrcoef(orig_corr_flat[mask], synth_corr_flat[mask])[0, 1]
            print(f"Correlation matrix similarity (Pearson): {corr_similarity:.4f}")
    
    def inverse_transform_synthetic_data(self, synthetic_numerical: torch.Tensor,
                                       synthetic_categorical: List[torch.Tensor]) -> pd.DataFrame:
        """Transform synthetic data back to original scales and format."""
        print("Transforming synthetic data back to original scales...")
        
        if self.ml_transformer is None:
            print("Warning: MLDataTransformer not available. Returning data in transformed scale.")
            # Create dataframe with transformed data
            synthetic_num_df = pd.DataFrame(
                synthetic_numerical.cpu().numpy(),
                columns=self.transformed_numerical_df.columns
            )
            
            # Convert categorical predictions
            synthetic_cat_data = {}
            for i, (col_name, cat_logits) in enumerate(zip(self.categorical_info.keys(), synthetic_categorical)):
                if cat_logits is not None:
                    pred = cat_logits.argmax(dim=-1).cpu().numpy()
                    synthetic_cat_data[col_name] = pred
            
            synthetic_cat_df = pd.DataFrame(synthetic_cat_data)
            
            # Combine dataframes
            synthetic_df = pd.concat([synthetic_num_df, synthetic_cat_df], axis=1)
            return synthetic_df
        
        # Create dataframe with synthetic data in transformed scale
        synthetic_num_df = pd.DataFrame(
            synthetic_numerical.cpu().numpy(),
            columns=self.transformed_numerical_df.columns
        )
        
        # Convert categorical predictions to encoded values
        synthetic_cat_data = {}
        for i, (col_name, cat_logits) in enumerate(zip(self.categorical_info.keys(), synthetic_categorical)):
            if cat_logits is not None:
                pred = cat_logits.argmax(dim=-1).cpu().numpy()
                synthetic_cat_data[col_name] = pred
        
        synthetic_cat_df = pd.DataFrame(synthetic_cat_data)
        
        # Combine dataframes
        synthetic_combined_df = pd.concat([synthetic_num_df, synthetic_cat_df], axis=1)
        
        # Apply inverse transformation
        synthetic_original_scale = self.ml_transformer.inverse_transform(synthetic_combined_df)
        
        # Ensure column order matches the original data order
        # Get original column order from transformer
        all_original_columns = self.ml_transformer.numerical_columns + self.ml_transformer.categorical_columns
        
        # Reorder columns to match original order
        synthetic_original_scale = synthetic_original_scale[all_original_columns]
        
        print(f"Synthetic data transformed to original scale: {synthetic_original_scale.shape}")
        
        return synthetic_original_scale
    
    def save_synthetic_data(self, synthetic_data_original: pd.DataFrame,
                           synthetic_numerical: torch.Tensor,
                           synthetic_categorical: List[torch.Tensor],
                           output_folder: str) -> None:
        """Save synthetic data in multiple formats."""
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save data in original scale
        original_scale_path = output_path / 'synthetic_data_original_scale.csv'
        synthetic_data_original.to_csv(original_scale_path, sep=';', index=False)
        print(f"Synthetic data (original scale) saved to: {original_scale_path}")
        
        # Save data in transformed scale
        synthetic_num_df = pd.DataFrame(
            synthetic_numerical.cpu().numpy(),
            columns=self.transformed_numerical_df.columns
        )
        
        synthetic_cat_data = {}
        for i, (col_name, cat_logits) in enumerate(zip(self.categorical_info.keys(), synthetic_categorical)):
            if cat_logits is not None:
                pred = cat_logits.argmax(dim=-1).cpu().numpy()
                synthetic_cat_data[col_name] = pred
        
        synthetic_cat_df = pd.DataFrame(synthetic_cat_data)
        synthetic_transformed_df = pd.concat([synthetic_num_df, synthetic_cat_df], axis=1)
        
        transformed_scale_path = output_path / 'synthetic_data_transformed_scale.csv'
        synthetic_transformed_df.to_csv(transformed_scale_path, sep=';', index=False)
        print(f"Synthetic data (transformed scale) saved to: {transformed_scale_path}")
        
        # Save generation metadata
        metadata = {
            'num_samples': len(synthetic_data_original),
            'num_features': len(synthetic_data_original.columns),
            'numerical_features': self.ml_transformer.numerical_columns if self.ml_transformer else [],
            'categorical_features': self.ml_transformer.categorical_columns if self.ml_transformer else [],
            'model_config': self.config,
            'categorical_info': self.categorical_info
        }
        
        metadata_path = output_path / 'generation_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Generation metadata saved to: {metadata_path}")
    
    def generate_and_save_all(self, num_samples: int, output_folder: str = 'synthetic_output') -> pd.DataFrame:
        """Complete pipeline: generate, analyze, transform, and save synthetic data."""
        # Load model and data
        self.load_model_and_data()
        
        # Generate synthetic data
        synthetic_numerical, synthetic_categorical = self.generate_synthetic_data(num_samples)
        
        # Create correlation comparison
        correlation_plot_path = Path(output_folder) / 'correlation_comparison.png'
        self.create_correlation_comparison(
            synthetic_numerical, synthetic_categorical, 
            save_path=str(correlation_plot_path)
        )
        
        # Transform back to original scale
        synthetic_data_original = self.inverse_transform_synthetic_data(
            synthetic_numerical, synthetic_categorical
        )
        
        # Save all data
        self.save_synthetic_data(
            synthetic_data_original, synthetic_numerical, synthetic_categorical, output_folder
        )
        
        print(f"\nSynthetic data generation completed!")
        print(f"Generated {num_samples} samples with {len(synthetic_data_original.columns)} features")
        print(f"All outputs saved to: {output_folder}")
        
        return synthetic_data_original


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data from trained VAE')
    parser.add_argument('--checkpoint', type=str, default='ckpt/best_model.pt',
                        help='Path to trained VAE checkpoint')
    parser.add_argument('--data_folder', type=str, default='FACE/processed',
                        help='Path to preprocessed data folder')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--output_folder', type=str, default='synthetic_output',
                        help='Output folder for synthetic data')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = SyntheticDataGenerator(
            checkpoint_path=args.checkpoint,
            data_folder=args.data_folder,
            device=args.device
        )
        
        # Generate synthetic data
        synthetic_data = generator.generate_and_save_all(
            num_samples=args.num_samples,
            output_folder=args.output_folder
        )
        
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Data folder: {args.data_folder}")
        print(f"Samples generated: {args.num_samples}")
        print(f"Features: {len(synthetic_data.columns)}")
        print(f"Output folder: {args.output_folder}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during synthetic data generation: {e}")
        raise


if __name__ == '__main__':
    main() 