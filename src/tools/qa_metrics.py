#!/usr/bin/env python3
"""
Simplified Quality Assessment (QA) metrics for comparing original and synthetic data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from scipy.stats import wasserstein_distance
import warnings
import torch


class QualityAssessment:
    """Simplified Quality Assessment pipeline for comparing original vs synthetic data."""
    
    def __init__(self, 
                 original_num: np.ndarray, 
                 original_cat: np.ndarray,
                 synthetic_num: np.ndarray, 
                 synthetic_cat: np.ndarray,
                 mapping: Dict[str, Any]):
        """
        Initialize QA pipeline with original and synthetic data.
        
        Args:
            original_num: Original numerical data matrix
            original_cat: Original categorical data matrix
            synthetic_num: Synthetic numerical data matrix
            synthetic_cat: Synthetic categorical data matrix
            mapping: Column mapping from split_numerical_categorical
        """
        # Convert tensors to numpy if needed
        self.original_num = self._to_numpy(original_num)
        self.original_cat = self._to_numpy(original_cat)
        self.synthetic_num = self._to_numpy(synthetic_num)
        self.synthetic_cat = self._to_numpy(synthetic_cat)
        
        self.mapping = mapping
        self.num_cols = list(mapping['numerical'].values())
        self.cat_cols = list(mapping['categorical'].values())
        
        # Combine matrices for easier processing
        self.original_data = np.hstack([self.original_num, self.original_cat]) if self.original_cat.size > 0 else self.original_num
        self.synthetic_data = np.hstack([self.synthetic_num, self.synthetic_cat]) if self.synthetic_cat.size > 0 else self.synthetic_num
        self.all_cols = self.num_cols + self.cat_cols
        
        self.n_num_features = len(self.num_cols)
        self.n_cat_features = len(self.cat_cols)
        
        self.results = {}
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
        """Convert tensor or list of tensors to numpy array if needed."""
        if isinstance(data, list):
            # Handle list of tensors (common for categorical data from VAE)
            data_list = []
            for item in data:
                if isinstance(item, torch.Tensor):
                    # Get argmax if it's a probability distribution
                    if item.dim() > 1 and item.shape[-1] > 1:
                        item_data = item.argmax(dim=-1).cpu().numpy()
                    else:
                        item_data = item.cpu().numpy()
                else:
                    item_data = np.array(item)
                data_list.append(item_data)
            
            # Stack into single array
            return np.column_stack(data_list) if data_list else np.empty((0, 0))
        elif isinstance(data, torch.Tensor):
            # Handle single tensor
            if data.dim() > 1 and data.shape[-1] > 1:
                # Assume it's categorical probabilities, take argmax
                return data.argmax(dim=-1).cpu().numpy()
            else:
                return data.cpu().numpy()
        else:
            # Assume it's already numpy
            return data
    
    def compute_global_correlations(self) -> Dict[str, float]:
        """
        Compute global correlation metrics between original and synthetic datasets.
        Uses horizontally stacked numerical and categorical data for comprehensive correlation analysis.
        """
        correlations = {}
        
        # Horizontally stack numerical and categorical matrices
        if self.n_num_features > 0 and self.n_cat_features > 0:
            # Both numerical and categorical features present
            original_stacked = np.hstack([self.original_num, self.original_cat])
            synthetic_stacked = np.hstack([self.synthetic_num, self.synthetic_cat])
            all_feature_names = self.num_cols + self.cat_cols
        elif self.n_num_features > 0:
            # Only numerical features
            original_stacked = self.original_num
            synthetic_stacked = self.synthetic_num
            all_feature_names = self.num_cols
        elif self.n_cat_features > 0:
            # Only categorical features
            original_stacked = self.original_cat
            synthetic_stacked = self.synthetic_cat
            all_feature_names = self.cat_cols
        else:
            # No features to process
            print("Warning: No features available for correlation computation")
            self.results['global_correlations'] = correlations
            return correlations
        
        # Convert to DataFrames
        original_df = pd.DataFrame(original_stacked, columns=all_feature_names)
        synthetic_df = pd.DataFrame(synthetic_stacked, columns=all_feature_names)
        
        # Compute Pearson correlations
        try:
            original_corr = original_df.corr(method='pearson')
            synthetic_corr = synthetic_df.corr(method='pearson')
            
            # Compute absolute difference between correlation matrices
            corr_diff = np.abs(original_corr.values - synthetic_corr.values)
            
            # Compute summary metrics
            # Mean absolute difference across all correlations
            correlations['mean_abs_corr_diff'] = np.mean(corr_diff)
            
            # Mean absolute difference excluding diagonal (self-correlations)
            mask = ~np.eye(corr_diff.shape[0], dtype=bool)
            correlations['mean_abs_corr_diff_offdiag'] = np.mean(corr_diff[mask])
            
            # Maximum absolute difference
            correlations['max_abs_corr_diff'] = np.max(corr_diff[mask])
            
            # Root mean squared error of correlations
            correlations['correlation_rmse'] = np.sqrt(np.mean((original_corr.values[mask] - synthetic_corr.values[mask])**2))
            
            # Correlation between the two correlation matrices (flattened upper triangular)
            mask_upper = np.triu(np.ones_like(original_corr, dtype=bool), k=1)
            orig_corr_flat = original_corr.values[mask_upper]
            synth_corr_flat = synthetic_corr.values[mask_upper]
            
            if len(orig_corr_flat) > 1:
                correlations['correlation_of_correlations'] = np.corrcoef(orig_corr_flat, synth_corr_flat)[0, 1]
            else:
                correlations['correlation_of_correlations'] = 1.0  # Perfect correlation for single feature
            
            # Store the correlation matrices and difference for visualization
            correlations['original_corr_matrix'] = original_corr
            correlations['synthetic_corr_matrix'] = synthetic_corr
            correlations['corr_diff_matrix'] = pd.DataFrame(corr_diff, 
                                                          index=original_corr.index, 
                                                          columns=original_corr.columns)
            
            # Fraction of correlation pairs with small differences (< 0.1)
            correlations['frac_small_diff'] = np.mean(corr_diff[mask] < 0.1)
            
        except Exception as e:
            print(f"Warning: Error computing correlations: {e}")
            correlations['error'] = str(e)
        
        self.results['global_correlations'] = correlations
        return correlations
    
    def compute_univariate_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute univariate distribution comparison metrics.
        """
        univariate_metrics = {}
        
        # Numerical features
        for i, col_name in enumerate(self.num_cols):
            orig_vals = self.original_num[:, i]
            synth_vals = self.synthetic_num[:, i]
            
            metrics = {}
            
            # Statistical tests
            try:
                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = stats.ks_2samp(orig_vals, synth_vals)
                metrics['ks_statistic'] = ks_stat
                metrics['ks_pvalue'] = ks_pval
                
                # Wasserstein distance
                metrics['wasserstein_distance'] = wasserstein_distance(orig_vals, synth_vals)
                
                # Basic statistics comparison
                metrics['mean_diff'] = abs(np.mean(orig_vals) - np.mean(synth_vals))
                metrics['std_diff'] = abs(np.std(orig_vals) - np.std(synth_vals))
                metrics['median_diff'] = abs(np.median(orig_vals) - np.median(synth_vals))
                
            except Exception as e:
                warnings.warn(f"Error computing metrics for {col_name}: {e}")
                metrics['error'] = str(e)
            
            univariate_metrics[f"num_{col_name}"] = metrics
        
        # Categorical features
        for i, col_name in enumerate(self.cat_cols):
            orig_vals = self.original_cat[:, i]
            synth_vals = self.synthetic_cat[:, i]
            
            metrics = {}
            
            try:
                # Get unique values and their counts
                orig_unique, orig_counts = np.unique(orig_vals, return_counts=True)
                synth_unique, synth_counts = np.unique(synth_vals, return_counts=True)
                
                # Align the categories
                all_categories = np.union1d(orig_unique, synth_unique)
                orig_probs = np.zeros(len(all_categories))
                synth_probs = np.zeros(len(all_categories))
                
                for j, cat in enumerate(all_categories):
                    orig_idx = np.where(orig_unique == cat)[0]
                    synth_idx = np.where(synth_unique == cat)[0]
                    
                    orig_probs[j] = orig_counts[orig_idx[0]] / len(orig_vals) if len(orig_idx) > 0 else 0
                    synth_probs[j] = synth_counts[synth_idx[0]] / len(synth_vals) if len(synth_idx) > 0 else 0
                
                # Total Variation Distance
                tvd = 0.5 * np.sum(np.abs(orig_probs - synth_probs))
                metrics['tvd'] = tvd
                metrics['accuracy'] = 1.0 - tvd  # Accuracy as 100% - TVD
                
                # Chi-square test
                try:
                    chi2, chi2_pval = stats.chisquare(synth_counts, orig_counts)
                    metrics['chi2_statistic'] = chi2
                    metrics['chi2_pvalue'] = chi2_pval
                except:
                    pass
                
            except Exception as e:
                warnings.warn(f"Error computing metrics for categorical {col_name}: {e}")
                metrics['error'] = str(e)
            
            univariate_metrics[f"cat_{col_name}"] = metrics
        
        self.results['univariate_metrics'] = univariate_metrics
        return univariate_metrics
    
    def run_full_assessment(self) -> Dict[str, Any]:
        """
        Run the simplified quality assessment pipeline.
        
        Returns:
            Complete results dictionary
        """
        print("Computing global correlations...")
        self.compute_global_correlations()
        
        print("Computing univariate metrics...")
        self.compute_univariate_metrics()
        
        return self.results
    
    def print_summary(self):
        """Print a summary of the quality assessment results."""
        print("\n" + "="*80)
        print("SIMPLIFIED QUALITY ASSESSMENT SUMMARY")
        print("="*80)
        
        # Global correlations
        if 'global_correlations' in self.results:
            print("\nGlobal Correlations:")
            corr_results = self.results['global_correlations']
            print(f"  Mean absolute correlation difference: {corr_results.get('mean_abs_corr_diff_offdiag', 'N/A'):.4f}")
            print(f"  Maximum absolute correlation difference: {corr_results.get('max_abs_corr_diff', 'N/A'):.4f}")
            print(f"  Correlation RMSE: {corr_results.get('correlation_rmse', 'N/A'):.4f}")
            print(f"  Correlation of correlations: {corr_results.get('correlation_of_correlations', 'N/A'):.4f}")
            print(f"  Fraction with small differences (<0.1): {corr_results.get('frac_small_diff', 'N/A'):.4f}")
        
        # Univariate analysis summary
        if 'univariate_metrics' in self.results:
            print("\nUnivariate Analysis Summary:")
            
            # Count features by type
            num_features = sum(1 for k in self.results['univariate_metrics'].keys() if k.startswith('num_'))
            cat_features = sum(1 for k in self.results['univariate_metrics'].keys() if k.startswith('cat_'))
            print(f"  Analyzed features: {num_features} numerical, {cat_features} categorical")
            
            # Average metrics
            ks_stats = []
            tvds = []
            for feature, metrics in self.results['univariate_metrics'].items():
                if 'ks_statistic' in metrics:
                    ks_stats.append(metrics['ks_statistic'])
                if 'tvd' in metrics:
                    tvds.append(metrics['tvd'])
            
            if ks_stats:
                print(f"  Average KS statistic (numerical): {np.mean(ks_stats):.4f}")
            if tvds:
                print(f"  Average TVD (categorical): {np.mean(tvds):.4f}")
        
        print("\n" + "="*80)
        
        return self.results 