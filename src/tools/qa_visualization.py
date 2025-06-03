#!/usr/bin/env python3
"""
Simplified visualization tools for Quality Assessment results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
from .qa_metrics import QualityAssessment


class QAVisualizer:
    """Simplified visualization tools for Quality Assessment results."""
    
    def __init__(self, qa_results: QualityAssessment):
        """
        Initialize visualizer with QA results.
        
        Args:
            qa_results: QualityAssessment instance with computed results
        """
        self.qa = qa_results
        self.results = qa_results.results
        
    def plot_correlation_comparison(self, figsize: Tuple[int, int] = (18, 6)) -> plt.Figure:
        """
        Plot correlation matrix comparison between original and synthetic data.
        Shows comprehensive correlations including both numerical and categorical features.
        """
        if 'global_correlations' not in self.results:
            print("No correlation results found. Run compute_global_correlations() first.")
            return None
            
        correlations = self.results['global_correlations']
        
        if 'original_corr_matrix' not in correlations:
            print("Skipping correlation plots: No correlation matrices available")
            return None
            
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        original_corr = correlations['original_corr_matrix']
        synthetic_corr = correlations['synthetic_corr_matrix']
        corr_diff = correlations['corr_diff_matrix']
        
        # Define color maps and formatting
        corr_cmap = 'coolwarm'
        diff_cmap = 'Reds'
        
        # Determine annotation settings based on matrix size
        n_features = len(original_corr)
        annotate = n_features <= 15  # Only annotate if not too many features
        annot_fontsize = max(6, 12 - n_features // 3)  # Smaller font for larger matrices
        
        # Original correlation matrix
        sns.heatmap(original_corr, annot=annotate, cmap=corr_cmap, center=0, 
                   vmin=-1, vmax=1, square=True, ax=axes[0], 
                   annot_kws={'size': annot_fontsize}, cbar_kws={'shrink': 0.8})
        axes[0].set_title('Original Data Correlations\n(All Features)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')
        
        # Synthetic correlation matrix
        sns.heatmap(synthetic_corr, annot=annotate, cmap=corr_cmap, center=0,
                   vmin=-1, vmax=1, square=True, ax=axes[1],
                   annot_kws={'size': annot_fontsize}, cbar_kws={'shrink': 0.8})
        axes[1].set_title('Synthetic Data Correlations\n(All Features)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('')
        axes[1].set_ylabel('')
        
        # Absolute difference matrix
        max_diff = corr_diff.values.max()
        sns.heatmap(corr_diff, annot=annotate, cmap=diff_cmap, 
                   vmin=0, vmax=max_diff, square=True, ax=axes[2],
                   annot_kws={'size': annot_fontsize}, cbar_kws={'shrink': 0.8})
        axes[2].set_title(f'Absolute Correlation Differences\n|Synthetic - Original|', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('')
        axes[2].set_ylabel('')
        
        # Add summary statistics as text
        if 'mean_abs_corr_diff_offdiag' in correlations:
            stats_text = (
                f"Mean Abs Diff: {correlations['mean_abs_corr_diff_offdiag']:.3f}\n"
                f"Max Abs Diff: {correlations['max_abs_corr_diff']:.3f}\n"
                f"Correlation RMSE: {correlations['correlation_rmse']:.3f}\n"
                f"Corr of Corrs: {correlations['correlation_of_correlations']:.3f}"
            )
            
            # Add text box with statistics
            axes[2].text(1.15, 0.5, stats_text, transform=axes[2].transAxes, 
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Improve layout and spacing
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=8)
            # Rotate labels if many features
            if n_features > 10:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.suptitle(f'Correlation Analysis - {n_features} Features ({len(self.qa.num_cols)} Numerical, {len(self.qa.cat_cols)} Categorical)', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return fig
    
    def plot_univariate_distributions(self, max_plots: Optional[int] = None, figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot univariate distributions for numerical and categorical features.
        Shows all features with superimposed original and synthetic distributions.
        
        Args:
            max_plots: Maximum number of plots to show. If None, shows all features.
            figsize: Figure size. If None, auto-calculated based on number of features.
        """
        # Determine number of plots
        total_features = len(self.qa.all_cols)
        n_plots = total_features if max_plots is None else min(max_plots, total_features)
        
        # Auto-calculate optimal layout
        if n_plots <= 4:
            n_cols = 2
        elif n_plots <= 9:
            n_cols = 3
        elif n_plots <= 16:
            n_cols = 4
        elif n_plots <= 25:
            n_cols = 5
        else:
            n_cols = 6  # Max 6 columns for readability
        
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            base_width = 4.5  # Base width per subplot
            base_height = 3.5  # Base height per subplot
            figsize = (n_cols * base_width, n_rows * base_height)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        # Define professional color palette
        colors = {
            'original': '#2E86AB',     # Professional blue
            'synthetic': '#F24236',    # Professional red
            'background': '#F8F9FA',   # Light background
            'grid': '#E9ECEF'          # Light grid
        }
        
        # Plot numerical features
        for i, col_name in enumerate(self.qa.num_cols):
            if plot_idx >= n_plots:
                break
                
            ax = axes[plot_idx]
            
            orig_vals = self.qa.original_num[:, i]
            synth_vals = self.qa.synthetic_num[:, i]
            
            # Plot histograms with professional styling
            ax.hist(orig_vals, bins=30, alpha=0.7, label='Original', density=True, 
                   color=colors['original'], edgecolor='white', linewidth=0.5)
            ax.hist(synth_vals, bins=30, alpha=0.7, label='Synthetic', density=True, 
                   color=colors['synthetic'], edgecolor='white', linewidth=0.5)
            
            ax.set_title(f'{col_name}\n(Numerical)', fontsize=10, fontweight='bold', pad=15)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, color=colors['grid'])
            ax.set_facecolor(colors['background'])
            
            # Improve axis formatting
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            
            plot_idx += 1
        
        # Plot categorical features
        for i, col_name in enumerate(self.qa.cat_cols):
            if plot_idx >= n_plots:
                break
                
            ax = axes[plot_idx]
            
            orig_vals = self.qa.original_cat[:, i]
            synth_vals = self.qa.synthetic_cat[:, i]
            
            # Get unique values and their counts
            all_unique = np.union1d(np.unique(orig_vals), np.unique(synth_vals))
            
            # Handle features with many categories (limit to top categories)
            max_categories = 15
            if len(all_unique) > max_categories:
                # Get top categories by frequency in original data
                orig_counts_dict = {val: np.sum(orig_vals == val) for val in all_unique}
                top_categories = sorted(orig_counts_dict.items(), key=lambda x: x[1], reverse=True)[:max_categories-1]
                top_vals = [item[0] for item in top_categories]
                
                # Group remaining as "Other"
                other_orig = len(orig_vals) - sum(np.sum(orig_vals == val) for val in top_vals)
                other_synth = len(synth_vals) - sum(np.sum(synth_vals == val) for val in top_vals)
                
                categories = top_vals + ['Other']
                orig_counts = [np.sum(orig_vals == val) for val in top_vals] + [other_orig]
                synth_counts = [np.sum(synth_vals == val) for val in top_vals] + [other_synth]
            else:
                categories = all_unique
                orig_counts = [np.sum(orig_vals == val) for val in categories]
                synth_counts = [np.sum(synth_vals == val) for val in categories]
            
            # Normalize to probabilities
            orig_probs = np.array(orig_counts) / len(orig_vals)
            synth_probs = np.array(synth_counts) / len(synth_vals)
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, orig_probs, width, label='Original', 
                          alpha=0.8, color=colors['original'], edgecolor='white', linewidth=0.5)
            bars2 = ax.bar(x + width/2, synth_probs, width, label='Synthetic', 
                          alpha=0.8, color=colors['synthetic'], edgecolor='white', linewidth=0.5)
            
            ax.set_title(f'{col_name}\n(Categorical)', fontsize=10, fontweight='bold', pad=15)
            ax.set_xlabel('Categories', fontsize=9)
            ax.set_ylabel('Probability', fontsize=9)
            ax.set_xticks(x)
            
            # Smart label formatting
            labels = []
            for cat in categories:
                label_str = str(cat)
                if len(label_str) > 8:
                    label_str = label_str[:8] + '...'
                labels.append(label_str)
            
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, axis='y', color=colors['grid'])
            ax.set_facecolor(colors['background'])
            
            # Improve axis formatting
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            plot_idx += 1
        
        # Hide unused subplots with professional styling
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        # Add overall title and improve spacing
        feature_type_counts = f"({len(self.qa.num_cols)} numerical, {len(self.qa.cat_cols)} categorical)"
        plt.suptitle(f'Univariate Distribution Comparison - {total_features} Features {feature_type_counts}', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Leave space for suptitle
        
        return fig
    
    def save_all_plots(self, output_dir: str = 'qa_plots') -> None:
        """
        Generate and save all QA plots to specified directory.
        
        Args:
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving simplified QA plots to {output_dir}/...")
        
        # Correlation comparison
        try:
            fig = self.plot_correlation_comparison()
            if fig:
                fig.savefig(f'{output_dir}/correlation_comparison.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                print("  ✓ Correlation comparison saved")
        except Exception as e:
            print(f"  ✗ Error saving correlation comparison: {e}")
        
        # Univariate distributions
        try:
            fig = self.plot_univariate_distributions()
            if fig:
                fig.savefig(f'{output_dir}/univariate_distributions.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                total_features = len(self.qa.all_cols)
                print(f"  ✓ Univariate distributions saved ({total_features} features)")
        except Exception as e:
            print(f"  ✗ Error saving univariate distributions: {e}")
        
        print("All simplified plots saved successfully!") 