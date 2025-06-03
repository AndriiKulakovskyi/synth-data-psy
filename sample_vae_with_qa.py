#!/usr/bin/env python3
import os
import argparse
import torch
import pandas as pd
import numpy as np
import json
from src.data.clean import clean
from src.data.wrangle import wrangle
from src.data import split_numerical_categorical, reconstruct_decoded_dataframe
from src.trainer.vae_trainer import VAETrainer
from src.tools import QualityAssessment, QAVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate synthetic data with VAE and run quality assessment')
    parser.add_argument('--data_path', type=str, default='FACE/neuropsy_v0.csv',
                        help='Path to wrangled dataset')
    parser.add_argument('--checkpoint', type=str, default='ckpt/model.pt',
                        help='Path to saved VAE model checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--n_cols', type=int, default=20,
                        help='Number of columns to use from dataset')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for inference (cpu/cuda)')
    parser.add_argument('--output', type=str, default='generated_samples/synthetic_data.csv',
                        help='Path to save generated synthetic data')
    parser.add_argument('--qa_output_dir', type=str, default='generated_samples/qa_results',
                        help='Directory to save QA results and plots')
    parser.add_argument('--generate_plots', action='store_true', default=True,
                        help='Generate visualization plots')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("="*80)
    print("VAE SYNTHETIC DATA GENERATION WITH QUALITY ASSESSMENT")
    print("="*80)

    # ─── load and preprocess dataset identically to training ────────────────
    print("\n1. Loading and preprocessing data...")
    df_raw = pd.read_csv(args.data_path, sep=';', low_memory=False)
    
    # drop duplicated ID/visit columns if present
    cols_to_drop = [
        'usubjid_neuropsychologie',
        'visitnum_neuropsychologie',
        'visit_neuropsychologie',
    ]
    df_raw = df_raw.drop(columns=[c for c in cols_to_drop if c in df_raw.columns], errors='ignore')

    df_clean = clean(df_raw)
    df_wrangled, _params, wrangler = wrangle(
        df_clean,
        conversion_threshold=0.80,
        cardinality_threshold=10,
        scale_numeric=True,
    )

    df_partial = df_wrangled[df_wrangled.columns[: args.n_cols]]
    print(f"   Using {len(df_partial.columns)} columns from wrangled data")
    print(f"   Original data shape: {df_partial.shape}")

    # Split into numerical and categorical matrices for original data
    num_mat, cat_mat, mapping = split_numerical_categorical(df_partial, cardinality_threshold=10)
    num_mat = num_mat.astype(np.float32)
    cat_mat = cat_mat.astype(np.int64)
    
    print(f"   Numerical features: {num_mat.shape[1]}")
    print(f"   Categorical features: {cat_mat.shape[1]}")

    # ─── load model ──────────────────────────────────────────────────────────
    print("\n2. Loading trained VAE model...")
    model, _model_cfg = VAETrainer.load_model(args.checkpoint, device=args.device)
    print(f"   Model loaded on device: {args.device}")

    # ─── generate synthetic samples ─────────────────────────────────────────
    print(f"\n3. Generating {args.num_samples} synthetic samples...")
    with torch.no_grad():
        x_num, x_cat = model.sample(args.num_samples, current_device=args.device)

    # Convert to numpy for QA
    x_num_np = x_num.cpu().numpy() if isinstance(x_num, torch.Tensor) else x_num
    
    # Handle categorical data that might be a list of tensors
    if isinstance(x_cat, list):
        # Convert list of tensors to single numpy array
        x_cat_list = []
        for cat_tensor in x_cat:
            if isinstance(cat_tensor, torch.Tensor):
                # Get the argmax if it's a probability distribution
                if cat_tensor.dim() > 1 and cat_tensor.shape[-1] > 1:
                    cat_data = cat_tensor.argmax(dim=-1).cpu().numpy()
                else:
                    cat_data = cat_tensor.cpu().numpy()
            else:
                cat_data = np.array(cat_tensor)
            x_cat_list.append(cat_data)
        
        # Stack into single array
        x_cat_np = np.column_stack(x_cat_list) if x_cat_list else np.empty((args.num_samples, 0))
        print(f"   Generated categorical data: {len(x_cat)} variables -> {x_cat_np.shape}")
    else:
        # Handle single tensor/array case
        if isinstance(x_cat, torch.Tensor):
            x_cat_np = x_cat.cpu().numpy()
            if x_cat.dim() > 1 and x_cat.shape[-1] > 1:
                x_cat_np = x_cat_np.argmax(axis=-1)
        else:
            x_cat_np = np.array(x_cat)
            if x_cat_np.ndim > 1 and x_cat_np.shape[-1] > 1:
                x_cat_np = x_cat_np.argmax(axis=-1)
        print(f"   Generated categorical data shape: {x_cat_np.shape}")
    
    print(f"   Generated numerical data shape: {x_num_np.shape}")

    # ─── quality assessment ─────────────────────────────────────────────────
    print("\n4. Running Quality Assessment...")
    
    # Initialize QA pipeline
    qa = QualityAssessment(
        original_num=num_mat,
        original_cat=cat_mat,
        synthetic_num=x_num_np,
        synthetic_cat=x_cat_np,
        mapping=mapping
    )
    
    # Run full assessment
    qa_results = qa.run_full_assessment()
    
    # Print summary
    qa.print_summary()

    # ─── save results ───────────────────────────────────────────────────────
    print(f"\n5. Saving results...")
    
    # Create output directories
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.qa_output_dir, exist_ok=True)
    
    # Reconstruct and save synthetic dataframe
    df_generated = reconstruct_decoded_dataframe(
        numerical_matrix=x_num,
        categorical_matrix=x_cat,
        mapping=mapping,
        wrangler=wrangler,
        drop_scaled=True,
    )
    
    df_generated.to_csv(args.output, index=False)
    print(f"   ✓ Saved {len(df_generated)} synthetic rows to {args.output}")
    
    # Save QA results as JSON
    qa_results_file = os.path.join(args.qa_output_dir, 'qa_results.json')
    
    # Convert numpy types to regular Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()  # Convert DataFrame to nested dict
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif pd.isna(obj):  # Handle NaN values
            return None
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    qa_results_json = convert_for_json(qa_results)
    
    with open(qa_results_file, 'w') as f:
        json.dump(qa_results_json, f, indent=2)
    print(f"   ✓ Saved QA results to {qa_results_file}")
    
    # Generate and save plots
    if args.generate_plots:
        print("\n6. Generating QA visualizations...")
        try:
            visualizer = QAVisualizer(qa)
            plot_output_dir = os.path.join(args.qa_output_dir, 'plots')
            visualizer.save_all_plots(output_dir=plot_output_dir)
        except Exception as e:
            print(f"   ⚠ Error generating plots: {e}")
            print("   You may need to install additional visualization dependencies (matplotlib, seaborn)")
    
    # ─── summary report ─────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    # Correlation analysis summary
    if 'global_correlations' in qa_results and qa_results['global_correlations']:
        corr_data = qa_results['global_correlations']
        print(f"Correlation Analysis:")
        print(f"  Mean absolute correlation difference: {corr_data.get('mean_abs_corr_diff_offdiag', 'N/A'):.4f}")
        print(f"  Maximum absolute correlation difference: {corr_data.get('max_abs_corr_diff', 'N/A'):.4f}")
        print(f"  Correlation RMSE: {corr_data.get('correlation_rmse', 'N/A'):.4f}")
        print(f"  Correlation of correlations: {corr_data.get('correlation_of_correlations', 'N/A'):.4f}")
        print(f"  Fraction with small differences (<0.1): {corr_data.get('frac_small_diff', 'N/A'):.4f}")
    
    # Univariate analysis summary
    if 'univariate_metrics' in qa_results:
        print(f"\nUnivariate Analysis:")
        
        # Count features and compute averages
        num_features = sum(1 for k in qa_results['univariate_metrics'].keys() if k.startswith('num_'))
        cat_features = sum(1 for k in qa_results['univariate_metrics'].keys() if k.startswith('cat_'))
        print(f"  Analyzed features: {num_features} numerical, {cat_features} categorical")
        
        # Average metrics
        ks_stats = []
        tvds = []
        for feature, metrics in qa_results['univariate_metrics'].items():
            if 'ks_statistic' in metrics:
                ks_stats.append(metrics['ks_statistic'])
            if 'tvd' in metrics:
                tvds.append(metrics['tvd'])
        
        if ks_stats:
            print(f"  Average KS statistic (numerical): {np.mean(ks_stats):.4f}")
        if tvds:
            print(f"  Average TVD (categorical): {np.mean(tvds):.4f}")
    
    print(f"\nFiles Generated:")
    print(f"  • Synthetic data: {args.output}")
    print(f"  • QA results: {qa_results_file}")
    if args.generate_plots:
        print(f"  • QA plots: {os.path.join(args.qa_output_dir, 'plots')}/")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main() 