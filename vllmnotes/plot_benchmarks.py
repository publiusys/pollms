#!/usr/bin/env python3
"""
Complete plotting script for vLLM benchmark results.
Processes JSON outputs and creates publication-quality comparison plots.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configuration
RESULTS_DIR = Path("../results")
OUTPUT_DIR = Path("./plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10


def parse_results():
    """Parse all benchmark JSON files and create a summary dataframe."""
    results = []
    
    for json_file in RESULTS_DIR.glob("results_*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Parse filename: results_Qwen_Qwen2-7B_tp2_io512_bs16.json
            filename = json_file.stem.replace("results_", "")
            parts = filename.split("_")
            
            # Extract model name (everything before _tp)
            tp_idx = next(i for i, p in enumerate(parts) if p.startswith('tp'))
            model_name = "_".join(parts[:tp_idx])
            
            # Extract parameters
            tensor_parallel = int(parts[tp_idx].replace('tp', ''))
            input_output_length = int(parts[tp_idx + 1].replace('io', ''))
            batch_size = int(parts[tp_idx + 2].replace('bs', ''))
            
            # Extract metrics from vLLM output
            # Handle both throughput and latency benchmark formats
            
            # Calculate throughput from latency if not directly provided
            avg_latency = data.get('avg_latency', 0)
            if avg_latency > 0:
                # Throughput = batch_size / latency (in seconds)
                # Assuming input_output_length tokens per request
                throughput = (batch_size * input_output_length * 2) / avg_latency  # *2 for input+output
            else:
                throughput = data.get('throughput', data.get('tokens_per_second', 0))
            
            # Get percentiles if available
            percentiles = data.get('percentiles', {})
            
            result = {
                'model': model_name,
                'tensor_parallel': tensor_parallel,
                'input_output_length': input_output_length,
                'batch_size': batch_size,
                'throughput': throughput,
                'mean_latency': avg_latency,
                'p50_latency': percentiles.get('50', data.get('p50_latency', 0)),
                'p99_latency': percentiles.get('99', data.get('p99_latency', 0)),
                'p10_latency': percentiles.get('10', 0),
                'p90_latency': percentiles.get('90', 0),
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    if not results:
        print("No results found! Check your RESULTS_DIR path.")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    print(f"\nLoaded {len(df)} benchmark results")
    print(f"Models: {df['model'].unique()}")
    print(f"Tensor parallel sizes: {sorted(df['tensor_parallel'].unique())}")
    print(f"Batch sizes: {sorted(df['batch_size'].unique())}")
    print(f"I/O lengths: {sorted(df['input_output_length'].unique())}")
    
    return df


def plot_io_length_comparison(df, model_name, batch_size=1):
    """
    Plot throughput vs input/output length for different tensor parallel sizes.
    Similar to Figure 2(a).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    subset = df[(df['model'] == model_name) & (df['batch_size'] == batch_size)]
    
    if subset.empty:
        print(f"No data for model={model_name}, batch_size={batch_size}")
        return
    
    colors = sns.color_palette("husl", n_colors=len(subset['tensor_parallel'].unique()))
    
    for idx, tp_size in enumerate(sorted(subset['tensor_parallel'].unique())):
        data = subset[subset['tensor_parallel'] == tp_size].sort_values('input_output_length')
        ax.plot(data['input_output_length'], data['throughput'],
                marker='o', linewidth=2.5, markersize=10,
                color=colors[idx], label=f'TP={tp_size}')
    
    ax.set_xlabel('Input/Output Length', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=13, fontweight='bold')
    ax.set_title(f'Effect of Tensor Parallelism on Throughput\n{model_name} with Batch Size {batch_size}',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(title='Tensor Parallel', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    filename = f"throughput_vs_io_{model_name.replace('/', '_')}_bs{batch_size}.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_batch_size_scaling(df, model_name, io_length=1024):
    """
    Plot throughput vs batch size for different tensor parallel sizes.
    Similar to Figure 2(b).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    subset = df[(df['model'] == model_name) & (df['input_output_length'] == io_length)]
    
    if subset.empty:
        print(f"No data for model={model_name}, io_length={io_length}")
        return
    
    colors = sns.color_palette("husl", n_colors=len(subset['tensor_parallel'].unique()))
    
    for idx, tp_size in enumerate(sorted(subset['tensor_parallel'].unique())):
        data = subset[subset['tensor_parallel'] == tp_size].sort_values('batch_size')
        ax.plot(data['batch_size'], data['throughput'],
                marker='o', linewidth=2.5, markersize=10,
                color=colors[idx], label=f'TP={tp_size}')
    
    ax.set_xlabel('Batch Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=13, fontweight='bold')
    ax.set_title(f'Batch Size Scaling at I/O Length={io_length}\n{model_name}',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(title='Tensor Parallel', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    filename = f"throughput_vs_batch_{model_name.replace('/', '_')}_io{io_length}.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_heatmap(df, model_name, tp_size=1):
    """
    Create a heatmap showing throughput across batch size and I/O length.
    """
    subset = df[(df['model'] == model_name) & (df['tensor_parallel'] == tp_size)]
    
    if subset.empty:
        print(f"No data for model={model_name}, tp_size={tp_size}")
        return
    
    # Pivot to create matrix
    pivot = subset.pivot(index='input_output_length', 
                         columns='batch_size', 
                         values='throughput')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                cbar_kws={'label': 'Throughput (tokens/sec)'}, ax=ax)
    
    ax.set_title(f'Throughput Heatmap: {model_name} (TP={tp_size})',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Input/Output Length', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    filename = f"heatmap_{model_name.replace('/', '_')}_tp{tp_size}.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_latency_comparison(df, model_name, batch_size=1):
    """
    Plot latency vs input/output length for different tensor parallel sizes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    subset = df[(df['model'] == model_name) & (df['batch_size'] == batch_size)]
    
    if subset.empty:
        print(f"No data for model={model_name}, batch_size={batch_size}")
        return
    
    colors = sns.color_palette("husl", n_colors=len(subset['tensor_parallel'].unique()))
    
    for idx, tp_size in enumerate(sorted(subset['tensor_parallel'].unique())):
        data = subset[subset['tensor_parallel'] == tp_size].sort_values('input_output_length')
        ax.plot(data['input_output_length'], data['mean_latency'],
                marker='s', linewidth=2.5, markersize=10,
                color=colors[idx], label=f'TP={tp_size}')
    
    ax.set_xlabel('Input/Output Length', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title(f'Latency vs Input/Output Length\n{model_name} with Batch Size {batch_size}',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(title='Tensor Parallel', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    filename = f"latency_vs_io_{model_name.replace('/', '_')}_bs{batch_size}.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_model_comparison(df, io_length=1024, batch_size=16, tp_size=1):
    """
    Compare throughput across different models.
    """
    subset = df[(df['input_output_length'] == io_length) & 
                (df['batch_size'] == batch_size) &
                (df['tensor_parallel'] == tp_size)]
    
    if subset.empty:
        print(f"No data for comparison with io={io_length}, bs={batch_size}, tp={tp_size}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(subset))
    bars = ax.bar(x, subset['throughput'], color=sns.color_palette("Set2"), 
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=13, fontweight='bold')
    ax.set_title(f'Model Comparison\nI/O Length={io_length}, Batch Size={batch_size}, TP={tp_size}',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(subset['model'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filename = f"model_comparison_io{io_length}_bs{batch_size}_tp{tp_size}.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def create_summary_table(df):
    """Create a summary table with key statistics."""
    summary = df.groupby(['model', 'tensor_parallel', 'batch_size', 'input_output_length']).agg({
        'throughput': ['mean', 'std', 'min', 'max'],
        'mean_latency': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    summary_file = OUTPUT_DIR / "summary_statistics.csv"
    summary.to_csv(summary_file)
    print(f"\nSaved summary statistics to: {summary_file}")
    
    return summary


def main():
    """Main execution function."""
    print("="*60)
    print("vLLM Benchmark Plotting Script")
    print("="*60)
    
    # Parse all results
    df = parse_results()
    
    if df.empty:
        print("\nNo data to plot. Exiting.")
        return
    
    # Save raw aggregated data
    df.to_csv(OUTPUT_DIR / "aggregated_results.csv", index=False)
    print(f"\nSaved aggregated results to: {OUTPUT_DIR / 'aggregated_results.csv'}")
    
    print("\n" + "="*60)
    print("Generating Plots...")
    print("="*60 + "\n")
    
    # Get unique models
    models = df['model'].unique()
    
    # Generate plots for each model
    for model in models:
        print(f"\n--- Plotting for {model} ---")
        
        # 1. Throughput vs I/O length (different batch sizes)
        for bs in [1, 16, 32]:
            if bs in df[df['model'] == model]['batch_size'].values:
                plot_io_length_comparison(df, model, batch_size=bs)
        
        # 2. Throughput vs batch size (different I/O lengths)
        for io_len in [512, 1024, 2048]:
            if io_len in df[df['model'] == model]['input_output_length'].values:
                plot_batch_size_scaling(df, model, io_length=io_len)
        
        # 3. Heatmaps for each TP size
        for tp in df[df['model'] == model]['tensor_parallel'].unique():
            plot_heatmap(df, model, tp_size=tp)
        
        # 4. Latency comparisons
        for bs in [1, 16]:
            if bs in df[df['model'] == model]['batch_size'].values:
                plot_latency_comparison(df, model, batch_size=bs)
    
    # 5. Cross-model comparison
    if len(models) > 1:
        print("\n--- Cross-Model Comparison ---")
        plot_model_comparison(df, io_length=1024, batch_size=16, tp_size=1)
    
    # 6. Create summary table
    print("\n--- Generating Summary Statistics ---")
    create_summary_table(df)
    
    print("\n" + "="*60)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()