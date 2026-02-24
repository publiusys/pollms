#!/usr/bin/env python3
"""
Power Analysis Script for vLLM Benchmarks
Analyzes power consumption CSV files and correlates with benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configuration
RESULTS_DIR = Path("../results")
OUTPUT_DIR = Path("./plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


def parse_power_csv(power_file):
    """Parse a power CSV file and return summary statistics."""
    try:
        df = pd.read_csv(power_file, skipinitialspace=True)
        
        # Clean column names (remove units)
        df.columns = df.columns.str.strip()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract numeric power values (handle "power.draw [W]" or "power.draw")
        power_col = [col for col in df.columns if 'power' in col.lower()][0]
        df['power_w'] = pd.to_numeric(df[power_col], errors='coerce')
        
        # Calculate statistics (exclude idle/warmup - use data where utilization > 50%)
        if 'utilization.gpu [%]' in df.columns or 'utilization.gpu' in df.columns:
            util_col = [col for col in df.columns if 'utilization.gpu' in col][0]
            df['gpu_util'] = pd.to_numeric(df[util_col], errors='coerce')
            active_df = df[df['gpu_util'] > 50]
        else:
            active_df = df
        
        if len(active_df) == 0:
            active_df = df  # Fallback if no high utilization periods
        
        stats = {
            'avg_power': active_df['power_w'].mean(),
            'max_power': active_df['power_w'].max(),
            'min_power': active_df['power_w'].min(),
            'std_power': active_df['power_w'].std(),
            'median_power': active_df['power_w'].median(),
            'duration_sec': (df['timestamp'].max() - df['timestamp'].min()).total_seconds(),
            'energy_wh': active_df['power_w'].mean() * len(active_df) / 3600,  # Watt-hours
        }
        
        return stats, df
    except Exception as e:
        print(f"Error parsing {power_file}: {e}")
        return None, None


def parse_benchmark_json(json_file):
    """Parse a benchmark JSON file."""
    try:
        with open(json_file) as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error parsing {json_file}: {e}")
        return None


def match_power_to_benchmarks():
    """Match power CSV files to benchmark JSON files and create combined analysis."""
    results = []
    
    for json_file in RESULTS_DIR.glob("results_*.json"):
        # Find corresponding power file
        power_file = json_file.parent / json_file.name.replace("results_", "power_").replace(".json", ".csv")
        
        if not power_file.exists():
            print(f"Warning: No power file found for {json_file.name}")
            continue
        
        # Parse both files
        bench_data = parse_benchmark_json(json_file)
        power_stats, power_df = parse_power_csv(power_file)
        
        if bench_data is None or power_stats is None:
            continue
        
        # Extract parameters from filename
        filename = json_file.stem.replace("results_", "")
        parts = filename.split("_")
        
        tp_idx = next(i for i, p in enumerate(parts) if p.startswith('tp'))
        model_name = "_".join(parts[:tp_idx])
        tensor_parallel = int(parts[tp_idx].replace('tp', ''))
        input_output_length = int(parts[tp_idx + 1].replace('io', ''))
        batch_size = int(parts[tp_idx + 2].replace('bs', ''))
        
        # Calculate throughput from latency
        avg_latency = bench_data.get('avg_latency', 0)
        if avg_latency > 0:
            throughput = (batch_size * input_output_length * 2) / avg_latency
        else:
            throughput = bench_data.get('throughput', 0)
        
        # Combine results
        results.append({
            'model': model_name,
            'tensor_parallel': tensor_parallel,
            'input_output_length': input_output_length,
            'batch_size': batch_size,
            'throughput_tokens_per_sec': throughput,
            'avg_latency_sec': avg_latency,
            'avg_power_w': power_stats['avg_power'],
            'max_power_w': power_stats['max_power'],
            'energy_wh': power_stats['energy_wh'],
            'efficiency_tokens_per_watt': throughput / power_stats['avg_power'] if power_stats['avg_power'] > 0 else 0,
            'duration_sec': power_stats['duration_sec'],
        })
    
    if not results:
        print("No matching power/benchmark pairs found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "power_efficiency_analysis.csv", index=False)
    print(f"\nCreated combined analysis: {OUTPUT_DIR / 'power_efficiency_analysis.csv'}")
    print(f"Total combinations analyzed: {len(df)}")
    
    return df


def plot_power_efficiency(df, model_name):
    """Plot throughput vs power consumption."""
    subset = df[df['model'] == model_name]
    
    if subset.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Throughput vs Power by batch size
    for bs in sorted(subset['batch_size'].unique()):
        data = subset[subset['batch_size'] == bs].sort_values('input_output_length')
        ax1.scatter(data['avg_power_w'], data['throughput_tokens_per_sec'], 
                   s=100, label=f'BS={bs}', alpha=0.7)
    
    ax1.set_xlabel('Average Power (W)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Power vs Throughput\n{model_name}', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency (tokens per watt)
    for tp in sorted(subset['tensor_parallel'].unique()):
        data = subset[subset['tensor_parallel'] == tp].sort_values('batch_size')
        ax2.plot(data['batch_size'], data['efficiency_tokens_per_watt'],
                marker='o', linewidth=2, markersize=8, label=f'TP={tp}')
    
    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Efficiency (tokens/watt)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Energy Efficiency\n{model_name}', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"power_efficiency_{model_name.replace('/', '_')}.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_power_over_time(power_csv_file, benchmark_json_file):
    """Plot power consumption over time for a specific benchmark run."""
    power_stats, power_df = parse_power_csv(power_csv_file)
    bench_data = parse_benchmark_json(benchmark_json_file)
    
    if power_df is None or bench_data is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot power over time
    time_offset = (power_df['timestamp'] - power_df['timestamp'].min()).dt.total_seconds()
    ax1.plot(time_offset, power_df['power_w'], linewidth=1.5, color='red')
    ax1.axhline(power_stats['avg_power'], color='blue', linestyle='--', 
                label=f"Avg: {power_stats['avg_power']:.1f}W")
    ax1.set_ylabel('Power (W)', fontsize=12, fontweight='bold')
    ax1.set_title('GPU Power Consumption Over Time', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot GPU utilization if available
    if 'gpu_util' in power_df.columns:
        ax2.plot(time_offset, power_df['gpu_util'], linewidth=1.5, color='green')
        ax2.set_ylabel('GPU Util (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('GPU Utilization Over Time', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    filename = power_csv_file.stem + "_timeline.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def create_power_summary_table(df):
    """Create a summary table of power efficiency metrics."""
    summary = df.groupby(['model', 'tensor_parallel', 'batch_size', 'input_output_length']).agg({
        'throughput_tokens_per_sec': 'mean',
        'avg_power_w': 'mean',
        'energy_wh': 'sum',
        'efficiency_tokens_per_watt': 'mean',
    }).round(2)
    
    summary_file = OUTPUT_DIR / "power_efficiency_summary.csv"
    summary.to_csv(summary_file)
    print(f"Saved power summary: {summary_file}")
    
    return summary


def main():
    print("="*60)
    print("Power Analysis for vLLM Benchmarks")
    print("="*60)
    
    # Match and analyze power data
    df = match_power_to_benchmarks()
    
    if df.empty:
        print("\nNo data to analyze. Check that you have matching:")
        print("  - results_*.json files")
        print("  - power_*.csv files")
        print("in the same directory.")
        return
    
    print("\n" + "="*60)
    print("Generating Power Analysis Plots...")
    print("="*60 + "\n")
    
    # Generate plots for each model
    for model in df['model'].unique():
        print(f"\n--- Analyzing {model} ---")
        plot_power_efficiency(df, model)
    
    # Generate timeline plots for a few example runs
    print("\n--- Generating Timeline Plots ---")
    power_files = list(RESULTS_DIR.glob("power_*.csv"))[:3]  # First 3 runs
    for power_file in power_files:
        json_file = power_file.parent / power_file.name.replace("power_", "results_").replace(".csv", ".json")
        if json_file.exists():
            plot_power_over_time(power_file, json_file)
    
    # Create summary table
    print("\n--- Creating Summary Tables ---")
    create_power_summary_table(df)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()