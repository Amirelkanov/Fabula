import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def generate_results_plots_for_benchmark(results, temperature):
    plots_dir = "benchmark_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {
        'auto': '#3498db',    
        'spec': '#2ecc71',    
        'highlight': '#e74c3c'
    }
    
    speedups = results['speedups']
    auto_latencies, auto_throughputs = results['auto_latencies'], results['auto_throughputs']
    spec_latencies, spec_throughputs = results['spec_latencies'], results['spec_throughputs']
    
    create_speedup_distribution_plot(speedups, results['overall_speedup'], colors, plots_dir, temperature)
    create_latency_distribution_plot(auto_latencies, spec_latencies, colors, plots_dir, temperature)
    create_throughput_distribution_plot(auto_throughputs, spec_throughputs, colors, plots_dir, temperature)
    
    print(f"Enhanced plots saved to {plots_dir}/")


def create_speedup_distribution_plot(speedups, overall_speedup, colors, plots_dir, temperature):
    plt.figure(figsize=(10, 6))
    
    sns.histplot(
        speedups, 
        bins=15, 
        kde=True, 
        alpha=0.7, 
        color=colors['highlight']
    )
    
    plt.axvline(
        overall_speedup, 
        color='black', 
        linestyle='dashed', 
        linewidth=2, 
        label=f'Overall: {overall_speedup:.2f}x'
    )
    
    stats_text = (
        f"Mean: {np.mean(speedups):.2f}x\n"
        f"Median: {np.median(speedups):.2f}x\n"
        f"Min: {np.min(speedups):.2f}x\n"
        f"Max: {np.max(speedups):.2f}x\n"
        f"Std Dev: {np.std(speedups):.2f}"
    )
    plt.text(
        0.05, 0.95, stats_text, 
        transform=plt.gca().transAxes,
        verticalalignment='top', 
        bbox=dict(boxstyle='round', alpha=0.1)
    )
    
    plt.title(f"Speculative Sampling Speedup Distribution (temperature: {temperature}, seed: {torch.random.initial_seed()})", fontsize=14, fontweight='bold')
    plt.xlabel('Speedup Factor (x times)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{plots_dir}/speedup_distribution.png")
    plt.close()


def create_latency_distribution_plot(auto_latencies, spec_latencies, colors, plots_dir, temperature):
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(auto_latencies, color=colors['auto'], fill=True, alpha=0.5, label='Autoregressive')
    sns.kdeplot(spec_latencies, color=colors['spec'], fill=True, alpha=0.5, label='Speculative')
    
    auto_mean, spec_mean = np.mean(auto_latencies), np.mean(spec_latencies)
    
    plt.axvline(auto_mean, color=colors['auto'], linestyle='--', 
                label=f'Auto mean: {auto_mean:.4f}s')
    plt.axvline(spec_mean, color=colors['spec'], linestyle='--',
                label=f'Spec mean: {spec_mean:.4f}s')
    
    reduction_pct = (1 - spec_mean/auto_mean) * 100
    plt.text(
        0.5, 0.95, 
        f"Latency reduction: {reduction_pct:.1f}%", 
        transform=plt.gca().transAxes, 
        ha='center',
        bbox=dict(boxstyle='round', alpha=0.1)
    )
    
    plt.title(f"Latency Distribution Comparison (temperature: {temperature}, seed: {torch.random.initial_seed()})", fontsize=14, fontweight='bold')
    plt.xlabel('Latency (seconds)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{plots_dir}/latency_distributions.png")
    plt.close()


def create_throughput_distribution_plot(auto_throughputs, spec_throughputs, colors, plots_dir, temperature):
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(auto_throughputs, color=colors['auto'], fill=True, alpha=0.5, label='Autoregressive')
    sns.kdeplot(spec_throughputs, color=colors['spec'], fill=True, alpha=0.5, label='Speculative')
    
    auto_mean, spec_mean = np.mean(auto_throughputs), np.mean(spec_throughputs)
    
    plt.axvline(auto_mean, color=colors['auto'], linestyle='--', 
                label=f'Auto mean: {auto_mean:.2f} tokens/s')
    plt.axvline(spec_mean, color=colors['spec'], linestyle='--',
                label=f'Spec mean: {spec_mean:.2f} tokens/s')
    
    improvement_pct = ((spec_mean/auto_mean) - 1) * 100
    plt.text(
        0.5, 0.95, 
        f"Throughput improvement: {improvement_pct:.1f}%", 
        transform=plt.gca().transAxes, 
        ha='center',
        bbox=dict(boxstyle='round', alpha=0.1)
    )
    
    plt.title(f"Throughput Distribution Comparison (temperature: {temperature}, seed: {torch.random.initial_seed()})", fontsize=14, fontweight='bold')
    plt.xlabel('Throughput (tokens/second)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{plots_dir}/throughput_distributions.png")
    plt.close()
