import torch
from constants import EPS
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def get_distribution(logits, temperature=1.0):
    if temperature <= EPS:
        # In deterministic mode create a one-hot distribution for the max logit
        max_indices = torch.argmax(logits, dim=-1, keepdim=True)
        distribution = torch.zeros_like(logits).scatter_(-1, max_indices, 1.0)
        return distribution
    else: # Regular softmax with temperature
        return F.softmax(logits / temperature, dim=-1)

def sample_from_logits(logits, temperature=1.0):
    # In deterministic mode just return the argmax directly
    if temperature <= EPS:
        return torch.argmax(logits, dim=-1)
    
    # Otherwise sample from the distribution
    probs = get_distribution(logits, temperature)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def generate_draft_sequence(draft_model, prompt, lookahead_k, temperature):
    draft_seq = prompt.clone()
    draft_tokens = []
    draft_distributions = []
    
    for _ in range(lookahead_k):
        outputs = draft_model(draft_seq)
        logits = outputs.logits[:, -1, :]
        distribution = get_distribution(logits, temperature)
        
        token = torch.argmax(logits, dim=-1) if temperature <= EPS else torch.multinomial(distribution, num_samples=1).squeeze(-1)
        
        draft_tokens.append(token.item())
        draft_distributions.append(distribution)
        draft_seq = torch.cat([draft_seq, token.unsqueeze(1)], dim=1)
    
    return draft_seq, draft_tokens, draft_distributions

def sample_from_residual_distribution(target_dist, draft_dist): 
    # Residual distribution means (q - p)+.
    diff = torch.clamp(target_dist - draft_dist, min=0.0)
    if diff.sum() > EPS:
        diff = diff / diff.sum()
        return torch.multinomial(diff, num_samples=1)
    else:
        # Fallback if numerical issues occur
        return torch.multinomial(target_dist, num_samples=1)

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def show_results_and_update_metrics(
    input_ids, output, sampling_time, metrics, tokenizer
):
    tokens_generated = output.shape[1] - input_ids.shape[1]
    throughput = tokens_generated / sampling_time

    metrics.update(sampling_time, tokens_generated)
    metrics_data = metrics.compute()

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nOutput:")
    print(text)
    print(f"\nLatency: {sampling_time:.4f} seconds (avg: {metrics_data['avg_latency']:.4f})")
    print(f"Throughput: {throughput:.2f} tokens/second (avg: {metrics_data['throughput']:.2f})")

def generate_results_plots_for_benchmark(results):
    plots_dir = "benchmark_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(results['speedups'], bins=15, alpha=0.7, color='blue')
    plt.axvline(results['overall_speedup'], color='red', linestyle='dashed', linewidth=2, label=f'Overall: {results["overall_speedup"]:.2f}x')
    plt.title('Speculative Sampling Speedup Distribution')
    plt.xlabel('Speedup (x times)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{plots_dir}/speedup_distribution.png")
    
    plt.figure(figsize=(10, 6))
    methods = ['Autoregressive', 'Speculative']
    latencies = [
        results['auto_results']['avg_latency'].item(), 
        results['spec_results']['avg_latency'].item()
    ]
    plt.bar(methods, latencies, color=['blue', 'green'])
    plt.title('Average Latency Comparison')
    plt.ylabel('Latency (seconds)')
    plt.grid(axis='y', alpha=0.3)
    for i, v in enumerate(latencies):
        plt.text(i, v + 0.01, f"{v:.4f}s", ha='center', fontweight='bold')
    plt.savefig(f"{plots_dir}/latency_comparison.png")
    
    plt.figure(figsize=(10, 6))
    throughputs = [
        results['auto_results']['throughput'].item(), 
        results['spec_results']['throughput'].item()
    ]
    plt.bar(methods, throughputs, color=['blue', 'green'])
    plt.title('Throughput Comparison')
    plt.ylabel('Throughput (tokens/second)')
    plt.grid(axis='y', alpha=0.3)
    for i, v in enumerate(throughputs):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center', fontweight='bold')
    plt.savefig(f"{plots_dir}/throughput_comparison.png")
    
    print(f"Plots saved to {plots_dir}/")