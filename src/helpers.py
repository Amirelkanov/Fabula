import torch
from constants import EPS
import torch.nn.functional as F
import numpy as np
import random
import numpy as np

def get_distribution(logits, temperature):
    if temperature <= EPS:
        # In deterministic mode create a one-hot distribution for the max logit
        max_indices = torch.argmax(logits, dim=-1, keepdim=True)
        distribution = torch.zeros_like(logits).scatter_(-1, max_indices, 1.0)
        return distribution
    else: # Regular softmax with temperature
        return F.softmax(logits / temperature, dim=-1)

def sample_from_logits(logits, temperature):
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
