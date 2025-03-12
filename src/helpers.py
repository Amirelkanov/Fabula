import torch
from constants import EPS
import torch.nn.functional as F

def get_distribution(logits, temperature=1.0):
    return F.softmax(logits / (temperature + EPS), dim=-1)

def sample_from_logits(logits, temperature=1.0):
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
        token = torch.multinomial(distribution, num_samples=1).squeeze(-1)
        
        draft_tokens.append(token.item())
        draft_distributions.append(distribution)
        draft_seq = torch.cat([draft_seq, token.unsqueeze(1)], dim=1)
    
    return draft_seq, draft_tokens, draft_distributions

def sample_from_residual_distribution(target_dist, draft_dist): 
    # Residual distribution means (q - p)+.
    # I named it residual because we will generate token from this part of distribution only if llm-helper's token was rejected
    diff = torch.clamp(target_dist - draft_dist, min=0.0)
    if diff.sum() > EPS:
        diff = diff / diff.sum()
        return torch.multinomial(diff, num_samples=1)
    else:
        # Fallback if numerical issues occur
        return torch.multinomial(target_dist, num_samples=1)
