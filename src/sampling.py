import torch
from constants import EPS
from helpers import generate_draft_sequence, sample_from_logits, get_distribution, sample_from_residual_distribution

@torch.no_grad()
def autoregressive_sampling(model, initial_prompt_seq, target_len, temperature):
    n = initial_prompt_seq.shape[1]
    result_seq = initial_prompt_seq.detach().clone()
    
    while n < target_len:
        sample_token_logits = model(result_seq).logits[:, -1, :]
        sample_token = sample_from_logits(sample_token_logits, temperature).unsqueeze(-1)
        result_seq = torch.cat([result_seq, sample_token], dim=1)
        n += 1
        
    return result_seq


@torch.no_grad()
def speculative_sampling(target_model, draft_model, initial_prompt_seq, target_len, temperature, lookahead_k=4):
    n = initial_prompt_seq.shape[1]
    result_seq = initial_prompt_seq.detach().clone()
    device = initial_prompt_seq.device
    
    while n < target_len:
        draft_seq, draft_tokens, draft_distributions = generate_draft_sequence(
            draft_model, result_seq, lookahead_k, temperature
        )
        
        target_outputs = target_model(draft_seq)
        target_logits = target_outputs.logits
        
        # Starting position for target logits evaluation (the position before the first draft token)
        start_pos = result_seq.shape[1] - 1
        
        all_accepted = True
        for t in range(lookahead_k):
            # Looking at the current position
            q, p = get_distribution(target_logits[:, start_pos + t, :], temperature), draft_distributions[t] 
            current_token = draft_tokens[t]
            
            if temperature <= EPS:
                # In deterministic mode accept only if target and draft agree on the token
                target_token = torch.argmax(target_logits[:, start_pos + t, :], dim=-1).item()
                if current_token == target_token:
                    result_seq = torch.cat([result_seq, torch.tensor(current_token, device=device).view(1, 1)], dim=1)
                    n += 1
                    
                    if n >= target_len:
                        break
                else:
                    # Reject and use target's choice
                    result_seq = torch.cat([result_seq, torch.tensor(target_token, device=device).view(1, 1)], dim=1)
                    n += 1
                    all_accepted = False
                    break
            else:
                # Standard rejection sampling for temperature > 0
                r = torch.rand(1).item()
                if r < min(1.0, (q[0, current_token] / (p[0, current_token] + EPS)).item()):
                    # Accept the draft token
                    result_seq = torch.cat([result_seq, torch.tensor(current_token, device=device).view(1, 1)], dim=1)
                    n += 1
                    
                    if n >= target_len:
                        break
                else:
                    # Sample from (q - p)+ and break
                    resampled_token = sample_from_residual_distribution(q, p)
                    result_seq = torch.cat([result_seq, resampled_token], dim=1)
                    n += 1
                    
                    all_accepted = False
                    break
        
        # Sample extra token if all draft tokens are accepted
        if all_accepted and n < target_len:
            next_logits = target_logits[:, start_pos + lookahead_k, :]
            next_token = sample_from_logits(next_logits, temperature)
            result_seq = torch.cat([result_seq, next_token.unsqueeze(0)], dim=1)
            n += 1
    
    return result_seq
