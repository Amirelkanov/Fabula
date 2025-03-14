import argparse
import torch
import os
import lightning as L
from transformers import AutoTokenizer, AutoModelForCausalLM
from constants import CUDA_DEVICE, SMALL_DRAFT_MODEL, SMALL_TARGET_MODEL, TARGET_MODEL, DRAFT_MODEL
from mode import interactive_mode, benchmark_mode
from plots import generate_results_plots_for_benchmark
from finetune_draft_model import Lit, create_peft_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Speculative Sampling Demo')
    parser.add_argument('--temperature', type=float, default=0.0, 
                        help='Temperature for sampling (default: 0.0)')
    parser.add_argument('--max_tokens', type=int, default=50, 
                        help='Maximum number of tokens to generate (default: 50)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed (default: 42)')
    parser.add_argument('--benchmark', action='store_true', 
                        help='Run in benchmark mode')
    parser.add_argument('--device', type=str, default=CUDA_DEVICE if torch.cuda.is_available() else 'cpu', 
                        help=f"Device to run on (default: {CUDA_DEVICE} or cpu if CUDA not available)")
    parser.add_argument('--lookahead_k', type=int, default=4, 
                        help='Number of tokens to lookahead in speculative sampling (default: 4)')
    
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size for benchmark mode (default: 8)')
    parser.add_argument('--min_prompt_len', type=int, default=5, 
                        help='Batch size for benchmark mode (default: 5)')
    parser.add_argument('--max_prompt_len', type=int, default=50, 
                        help='Batch size for benchmark mode (default: 50)')
    parser.add_argument('--num_batches', type=int, default=10, 
                        help='Number of batches to use for benchmark mode (default: 10)')
    parser.add_argument('--plot_results', action='store_true',
                        help='Generate plots of benchmark results')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    L.seed_everything(args.seed)
    
    print(f"Running with arguments: {args}")
    print("Loading models...")
    
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL).to(args.device)
    draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL).to(args.device)
    # If you want to use fine-tuned model from checkpoint as draft model, uncomment four rows below
    #draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL)
    #draft_model = create_peft_config(draft_model)
    #draft_model.load_state_dict(torch.load("fine_tuned_weights.pth"), strict=False)
    #draft_model = draft_model.to(args.device)
    
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    print("Models loaded successfully.")
    
    if args.benchmark:
        results = benchmark_mode(target_model, draft_model, tokenizer, args)
        
        if args.plot_results:
            generate_results_plots_for_benchmark(results, args.temperature)   
    else:
        interactive_mode(target_model, draft_model, tokenizer, args)