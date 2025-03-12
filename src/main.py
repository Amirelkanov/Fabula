import argparse
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from constants import TARGET_MODEL, DRAFT_MODEL
from modes import interactive_mode, benchmark
from dataloader import get_dataloader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Speculative Sampling Demo')
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Temperature for sampling (default: 1.0)')
    parser.add_argument('--max_tokens', type=int, default=200, 
                        help='Maximum number of tokens to generate (default: 200)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed (default: 42)')
    parser.add_argument('--benchmark', action='store_true', 
                        help='Run in benchmark mode')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run on (default: cuda:0 or cpu if CUDA not available)')
    parser.add_argument('--lookahead_k', type=int, default=4, 
                        help='Number of tokens to lookahead in speculative sampling (default: 4)')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for benchmark mode (default: 1)')
    parser.add_argument('--num_samples', type=int, default=10, 
                        help='Number of samples to use for benchmarking (default: 10)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Running with arguments: {args}")
    print("Loading models...")
    
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, 
        torch_dtype=torch.float16
    ).to(args.device)
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL, 
        torch_dtype=torch.float16
    ).to(args.device)
    
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    print("Models loaded successfully.")
    
    if args.benchmark:
        print("Loading Shakespeare dataset from Hugging Face...")
        dataloader = get_dataloader(tokenizer, batch_size=args.batch_size)
        benchmark(target_model, draft_model, tokenizer, dataloader, args)
    else:
        interactive_mode(target_model, draft_model, tokenizer, args)