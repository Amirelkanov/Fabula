from sampling import autoregressive_sampling, speculative_sampling
from helpers import show_results_and_update_metrics
from metrics import GenerationMetrics, measure_generation_time
from datamodule import WikiTextV2Datamodule
from tqdm import tqdm
import numpy as np
    

def interactive_mode(target_model, draft_model, tokenizer, args):
    print("Enter a prompt (type 'exit' or press CTRL+C to quit):")
    
    auto_metrics, spec_metrics = GenerationMetrics(), GenerationMetrics()
    
    try:
        while True:
            prompt_text = input("> ")
            if prompt_text.lower() == 'exit':
                break
            
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(args.device)
            
            print("\nRunning autoregressive sampling...")
            auto_output, auto_time = measure_generation_time(
                autoregressive_sampling,
                args.device,
                target_model, 
                input_ids, 
                args.max_tokens + input_ids.shape[1],
                temperature=args.temperature
            )
            
            show_results_and_update_metrics(
                input_ids, auto_output, auto_time, auto_metrics, tokenizer
            )
            
            print("\nRunning speculative sampling...")
            spec_output, spec_time = measure_generation_time(
                speculative_sampling,
                args.device,
                target_model, 
                draft_model, 
                input_ids, 
                args.max_tokens + input_ids.shape[1],
                lookahead_k=args.lookahead_k,
                temperature=args.temperature
            )
            
            show_results_and_update_metrics(
                input_ids, spec_output, spec_time, spec_metrics, tokenizer
            )
            
            print(f"Speedup: {(auto_time / spec_time):.2f}x")
    
    except KeyboardInterrupt:
        print("\nInterrupt received, exiting...")

def benchmark_mode(target_model, draft_model, tokenizer, args):
    print(f"\nRunning benchmark with {args.num_batches} batches, batch size {args.batch_size}, seed: {args.seed}.")
    
    print("Loading dataset...")
    data_module = WikiTextV2Datamodule(min_len=args.min_prompt_len, max_len=args.max_prompt_len, batch_size=args.batch_size)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    
    auto_metrics, spec_metrics = GenerationMetrics(), GenerationMetrics()

    speedups = []
    auto_latencies, auto_throughputs = [], []
    spec_latencies, spec_throughputs = [], []
    
    print("Starting benchmark...")
    for batch in tqdm(range(args.num_batches)):
        batch = next(iter(test_loader))
        texts = batch["text"]
        for text in texts:
            input_ids = tokenizer.encode(text, return_tensors="pt").to(args.device)
            
            target_length = input_ids.shape[1] + args.max_tokens
            
            auto_output, auto_time = measure_generation_time(
                autoregressive_sampling,
                args.device,
                target_model, 
                input_ids, 
                target_length,
                temperature=args.temperature
            )
            
            auto_tokens_generated = auto_output.shape[1] - input_ids.shape[1]
            auto_metrics.update(auto_time, auto_tokens_generated)
            
            auto_latencies.append(auto_time)
            auto_throughputs.append(auto_tokens_generated / auto_time)
            
            spec_output, spec_time = measure_generation_time(
                speculative_sampling,
                args.device,
                target_model, 
                draft_model, 
                input_ids, 
                target_length,
                lookahead_k=args.lookahead_k,
                temperature=args.temperature
            )
            
            spec_tokens_generated = spec_output.shape[1] - input_ids.shape[1]
            spec_metrics.update(spec_time, spec_tokens_generated)
            
            spec_latencies.append(spec_time)
            spec_throughputs.append(spec_tokens_generated / spec_time)
            
            speedup = auto_time / spec_time
            speedups.append(speedup)
    
    auto_results, spec_results = auto_metrics.compute(), spec_metrics.compute()
    
    overall_speedup = auto_results["avg_latency"] / spec_results["avg_latency"]
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    
    print("\nAutoregressive Sampling:")
    print(f"  Average latency: {auto_results['avg_latency']:.4f} seconds")
    print(f"  Throughput: {auto_results['throughput']:.2f} tokens/second")
    print(f"  Min latency: {auto_results['min_latency']:.4f} seconds")
    print(f"  Max latency: {auto_results['max_latency']:.4f} seconds")
    print(f"  Total tokens generated: {auto_results['total_tokens'].item()}")
    
    print("\nSpeculative Sampling:")
    print(f"  Average latency: {spec_results['avg_latency']:.4f} seconds")
    print(f"  Throughput: {spec_results['throughput']:.2f} tokens/second")
    print(f"  Min latency: {spec_results['min_latency']:.4f} seconds")
    print(f"  Max latency: {spec_results['max_latency']:.4f} seconds")
    print(f"  Total tokens generated: {spec_results['total_tokens'].item()}")
    
    print("\nPerformance Comparison:")
    print(f"  Overall speedup: {overall_speedup:.2f}x")
    print(f"  Speedup statistics:")
    print(f"    Mean: {np.mean(speedups):.2f}x")
    print(f"    Median: {np.median(speedups):.2f}x")
    print(f"    Std Dev: {np.std(speedups):.2f}")
    print(f"    Min: {np.min(speedups):.2f}x")
    print(f"    Max: {np.max(speedups):.2f}x")
    
    throughput_improvement = ((spec_results['throughput'] / auto_results['throughput']) - 1) * 100
    print(f"  Throughput improvement: {throughput_improvement:.2f}%")
    
    print("\n" + "="*50)
    
    return {
        'auto_results': auto_results,
        'spec_results': spec_results,
        'overall_speedup': overall_speedup,
        'speedups': speedups,
        'auto_latencies': auto_latencies,
        'spec_latencies': spec_latencies,
        'auto_throughputs': auto_throughputs,
        'spec_throughputs': spec_throughputs
    }