from sampling import autoregressive_sampling, speculative_sampling
from helpers import show_results_and_update_metrics

import time

import time
from metrics import GenerationMetrics, measure_generation_time


def interactive_mode(target_model, draft_model, tokenizer, args):
    print("\n===== Interactive Mode =====")
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

# TODO
"""def benchmark(target_model, draft_model, tokenizer, dataloader, args):
    print("\n===== Running Benchmark =====")
    auto_latencies = []
    spec_latencies = []
    auto_throughputs = []
    spec_throughputs = []
    
    max_samples = min(args.num_samples, len(dataloader))
    sample_count = 0
    
    for batch in dataloader:
        if sample_count >= max_samples:
            break
            
        sample_count += 1
        input_ids = batch['input_ids'].to(args.device)

        start_time = time.time()
        auto_output = autoregressive_sampling(
            target_model, 
            input_ids, 
            args.max_tokens + input_ids.shape[1],
            temperature=args.temperature
        )
        auto_time = time.time() - start_time
        auto_tokens_generated = auto_output.shape[1] - input_ids.shape[1]
        auto_throughput = auto_tokens_generated / auto_time  
        
        auto_latencies.append(auto_time)
        auto_throughputs.append(auto_throughput)
        
        start_time = time.time()
        spec_output = speculative_sampling(
            target_model, 
            draft_model, 
            input_ids, 
            args.max_tokens + input_ids.shape[1],
            lookahead_k=args.lookahead_k,
            temperature=args.temperature
        )
        spec_time = time.time() - start_time
        spec_tokens_generated = spec_output.shape[1] - input_ids.shape[1]
        spec_throughput = spec_tokens_generated / spec_time
        
        spec_latencies.append(spec_time)
        spec_throughputs.append(spec_throughput)
        
        print(f"\nSample text: {tokenizer.decode(input_ids[0], skip_special_tokens=True)[:50]}...")
        print(f"Generated {auto_tokens_generated} tokens")
        print("\nAutoregressive Results:")
        print(f"  Latency: {auto_time:.4f} seconds")
        print(f"  Throughput: {auto_throughput:.2f} tokens/second")
        print("\nSpeculative Results:")
        print(f"  Latency: {spec_time:.4f} seconds")
        print(f"  Throughput: {spec_throughput:.2f} tokens/second")
        print(f"  Speedup: {auto_time/spec_time:.2f}x")
    
    print("\n===== Benchmark Summary =====")
    print(f"Number of samples: {len(auto_latencies)}")
    print("\nAutoregressive:")
    print(f"  Avg Latency: {statistics.mean(auto_latencies):.4f} seconds")
    print(f"  Avg Throughput: {statistics.mean(auto_throughputs):.2f} tokens/second")
    print("\nSpeculative:")
    print(f"  Avg Latency: {statistics.mean(spec_latencies):.4f} seconds")
    print(f"  Avg Throughput: {statistics.mean(spec_throughputs):.2f} tokens/second")
    print(f"  Avg Speedup: {statistics.mean(auto_latencies)/statistics.mean(spec_latencies):.2f}x")
"""