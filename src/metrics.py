import torch
from torchmetrics import Metric
import time

class GenerationMetrics(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.add_state("total_latency", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tokens", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_generations", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("min_latency", default=torch.tensor(float('inf')), dist_reduce_fx="min")
        self.add_state("max_latency", default=torch.tensor(0.0), dist_reduce_fx="max")
        
    def update(self, latency, tokens_generated):
        upd_latency = torch.tensor(latency, device=self.total_latency.device)
        upd_tokens = torch.tensor(tokens_generated, device=self.total_tokens.device)
        
        self.total_latency += upd_latency
        self.total_tokens += upd_tokens
        self.num_generations += 1
        
        self.min_latency = torch.min(self.min_latency, upd_latency)
        self.max_latency = torch.max(self.max_latency, upd_latency)
    
    def compute(self):
        if self.num_generations == 0:
            return {
                "avg_latency": torch.tensor(0.0),
                "throughput": torch.tensor(0.0),
                "min_latency": torch.tensor(0.0),
                "max_latency": torch.tensor(0.0),
                "total_tokens": self.total_tokens,
                "num_generations": self.num_generations
            }
        
        avg_latency = self.total_latency / self.num_generations
        throughput = self.total_tokens / self.total_latency if self.total_latency > 0 else torch.tensor(0.0)
        
        return {
            "avg_latency": avg_latency,
            "throughput": throughput,
            "min_latency": self.min_latency,
            "max_latency": self.max_latency,
            "total_tokens": self.total_tokens,
            "num_generations": self.num_generations
        }

def measure_generation_time(generation_fn, device, *args, **kwargs):
    if "cuda" in device:
        torch.cuda.synchronize()

    start_time = time.time()
    output = generation_fn(*args, **kwargs)
    if "cuda" in device:
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    return output, elapsed_time
