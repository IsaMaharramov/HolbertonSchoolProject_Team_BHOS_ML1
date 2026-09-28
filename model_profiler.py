"""
model_profiler.py - Production Hardware & Latency Benchmark
Measures model complexity, parameter counts, memory usage, and throughput.
"""
import time
import torch
from model import SeismicFirstBreakNet

def profile_model(num_runs=100, batch_size=1, time_samples=1500, traces=120):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicFirstBreakNet().to(device)
    model.eval()

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy_input = torch.randn(batch_size, 1, time_samples, traces).to(device)

    # Warm-up runs
    for _ in range(15):
        with torch.no_grad():
            _ = model(dummy_input)

    # Synchronize if GPU
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed inference loop
    start_time = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - start_time

    avg_latency_ms = (total_elapsed / num_runs) * 1000
    fps = num_runs / total_elapsed
    trace_throughput = (num_runs * traces) / total_elapsed

    print("\n" + "=" * 55)
    print("        SEISMIC FIRST BREAK NET PROFILING REPORT       ")
    print("=" * 55)
    print(f"Device:                 {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'Host CPU'})")
    print(f"Input Dimension:        (Batch={batch_size}, 1, Time={time_samples}, Traces={traces})")
    print(f"Total Parameters:       {total_params:,}")
    print(f"Trainable Parameters:   {trainable_params:,}")
    print(f"Model File Size Approx: ~{total_params * 4 / (1024 * 1024):.2f} MB (float32)")
    print("-" * 55)
    print(f"Average Latency:        {avg_latency_ms:.3f} ms / gather")
    print(f"Inference Speed:        {fps:.2f} gathers / sec")
    print(f"Picking Throughput:     {trace_throughput:,.1f} traces / sec")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    profile_model()
