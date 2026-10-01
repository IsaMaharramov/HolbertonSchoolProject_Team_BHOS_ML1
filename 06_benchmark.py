"""
06_benchmark.py - Multi-Gather Quantitative Evaluation & Tolerance Benchmark
Evaluates the fine-tuned model across all available validation gathers without retraining.
"""
import os
import glob
import numpy as np
import torch
from model import SeismicFirstBreakNet

def run_benchmark(data_dir="sample_data", model_path="first_break_picker_finetuned.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running seismic benchmark on: {device}")
    
    if not os.path.exists(model_path):
        model_path = "baseline_model.pth"
        
    model = SeismicFirstBreakNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    trace_files = sorted(glob.glob(os.path.join(data_dir, "*_traces.npy")))
    label_files = sorted(glob.glob(os.path.join(data_dir, "*_labels.npy")))

    if not trace_files:
        print(f"No validation data found in '{data_dir}'.")
        return

    all_errors = []
    all_mape_errors = []
    gather_results = []

    for t_file, l_file in zip(trace_files, label_files):
        gather_name = os.path.basename(t_file).replace("_traces.npy", "")
        traces = np.load(t_file).astype(np.float32).T  # (Traces, Time) -> (Time, Traces)
        labels = np.load(l_file).astype(np.float32).reshape(-1)

        mean, std = np.mean(traces), np.std(traces)
        norm = (traces - mean) / std if std != 0 else traces
        tensor = torch.from_numpy(np.ascontiguousarray(norm)).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(tensor)[0].cpu().numpy().reshape(-1)

        err = np.abs(preds - labels)
        all_errors.extend(err.tolist())

        rel_err = np.abs(preds - labels) / np.maximum(labels, 1e-6)
        all_mape_errors.extend(rel_err.tolist())

        mae = float(np.mean(err))
        rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
        mape = float(np.mean(rel_err) * 100.0)
        gather_results.append((gather_name, len(labels), mae, rmse, mape))

    all_errors = np.array(all_errors)
    all_mape_errors = np.array(all_mape_errors)
    total_picks = len(all_errors)

    # Industry acceptance thresholds
    within_5ms = (np.sum(all_errors <= 5.0) / total_picks) * 100
    within_10ms = (np.sum(all_errors <= 10.0) / total_picks) * 100
    within_15ms = (np.sum(all_errors <= 15.0) / total_picks) * 100

    print("\n" + "=" * 65)
    print("        SEISMIC FIRST BREAK PICKING BENCHMARK REPORT         ")
    print("=" * 65)
    print(f"{'Gather Name':<20} | {'Traces':<8} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10}")
    print("-" * 75)
    for g_name, count, mae, rmse, mape in gather_results:
        print(f"{g_name:<20} | {count:<8} | {mae:<10.3f} | {rmse:<10.3f} | {mape:<10.2f}%")
    print("-" * 75)
    print(f"Total Evaluated Picks: {total_picks}")
    print(f"Global MAE:            {np.mean(all_errors):.3f} ms")
    print(f"Global RMSE:           {np.sqrt(np.mean(all_errors**2)):.3f} ms")
    print(f"Global MAPE:           {np.mean(all_mape_errors) * 100.0:.2f}%")
    print("\n--- Industry Tolerance Thresholds ---")
    print(f"Picks within +/- 5 ms:   {within_5ms:.2f}%")
    print(f"Picks within +/- 10 ms:  {within_10ms:.2f}%")
    print(f"Picks within +/- 15 ms:  {within_15ms:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    run_benchmark()