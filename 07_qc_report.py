"""
07_qc_report.py - Multi-Panel Geophysical Quality Control Report Generator
Generates publication-quality diagnostic plots comparing raw and post-processed arrivals.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import SeismicFirstBreakNet
from postprocess import regularize_picks

def generate_qc_report(trace_path="sample_data/image_0_traces.npy",
                       label_path="sample_data/image_0_labels.npy",
                       model_path="first_break_picker_finetuned.pth",
                       output_figure="qc_analysis_report.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        model_path = "baseline_model.pth"

    model = SeismicFirstBreakNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    traces_raw = np.load(trace_path).astype(np.float32)
    labels = np.load(label_path).astype(np.float32).reshape(-1)
    traces_t = traces_raw.T

    mean, std = np.mean(traces_t), np.std(traces_t)
    norm = (traces_t - mean) / std if std != 0 else traces_t
    tensor = torch.from_numpy(np.ascontiguousarray(norm)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_preds = model(tensor)[0].cpu().numpy().reshape(-1)

    filtered_preds = regularize_picks(raw_preds, max_velocity_jump=12.0, median_window=5)

    residuals = raw_preds - labels
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # Compute R² score
    ss_res = np.sum((labels - raw_preds) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2_score = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: Gather overlay
    ax1 = axes[0, 0]
    ax1.imshow(traces_t, aspect="auto", cmap="gray", interpolation="bilinear")
    x = np.arange(len(labels))
    ax1.plot(x, labels, color="cyan", linestyle="--", linewidth=1.5, label="Ground Truth")
    ax1.plot(x, raw_preds, color="red", alpha=0.8, linewidth=1.5, label=f"Raw AI (MAE={mae:.2f}ms)")
    ax1.plot(x, filtered_preds, color="lime", alpha=0.9, linewidth=1.8, label="Geophysically Filtered")
    ax1.set_title("Seismic Gather & Arrival Comparison", fontweight="bold")
    ax1.set_xlabel("Trace Index")
    ax1.set_ylabel("Time Sample")
    ax1.legend(loc="upper right", framealpha=0.8)

    # Panel 2: Error residual distribution
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=25, color="royalblue", edgecolor="black", alpha=0.7)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1)
    ax2.axvline(np.mean(residuals), color="red", label=f"Bias: {np.mean(residuals):.2f} ms")
    ax2.set_title(f"Residual Distribution (RMSE: {rmse:.2f} ms)", fontweight="bold")
    ax2.set_xlabel("Residual (Predicted - Label) [ms]")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Trace-by-trace absolute error profile
    ax3 = axes[1, 0]
    ax3.plot(x, np.abs(residuals), color="darkorange", label="Raw Trace Error")
    ax3.plot(x, np.abs(filtered_preds - labels), color="green", label="Post-Filtered Error")
    ax3.axhline(5.0, color="gray", linestyle=":", label="±5 ms Threshold")
    ax3.set_title("Per-Trace Error Profile Across Spread", fontweight="bold")
    ax3.set_xlabel("Trace Index")
    ax3.set_ylabel("Absolute Error [ms]")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Parity / Correlation scatter plot
    ax4 = axes[1, 1]
    ax4.scatter(labels, raw_preds, color="purple", alpha=0.6, edgecolors="none", s=30)
    min_val = min(labels.min(), raw_preds.min())
    max_val = max(labels.max(), raw_preds.max())
    ax4.plot([min_val, max_val], [min_val, max_val], "k--", label="Ideal 1:1 Agreement")
    ax4.set_title(f"Pick Correlation (R² = {r2_score:.4f})", fontweight="bold")
    ax4.set_xlabel("Manual Label [ms]")
    ax4.set_ylabel("AI Prediction [ms]")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_figure, dpi=300)
    plt.close(fig)
    print(f"Quality Control Report successfully generated: {output_figure}")

if __name__ == "__main__":
    generate_qc_report()
