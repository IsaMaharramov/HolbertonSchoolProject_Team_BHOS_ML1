"""
picker_cli.py - Headless Production First Break Picking CLI
Usage:
    python picker_cli.py --input sample_data/image_0_traces.npy --filter --export output.txt
"""
import argparse
import os
import sys
import numpy as np
import torch
from model import SeismicFirstBreakNet
from postprocess import regularize_picks
from export_picks import export_picks_to_ascii

def main():
    parser = argparse.ArgumentParser(description="Headless Seismic First Break Picker")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to traces .npy gather file")
    parser.add_argument("--model", "-m", type=str, default="first_break_picker_finetuned.pth", help="Model weights path")
    parser.add_argument("--filter", "-f", action="store_true", help="Apply geophysical cycle-skip filter")
    parser.add_argument("--export", "-e", type=str, default=None, help="Export predictions to ASCII table path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model_path = args.model if os.path.exists(args.model) else "baseline_model.pth"

    model = SeismicFirstBreakNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    traces = np.load(args.input).astype(np.float32)
    if traces.ndim != 2:
        print(f"Error: Traces array must be 2D. Got shape {traces.shape}")
        sys.exit(1)

    # Transpose to (Time, Traces)
    image = traces.T
    mean, std = np.mean(image), np.std(image)
    norm = (image - mean) / std if std != 0 else image
    tensor = torch.from_numpy(np.ascontiguousarray(norm)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor)[0].cpu().numpy().reshape(-1)

    if args.filter:
        preds = regularize_picks(preds)

    print(f"Successfully processed {len(preds)} traces on {device}.")
    print(f"Earliest arrival: {preds.min():.2f} ms | Latest arrival: {preds.max():.2f} ms")

    if args.export:
        export_picks_to_ascii(range(1, len(preds) + 1), preds, args.export)

if __name__ == "__main__":
    main()
