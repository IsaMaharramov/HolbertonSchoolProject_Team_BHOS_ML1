import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SeismicFirstBreakNet

def visualize_results(model_path, data_dir, image_index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    model = SeismicFirstBreakNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Grab the first available test image and its labels
    trace_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_traces.npy')])
    label_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_labels.npy')])
    
    traces_np = np.load(os.path.join(data_dir, trace_files[image_index])).astype(np.float32)
    labels_np = np.load(os.path.join(data_dir, label_files[image_index])).astype(np.float32)

    # FLIP THE AXES: Convert from (Traces, Time) to (Time, Traces) to match training!
    traces_np = traces_np.T

    # Normalize trace just like we do in the Dataset class
    mean = np.mean(traces_np)
    std = np.std(traces_np)
    traces_norm = (traces_np - mean) / std if std != 0 else traces_np

    # Format for the PyTorch model (Batch=1, Channel=1, Time, Traces)
    traces_tensor = torch.from_numpy(traces_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Predict the first break times
        predictions = model(traces_tensor).squeeze().cpu().numpy()

    # Set up the plot for your presentation
    plt.figure(figsize=(12, 8))
    
    # Plot the 2D seismic image in grayscale
    plt.imshow(traces_np, aspect='auto', cmap='gray')
    
    # X-axis represents the individual traces
    x_axis = np.arange(len(predictions))
    
    # Plot Ground Truth (Blue) and Model Prediction (Red)
    plt.plot(x_axis, labels_np, color='blue', linewidth=1, linestyle='--', label='Manual Label')
    plt.plot(x_axis, predictions, color='red', linewidth=2, label='AI Prediction')
    
    plt.title("Automated Seismic First Break Detection")
    plt.xlabel("Seismic Trace")
    plt.ylabel("Time / Sample Index")
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig("presentation_figure.png", dpi=300)
    print("Saved visualization to presentation_figure.png!")
    plt.show()

if __name__ == "__main__":
    # Pointed to the newly fine-tuned weights!
    visualize_results('first_break_picker_finetuned.pth', './processed_data/Sudbury')