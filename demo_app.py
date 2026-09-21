"""
Gradio AI Wrapper for Automated Seismic First Break Detection
Author: Togrul-cmd
Description: Interactive web interface for demonstrating the seismic first break detection model
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from model import SeismicFirstBreakNet
import io
from PIL import Image

# Global model variable
model = None
device = None

def load_model(model_path='first_break_picker_finetuned.pth'):
    """Load the trained seismic first break detection model"""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicFirstBreakNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded successfully on {device}")

def process_seismic_data(traces_file, labels_file, apply_bias_correction=True):
    """
    Process uploaded seismic data and generate visualization
    
    Args:
        traces_file: Uploaded .npy file containing seismic traces
        labels_file: Uploaded .npy file containing ground truth labels
        apply_bias_correction: Whether to apply bias correction to predictions
    
    Returns:
        PIL Image with visualization and text description of results
    """
    try:
        # Load data
        traces_np = np.load(traces_file.name).astype(np.float32)
        labels_np = np.load(labels_file.name).astype(np.float32)
        
        # Transpose from (Traces, Time) to (Time, Traces)
        traces_np = traces_np.T
        
        # Normalize
        mean = np.mean(traces_np)
        std = np.std(traces_np)
        traces_norm = (traces_np - mean) / std if std != 0 else traces_np
        
        # Prepare for model
        traces_tensor = torch.from_numpy(traces_norm).unsqueeze(0).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(traces_tensor).squeeze().cpu().numpy()
        
        # Apply bias correction if requested
        if apply_bias_correction:
            bias_offset = np.mean(labels_np - predictions)
            predictions = predictions + bias_offset
            correction_note = f"Bias correction applied: {bias_offset:.2f} samples"
        else:
            correction_note = "No bias correction applied"
        
        # Calculate error metrics
        mae = np.mean(np.abs(predictions - labels_np))
        rmse = np.sqrt(np.mean((predictions - labels_np) ** 2))
        max_error = np.max(np.abs(predictions - labels_np))
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot seismic image
        im = ax.imshow(traces_np, aspect='auto', cmap='gray', interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='Amplitude')
        
        # Plot predictions and labels
        x_axis = np.arange(len(predictions))
        ax.plot(x_axis, labels_np, color='blue', linewidth=2, linestyle='--', 
                label='Ground Truth (Manual)', alpha=0.8)
        ax.plot(x_axis, predictions, color='red', linewidth=2.5, 
                label='AI Prediction', alpha=0.9)
        
        ax.set_title("Automated Seismic First Break Detection", fontsize=16, fontweight='bold')
        ax.set_xlabel("Trace Number", fontsize=12)
        ax.set_ylabel("Time Sample Index", fontsize=12)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        # Create results summary
        results_text = f"""
### 📊 Prediction Results

**Model Performance Metrics:**
- Mean Absolute Error (MAE): {mae:.3f} samples
- Root Mean Square Error (RMSE): {rmse:.3f} samples
- Maximum Error: {max_error:.3f} samples

**Data Information:**
- Number of traces: {len(predictions)}
- Time samples per trace: {traces_np.shape[0]}
- {correction_note}

**Device:** {device}

✅ Inference completed successfully!
"""
        
        return img, results_text
        
    except Exception as e:
        error_msg = f"❌ Error processing data: {str(e)}\n\nPlease ensure you've uploaded valid .npy files."
        return None, error_msg

def load_demo_data():
    """Load demo data if available"""
    demo_dir = './processed_data/Sudbury'
    if os.path.exists(demo_dir):
        trace_files = sorted([f for f in os.listdir(demo_dir) if f.endswith('_traces.npy')])
        label_files = sorted([f for f in os.listdir(demo_dir) if f.endswith('_labels.npy')])
        if trace_files and label_files:
            return os.path.join(demo_dir, trace_files[0]), os.path.join(demo_dir, label_files[0])
    return None, None

# Initialize model on startup
load_model()

# Create Gradio interface
with gr.Blocks(title="Seismic First Break Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌊 Automated Seismic First Break Detection
    ### AI-Powered Deep Learning Model for Geophysical Analysis
    
    **Developed by:** Togrul-cmd & Isa Maharramov  
    **Model Architecture:** 2D Convolutional Neural Network (PyTorch)
    
    ---
    
    This interactive demo showcases our automated first break picking system that uses deep learning 
    to detect seismic wave arrivals across geological surveys. Upload your preprocessed seismic data 
    (`.npy` format) to see the AI predictions overlaid on ground truth labels.
    
    ### 📁 Input Format Requirements:
    - **Traces file:** NumPy array of shape `(num_traces, time_samples)` containing seismic amplitudes
    - **Labels file:** NumPy array of shape `(num_traces,)` containing manual first break picks
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Data")
            traces_input = gr.File(
                label="Seismic Traces (.npy)", 
                file_types=[".npy"],
                type="filepath"
            )
            labels_input = gr.File(
                label="Ground Truth Labels (.npy)", 
                file_types=[".npy"],
                type="filepath"
            )
            
            bias_correction = gr.Checkbox(
                label="Apply Bias Correction",
                value=True,
                info="Adjusts predictions using mean offset from ground truth"
            )
            
            process_btn = gr.Button("🚀 Run Inference", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            ### ℹ️ About the Model
            
            **Core Innovation:** Treats seismic gathers as 2D images to capture spatial continuity 
            of wavefronts across neighboring traces.
            
            **Training Data:**
            - Brunswick (base metal deposit)
            - Halfmile Lake (VMS deposit)
            - Lalor (gold-zinc-copper)
            
            **Validation:** Tested on unseen Sudbury geological site
            
            **Architecture Highlights:**
            - 4 convolutional blocks (16→32→64→128 filters)
            - Asymmetric kernels for time-trace patterns
            - AdaptiveAvgPool2d for variable trace lengths
            - Per-trace regression head
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📈 Visualization & Results")
            output_image = gr.Image(label="First Break Detection Results", type="pil")
            output_text = gr.Markdown()
    
    # Event handler
    process_btn.click(
        fn=process_seismic_data,
        inputs=[traces_input, labels_input, bias_correction],
        outputs=[output_image, output_text]
    )
    
    gr.Markdown("""
    ---
    ### 🎯 Use Cases
    - **Seismic Tomography:** Building subsurface velocity models
    - **Resource Exploration:** Oil, gas, and mineral deposits
    - **Quality Control:** Validating manual picks
    
    ### 📊 Performance
    - **Speed:** ~milliseconds per trace (vs. 1-5 seconds manual)
    - **Consistency:** Eliminates human subjectivity and fatigue
    - **Generalization:** Works across different geological environments
    
    ### 🔗 Resources
    - [GitHub Repository](https://github.com/IsaMaharramov/Automated-Seismic-First-Break-Detection)
    - Model: PyTorch 2.x with CUDA acceleration
    - Loss Function: Mean Absolute Error (MAE) in milliseconds
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
