"""
Gradio AI Wrapper for Automated Seismic First Break Detection
Author: Togrul-cmd
Description: Interactive web interface for the seismic first break detection model.

Changes vs. previous version:
- Bias correction is OFF by default and clearly labelled as using ground truth.
  Raw (uncorrected) error is always reported when labels are provided.
- Labels are optional: the app can predict on unlabeled gathers.
- Input validation with clear error messages.
- Non-interactive matplotlib backend (avoids threading errors on Windows).
- Removed unverified performance claims and unused code.
"""

import io
import os

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot (Gradio runs in worker threads)
import matplotlib.pyplot as plt
import numpy as np
import torch
import gradio as gr
from PIL import Image

from model import SeismicFirstBreakNet

MODEL_PATH = "first_break_picker_finetuned.pth"

model = None
device = None


def load_model(model_path=MODEL_PATH):
    """Load the trained seismic first break detection model."""
    global model, device
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights '{model_path}' not found. Place the .pth file next to demo_app.py."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicFirstBreakNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded successfully on {device}")


def _path(f):
    """Gradio may pass a str path or an object with .name depending on version."""
    return getattr(f, "name", f)


def process_seismic_data(traces_file, labels_file=None, apply_bias_correction=False):
    """
    Run first-break prediction on an uploaded gather.

    Args:
        traces_file: .npy file, array of shape (num_traces, time_samples)
        labels_file: optional .npy file, array of shape (num_traces,)
        apply_bias_correction: shift predictions by mean(labels - predictions).
            This USES the ground truth, so corrected metrics are optimistic.

    Returns:
        (PIL image, markdown text)
    """
    try:
        if traces_file is None:
            return None, "❌ Please upload a seismic traces `.npy` file."

        # ---- Load and validate ----
        traces_np = np.load(_path(traces_file)).astype(np.float32)
        if traces_np.ndim != 2:
            return None, (f"❌ Traces must be a 2D array (num_traces, time_samples); "
                          f"got shape {traces_np.shape}.")
        num_traces, num_samples = traces_np.shape

        labels_np = None
        if labels_file is not None:
            labels_np = np.load(_path(labels_file)).astype(np.float32).reshape(-1)
            if labels_np.shape[0] != num_traces:
                hint = ""
                if labels_np.shape[0] == num_samples:
                    hint = " Your traces array may be transposed (expected (num_traces, time_samples))."
                return None, (f"❌ Labels length ({labels_np.shape[0]}) does not match the number "
                              f"of traces ({num_traces}).{hint}")

        # ---- Preprocess: identical to dataset.py / 05_visualize.py ----
        image = traces_np.T  # (Traces, Time) -> (Time, Traces)
        mean, std = np.mean(image), np.std(image)
        image_norm = (image - mean) / std if std != 0 else image
        tensor = torch.from_numpy(np.ascontiguousarray(image_norm)).unsqueeze(0).unsqueeze(0).to(device)

        # ---- Inference ----
        with torch.no_grad():
            raw_pred = model(tensor)[0].cpu().numpy().reshape(-1)  # (Traces,)

        # ---- Optional bias correction (uses ground truth!) ----
        pred = raw_pred
        bias_offset = None
        if labels_np is not None and apply_bias_correction:
            bias_offset = float(np.mean(labels_np - raw_pred))
            pred = raw_pred + bias_offset

        # ---- Metrics ----
        metrics_md = "_No labels uploaded, so no error metrics were computed._"
        if labels_np is not None:
            raw_err = raw_pred - labels_np
            metrics_md = (
                "**Raw model error (no correction):**\n"
                f"- MAE: {np.mean(np.abs(raw_err)):.3f}\n"
                f"- RMSE: {np.sqrt(np.mean(raw_err ** 2)):.3f}\n"
                f"- Max error: {np.max(np.abs(raw_err)):.3f}\n"
                f"- Mean signed error (bias): {np.mean(raw_err):.3f}\n"
            )
            if bias_offset is not None:
                cor_err = pred - labels_np
                metrics_md += (
                    "\n**After bias correction** (⚠️ uses ground truth, so this is optimistic):\n"
                    f"- Offset applied: {bias_offset:.3f}\n"
                    f"- MAE: {np.mean(np.abs(cor_err)):.3f}\n"
                    f"- RMSE: {np.sqrt(np.mean(cor_err ** 2)):.3f}\n"
                )
            metrics_md += "\n_Units are the same as the label units in your dataset._"

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(image, aspect="auto", cmap="gray", interpolation="bilinear")
        plt.colorbar(im, ax=ax, label="Amplitude")
        x = np.arange(len(pred))
        if labels_np is not None:
            ax.plot(x, labels_np, color="blue", linewidth=2, linestyle="--",
                    label="Ground Truth (Manual)", alpha=0.8)
        pred_label = "AI Prediction (bias-corrected)" if bias_offset is not None else "AI Prediction"
        ax.plot(x, pred, color="red", linewidth=2.5, label=pred_label, alpha=0.9)
        ax.set_title("Automated Seismic First Break Detection", fontsize=16, fontweight="bold")
        ax.set_xlabel("Trace Number", fontsize=12)
        ax.set_ylabel("Time (label units)", fontsize=12)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img.load()

        results_text = f"""
### 📊 Prediction Results

{metrics_md}

**Data information:**
- Number of traces: {num_traces}
- Time samples per trace: {num_samples}
- Device: {device}
"""
        return img, results_text

    except Exception as e:
        return None, (f"❌ Error processing data: {e}\n\n"
                      "Please check that the uploaded files are valid `.npy` arrays.")


# Initialize model on startup
load_model()

with gr.Blocks(title="Seismic First Break Detection") as demo:
    gr.Markdown("""
    # 🌊 Automated Seismic First Break Detection
    ### Deep-learning first break picking (2D CNN, PyTorch)

    **Developed by:** Togrul-cmd & Isa Maharramov

    Upload a preprocessed seismic gather (`.npy`) to get first break predictions.
    Optionally upload manual picks to compare against and compute error metrics.

    ### 📁 Input format
    - **Traces file:** NumPy array of shape `(num_traces, time_samples)`
    - **Labels file (optional):** NumPy array of shape `(num_traces,)` with manual first break picks
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Data")
            traces_input = gr.File(label="Seismic Traces (.npy)", file_types=[".npy"], type="filepath")
            labels_input = gr.File(label="Ground Truth Labels (.npy) — optional",
                                   file_types=[".npy"], type="filepath")
            bias_correction = gr.Checkbox(
                label="Apply bias correction (requires labels)",
                value=False,
                info="Shifts predictions by the mean offset from the labels. "
                     "Uses ground truth, so corrected metrics are optimistic.",
            )
            process_btn = gr.Button("🚀 Run Inference", variant="primary", size="lg")

            gr.Markdown("""
            ---
            ### ℹ️ About the model
            Treats seismic gathers as 2D images so the network can use the continuity
            of the wavefront across neighboring traces.

            **Training data:** Brunswick, Halfmile Lake, Lalor
            **Validation:** unseen Sudbury site

            **Architecture:**
            - 4 convolutional blocks (16→32→64→128 filters)
            - Asymmetric kernels along the time / trace axes
            - AdaptiveAvgPool2d for variable trace counts
            - Per-trace regression head, L1 (MAE) loss
            """)

        with gr.Column(scale=2):
            gr.Markdown("### 📈 Visualization & Results")
            output_image = gr.Image(label="First Break Detection Results", type="pil")
            output_text = gr.Markdown()

    process_btn.click(
        fn=process_seismic_data,
        inputs=[traces_input, labels_input, bias_correction],
        outputs=[output_image, output_text],
    )

    gr.Markdown("""
    ---
    [GitHub Repository](https://github.com/IsaMaharramov/HolbertonSchoolProject_Team_BHOS_ML1)
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, show_error=True)
