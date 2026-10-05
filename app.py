"""
Gradio AI Wrapper for Automated Seismic First Break Detection
Author: Togrul-cmd & Isa Maharramov
Description: Interactive web interface with model fallback loading,
geophysical post-processing filter, ASCII export, and 1-click examples.
"""

import io
import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import gradio as gr
from PIL import Image

from model import SeismicFirstBreakNet
from postprocess import regularize_picks


model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model_path = None


def load_model():
    """Load the trained seismic first break detection model with fallback support."""
    global model, device, loaded_model_path

    candidates = [
        "first_break_picker_finetuned.pth",
        "first_break_picker.pth",
        "baseline_model.pth",
    ]

    model_path = next((p for p in candidates if os.path.exists(p) and os.path.getsize(p) > 1024), None)

    if not model_path:
        print("[WARNING] No valid .pth weight file found. Waiting for manual upload or Git LFS sync.")
        return False

    try:
        net = SeismicFirstBreakNet().to(device)
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        net.eval()
        model = net
        loaded_model_path = model_path
        print(f"Model loaded successfully from '{model_path}' on {device}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed loading weights from {model_path}: {e}")
        return False

def _path(file):
    """Return the filesystem path from a Gradio file value."""
    return getattr(file, "name", file)


def process_seismic_data(
    traces_file,
    labels_file=None,
    apply_bias_correction=False,
    apply_filter=False,
):
    """
    Run first-break prediction on an uploaded seismic gather.

    Args:
        traces_file:
            .npy file containing an array of shape
            (num_traces, time_samples).

        labels_file:
            Optional .npy file containing manual picks
            with shape (num_traces,).

        apply_bias_correction:
            Shift predictions by the mean difference between
            labels and predictions.

            WARNING:
            This uses ground truth labels and therefore produces
            optimistic corrected metrics.

        apply_filter:
            Apply geophysical spatial continuity regularization.

    Returns:
        PIL image,
        Markdown result text,
        path to downloadable ASCII pick file.
    """

    try:
        if traces_file is None:
            return (
                None,
                "❌ Please upload a seismic traces `.npy` file.",
                None,
            )

        if model is None and not load_model():
            return (
                None,
                "❌ **Model weights not loaded.** Ensure `.pth` weights are uploaded via Git LFS.",
                None,
            )

        # ---------------------------------------------------------
        # Load and validate seismic traces
        # ---------------------------------------------------------

        traces_np = np.load(
            _path(traces_file)
        ).astype(np.float32)

        if traces_np.ndim != 2:
            return (
                None,
                (
                    "❌ Traces must be a 2D array "
                    "(num_traces, time_samples); "
                    f"got shape {traces_np.shape}."
                ),
                None,
            )

        num_traces, num_samples = traces_np.shape

        # ---------------------------------------------------------
        # Load and validate labels
        # ---------------------------------------------------------

        labels_np = None

        if labels_file is not None:
            labels_np = np.load(
                _path(labels_file)
            ).astype(np.float32).reshape(-1)

            if labels_np.shape[0] != num_traces:
                hint = ""

                if labels_np.shape[0] == num_samples:
                    hint = (
                        " Your traces array may be transposed "
                        "(expected (num_traces, time_samples))."
                    )

                return (
                    None,
                    (
                        f"❌ Labels length ({labels_np.shape[0]}) "
                        f"does not match the number of traces "
                        f"({num_traces}).{hint}"
                    ),
                    None,
                )

        # ---------------------------------------------------------
        # Preprocessing
        # Same orientation/normalization used by dataset.py
        # and visualization pipeline.
        # ---------------------------------------------------------

        image = traces_np.T

        mean = np.mean(image)
        std = np.std(image)

        if std != 0:
            image_norm = (image - mean) / std
        else:
            image_norm = image

        tensor = (
            torch.from_numpy(
                np.ascontiguousarray(image_norm)
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        # ---------------------------------------------------------
        # Model inference
        # ---------------------------------------------------------

        with torch.no_grad():
            raw_pred = (
                model(tensor)[0]
                .cpu()
                .numpy()
                .reshape(-1)
            )

        # ---------------------------------------------------------
        # Geophysical spatial continuity filter
        # ---------------------------------------------------------

        if apply_filter:
            raw_pred = regularize_picks(
                raw_pred,
                max_velocity_jump=15.0,
                median_window=5,
            )

        # ---------------------------------------------------------
        # Optional bias correction
        # ---------------------------------------------------------

        pred = raw_pred
        bias_offset = None

        if labels_np is not None and apply_bias_correction:
            bias_offset = float(
                np.mean(labels_np - raw_pred)
            )

            pred = raw_pred + bias_offset

        # ---------------------------------------------------------
        # Calculate metrics
        # ---------------------------------------------------------

        metrics_md = (
            "_No labels uploaded, so no error metrics "
            "were computed._"
        )

        if labels_np is not None:
            raw_err = raw_pred - labels_np

            raw_mae = np.mean(
                np.abs(raw_err)
            )

            raw_rmse = np.sqrt(
                np.mean(raw_err ** 2)
            )

            raw_mape = (
                np.mean(
                    np.abs(raw_err)
                    / np.maximum(labels_np, 1e-6)
                )
                * 100.0
            )

            max_error = np.max(
                np.abs(raw_err)
            )

            mean_bias = np.mean(raw_err)

            metrics_md = (
                "**Model error metrics:**\n"
                f"- MAE: {raw_mae:.3f} ms\n"
                f"- RMSE: {raw_rmse:.3f} ms\n"
                f"- MAPE: {raw_mape:.2f}%\n"
                f"- Max error: {max_error:.3f} ms\n"
                f"- Mean signed error (bias): "
                f"{mean_bias:.3f} ms\n"
            )

            # -----------------------------------------------------
            # Corrected metrics
            # -----------------------------------------------------

            if bias_offset is not None:
                cor_err = pred - labels_np

                cor_mae = np.mean(
                    np.abs(cor_err)
                )

                cor_rmse = np.sqrt(
                    np.mean(cor_err ** 2)
                )

                cor_mape = (
                    np.mean(
                        np.abs(cor_err)
                        / np.maximum(labels_np, 1e-6)
                    )
                    * 100.0
                )

                metrics_md += (
                    "\n"
                    "**After bias correction** "
                    "(⚠️ uses ground truth, optimistic):\n"
                    f"- Offset applied: "
                    f"{bias_offset:.3f} ms\n"
                    f"- MAE: {cor_mae:.3f} ms\n"
                    f"- RMSE: {cor_rmse:.3f} ms\n"
                    f"- MAPE: {cor_mape:.2f}%\n"
                )

            metrics_md += (
                "\n"
                "_Units are in milliseconds (ms) / "
                "percentage (%)._"
            )

        # ---------------------------------------------------------
        # Create visualization
        # ---------------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(14, 8)
        )

        im = ax.imshow(
            image,
            aspect="auto",
            cmap="gray",
            interpolation="bilinear",
        )

        plt.colorbar(
            im,
            ax=ax,
            label="Amplitude",
        )

        x = np.arange(
            len(pred)
        )

        # Ground truth
        if labels_np is not None:
            ax.plot(
                x,
                labels_np,
                color="blue",
                linewidth=2,
                linestyle="--",
                label="Ground Truth (Manual)",
                alpha=0.8,
            )

        # Prediction
        if bias_offset is not None:
            pred_label = (
                "AI Prediction (bias-corrected)"
            )
        else:
            pred_label = "AI Prediction"

        ax.plot(
            x,
            pred,
            color="red",
            linewidth=2.5,
            label=pred_label,
            alpha=0.9,
        )

        ax.set_title(
            "Automated Seismic First Break Detection",
            fontsize=16,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Trace Number",
            fontsize=12,
        )

        ax.set_ylabel(
            "Time (label units)",
            fontsize=12,
        )

        ax.legend(
            loc="upper right",
            fontsize=11,
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        plt.tight_layout()

        # ---------------------------------------------------------
        # Convert matplotlib figure to PIL image
        # ---------------------------------------------------------

        buf = io.BytesIO()

        fig.savefig(
            buf,
            format="png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        buf.seek(0)

        img = Image.open(buf)
        img.load()

       # ---------------------------------------------------------
        # Export predicted picks as ASCII (Session-Isolated)
        # ---------------------------------------------------------

        fd, export_file = tempfile.mkstemp(prefix="picks_", suffix=".txt")

        with os.fdopen(fd, "w") as f:
            f.write("# SEISMIC FIRST BREAK AUTOMATED ARRIVAL PICKS\n")
            f.write("# Format: Trace_ID    Pick_Time_ms\n")
            f.write("# ----------------------------------------\n")
            for idx, pick in enumerate(pred, start=1):
                f.write(f"{idx:<12}\t{float(pick):<15.3f}\n")

        # ---------------------------------------------------------
        # Result information
        # ---------------------------------------------------------

        if apply_filter:
            filter_status = (
                "Enabled (Median + Jump Filter)"
            )
        else:
            filter_status = (
                "Disabled (Raw Model Output)"
            )

        results_text = f"""
### 📊 Prediction Results

{metrics_md}

**Data information:**

- Number of traces: {num_traces}
- Time samples per trace: {num_samples}
- Wavefront regularization: {filter_status}
- Device: {device}
"""

        return (
            img,
            results_text,
            export_file,
        )

    except Exception as e:
        return (
            None,
            (
                f"❌ Error processing data: {e}\n\n"
                "Please check that the uploaded files "
                "are valid `.npy` arrays."
            ),
            None,
        )


# =============================================================
# Initialize model
# =============================================================

load_model()


# =============================================================
# Gradio interface
# =============================================================

with gr.Blocks(
    title="Seismic First Break Detection"
) as demo:

    gr.Markdown(
        """
# 🌊 Automated Seismic First Break Detection

### Deep-learning first break picking (2D CNN, PyTorch)

**Developed by:** Togrul-cmd & Isa Maharramov

Upload a preprocessed seismic gather (`.npy`)
to get first break predictions.

Optionally upload manual picks to compare against
and compute error metrics.

### 📁 Input format

- **Traces file:** NumPy array of shape
  `(num_traces, time_samples)`
- **Labels file (optional):** NumPy array of shape
  `(num_traces,)` with manual first break picks
"""
    )

    with gr.Row():

        # =====================================================
        # Input column
        # =====================================================

        with gr.Column(scale=1):

            gr.Markdown(
                "### 📤 Upload Data"
            )

            traces_input = gr.File(
                label="Seismic Traces (.npy)",
                file_types=[".npy"],
                type="filepath",
            )

            labels_input = gr.File(
                label="Ground Truth Labels (.npy) — optional",
                file_types=[".npy"],
                type="filepath",
            )

            bias_correction = gr.Checkbox(
                label=(
                    "Apply bias correction "
                    "(requires labels)"
                ),
                value=False,
                info=(
                    "Shifts predictions by the mean "
                    "offset from the labels. "
                    "Uses ground truth, so corrected "
                    "metrics are optimistic."
                ),
            )

            filter_checkbox = gr.Checkbox(
                label="Apply Wavefront Regularization",
                value=True,
                info=(
                    "Geophysical cycle-skip filter "
                    "using moving-median regularization."
                ),
            )

            process_btn = gr.Button(
                "🚀 Run Inference",
                variant="primary",
                size="lg",
            )

            gr.Markdown(
                """
### ℹ️ About the model

Treats seismic gathers as 2D images to exploit
wavefront continuity across neighboring traces.

**Training data:** Brunswick, Halfmile Lake, Lalor

**Validation:** Unseen Sudbury site

**Architecture:**

- 4 convolutional blocks (16→32→64→128 filters)
- Asymmetric kernels along time & trace axes
- `AdaptiveAvgPool2d` for variable trace counts
- Trace-wise 1D Conv regression head
- L1 (MAE) loss
"""
            )

        # =====================================================
        # Output column
        # =====================================================

        with gr.Column(scale=2):

            gr.Markdown(
                "### 📈 Visualization & Results"
            )

            output_image = gr.Image(
                label="First Break Detection Results",
                type="pil",
            )

            download_picks = gr.File(
                label="📥 Download Picks (ASCII Table)"
            )

            output_text = gr.Markdown()

    # =========================================================
    # Main inference button
    # =========================================================

    process_btn.click(
        fn=process_seismic_data,
        inputs=[
            traces_input,
            labels_input,
            bias_correction,
            filter_checkbox,
        ],
        outputs=[
            output_image,
            output_text,
            download_picks,
        ],
    )

    # =========================================================
    # 1-Click Interactive Examples
    # =========================================================

    sample_traces_0 = (
        "sample_data/image_0_traces.npy"
    )

    sample_labels_0 = (
        "sample_data/image_0_labels.npy"
    )

    sample_traces_1 = (
        "sample_data/image_1_traces.npy"
    )

    sample_labels_1 = (
        "sample_data/image_1_labels.npy"
    )

    example_list = []

    if (
        os.path.exists(sample_traces_0)
        and os.path.exists(sample_labels_0)
    ):
        example_list.append(
            [
                sample_traces_0,
                sample_labels_0,
                False,
                True,
            ]
        )

    if (
        os.path.exists(sample_traces_1)
        and os.path.exists(sample_labels_1)
    ):
        example_list.append(
            [
                sample_traces_1,
                sample_labels_1,
                False,
                True,
            ]
        )

    if os.path.exists(sample_traces_0):
        example_list.append(
            [
                sample_traces_0,
                None,
                False,
                True,
            ]
        )

    if example_list:
        gr.Examples(
            examples=example_list,
            inputs=[
                traces_input,
                labels_input,
                bias_correction,
                filter_checkbox,
            ],
            outputs=[
                output_image,
                output_text,
                download_picks,
            ],
            fn=process_seismic_data,
            cache_examples=False,
            label=(
                "🎯 Click to Test Sample Gathers "
                "(No upload needed)"
            ),
        )

    gr.Markdown(
        """
[GitHub Repository](https://github.com/IsaMaharramov/HolbertonSchoolProject_Team_BHOS_ML1)
"""
    )


# =============================================================
# Local / container entry point
# =============================================================

if __name__ == "__main__":
    demo.launch(show_error=True)