---
title: Automated Seismic First Break Detection
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.42.0
app_file: app.py
pinned: false
license: mit
---

# Automated Seismic First Break Detection

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/Isa11111/holberton-project)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> 🚀 **Live Interactive Demo:** Test preprocessed gathers directly in the browser on [Hugging Face Spaces](https://huggingface.co/spaces/Isa11111/holberton-project).

A 2D Convolutional Neural Network (CNN) pipeline built in PyTorch to automatically detect seismic first breaks across complex geological assets. 

## Project Overview
First breaks are the initial arrivals of seismic waves. Identifying them is a critical optimization challenge in seismic refraction and tomography studies. This project automates the detection process across four distinct real-world datasets:
*   Brunswick
*   Halfmile
*   Lalor
*   Sudbury

The model is designed to handle varying signal-to-noise ratios and distortions in the first break lines by learning the spatial continuity of the wavefront across neighboring seismic traces.

## Methodology
The pipeline consists of an end-to-end data processing and model training architecture:
1.  **Data Extraction:** Automated decompression of heavily packed `.xz` files into raw HDF5 formats.
2.  **Dataset Reorganization:** Slicing raw 1D trace arrays into organized 2D seismic gathers based on unique receiver coordinates.
3.  **2D CNN Architecture:** A PyTorch-based neural network utilizing Conv2d layers to extract spatial features, standardizing feature height across variable trace lengths via AdaptiveAvgPool2d, and outputting continuous scalar time values.
4.  **Cross-Asset Validation:** The model is trained on the Brunswick, Halfmile, and Lalor assets, and structurally validated against the unseen Sudbury dataset to test pure generalization.

## Download links

The HDF5 files are hosted on AWS, and can be downloaded directly:
 - [Brunswick](https://d3sakqnghgsk6x.cloudfront.net/Brunswick_3D/Brunswick_orig_1500ms_V2.hdf5.xz)
 - [Halfmile Lake](https://d3sakqnghgsk6x.cloudfront.net/Halfmile_3D/Halfmile3D_add_geom_sorted.hdf5.xz)
 - [Lalor](https://d3sakqnghgsk6x.cloudfront.net/Lalor_3D/Lalor_raw_z_1500ms_norp_geom_v3.hdf5.xz)
 - [Sudbury](https://d3sakqnghgsk6x.cloudfront.net/Sudbury_3D/preprocessed_Sudbury3D.hdf.xz)

## How to Run

### Option 0: Live Interactive Web App (Zero Setup)
Launch and test the model immediately on cloud GPU infrastructure:
👉 [Open Hugging Face Space](https://huggingface.co/spaces/Isa11111/holberton-project)

### Option 1: Docker Container (Recommended for Web Demo)
Run the Gradio interface in a clean, memory-optimized Docker container without altering your host environment.

1. **Build the image:**
   ```bash
   docker build -t holberton_project .
   ```
2. **Run the container:**
   ```bash
   docker run -p 7860:7860 --name holberton_project -d holberton_project
   ```
3. **Access the interface:**
   Open your browser to `http://localhost:7860`

### Option 2: Local Python Demo
Launch the Gradio web interface directly on your host machine:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the demo app:
   ```bash
   python demo_app_2.py
   ```

### Option 3: Training Pipeline (Original)
1. Ensure raw datasets are placed in the `/data/` directory.

2. Extract archives: `python 00_unzip.py`

3. Process gathers: `python 01_process_hdf5.py`

4. Train the network: `python 04_train.py`

5. Visualize results: `python 05_visualize.py`

### Option 4: Running Unit Tests
Execute the automated test suite to verify tensor shapes, dataset standardization, and extraction logic:
```bash
python -m pytest -v
```

### Option 5: Multi-Gather Quantitative Benchmark
Generate a cross-gather error report with tolerance acceptance thresholds:
```bash
python 06_benchmark.py
```

## Project Structure
```plaintext
.
├── .github/
│   └── workflows/
│       └── ci.yml                         # Automated GitHub Actions CI test runner
├── sample_data/                           # Sample 2D gathers & ground-truth labels (image_0 to 6)
├── tests/                                 # Pytest automated test suite
│   ├── __init__.py                        # Test package identifier
│   ├── test_cli.py                        # Automated CLI interface & export tests
│   ├── test_dataset.py                    # Tensor shape transposition & standardization tests
│   ├── test_model.py                      # CNN forward pass shape & autograd flow tests
│   ├── test_postprocess.py                # Cycle-skip filter & spatial continuity tests
│   └── test_unzip.py                      # Archive decompression unit test
├── .dockerignore                          # Container exclusion rules for raw data & environments
├── .gitignore                             # Git exclusion rules for heavy archives, caches & data
├── 00_unzip.py                            # Automated .xz archive decompression script
├── 01_process_hdf5.py                     # HDF5 gather grouping by receiver coordinates
├── 04_train.py                            # Fine-tuning loop using AdamW and MAE loss
├── 05_visualize.py                        # Single-gather inference and visual comparison script
├── 06_benchmark.py                        # Multi-gather evaluation & tolerance report (±5/10/15 ms)
├── 07_qc_report.py                        # Multi-panel QC diagnostic generator (R², residuals, error)
├── baseline_model.pth                     # Initial trained baseline weights
├── dataset.py                             # PyTorch Dataset loader with zero-mean standardization
├── demo_app_2.py                          # Interactive Gradio web interface (bias-corrected inference)
├── Dockerfile                             # Production container configuration
├── export_picks.py                        # Industry-standard ASCII pick table exporter
├── FINAL_STEPS.md                         # Milestone delivery and setup checklist
├── first_break_picker_finetuned.pth       # Fine-tuned checkpoint weights
├── first_break_picker.pth                 # Intermediate checkpoint weights
├── launch_demo.bat                        # Windows one-click batch launcher for demo app
├── model.py                               # 2D CNN architecture (SeismicFirstBreakNet)
├── model_profiler.py                      # Latency, parameter complexity, and throughput benchmark
├── picker_cli.py                          # Headless production CLI for automated batch picking
├── postprocess.py                         # Geophysical wavefront continuity & outlier filter
├── presentation_figure.png                # High-resolution visual comparison figure
├── README.md                              # Main project documentation and run guides
├── requirements.txt                       # Project dependencies
└── test_demo.py                           # Quick dependency & model loading verification
```

## Authors:

*Core Model & Training Pipeline: Isa Maharramov*

*AI Wrapper & Demo Interface: Togrul-cmd*