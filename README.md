# Automated Seismic First Break Detection

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

## Project Structure
```plaintext
.
├── 00_unzip.py                          # Data extraction
├── 01_process_hdf5.py                   # HDF5 preprocessing
├── 04_train.py                          # Training pipeline
├── 05_visualize.py                      # Visualization script
├── dataset.py                           # PyTorch dataset class
├── demo_app_2.py                        # Gradio AI wrapper (Togrul-cmd)
├── Dockerfile                           # Docker configuration
├── model.py                             # Neural network architecture
├── requirements.txt                     # Python dependencies
├── baseline_model.pth                   # Initial trained weights
├── first_break_picker_finetuned.pth     # Fine-tuned model weights
├── sample_data/                         # Sample 2D gathers and manual labels
├── tests/                               # Pytest automated testing suite
└── README.md                            # This file
```

## Authors:

*Core Model & Training Pipeline: Isa Maharramov*

*AI Wrapper & Demo Interface: Togrul-cmd*