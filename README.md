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

## Project Structure
*   `00_unzip.py`: Extracts compressed `.xz` HDF5 data.
*   `01_process_hdf5.py`: Parses HDF5 matrices and cleans valid trace windows into `.npy` tensors.
*   `dataset.py`: Custom PyTorch `Dataset` class for matrix transposition and standardization.
*   `model.py`: Core 2D CNN PyTorch architecture.
*   `04_train.py`: Training loop leveraging CUDA-accelerated hardware with dynamic MAE loss tracking.
*   `05_visualize.py`: Evaluates the model on test data and plots AI predictions directly against manual ground-truth labels using Matplotlib.


## Download links

The HDF5 files are hosted on AWS, and can be downloaded directly:
 - [Brunswick](https://d3sakqnghgsk6x.cloudfront.net/Brunswick_3D/Brunswick_orig_1500ms_V2.hdf5.xz)
 - [Halfmile Lake](https://d3sakqnghgsk6x.cloudfront.net/Halfmile_3D/Halfmile3D_add_geom_sorted.hdf5.xz)
 - [Lalor](https://d3sakqnghgsk6x.cloudfront.net/Lalor_3D/Lalor_raw_z_1500ms_norp_geom_v3.hdf5.xz)
 - [Sudbury](https://d3sakqnghgsk6x.cloudfront.net/Sudbury_3D/preprocessed_Sudbury3D.hdf.xz)


## How to Run

### Option 1: Interactive AI Wrapper Demo (Recommended)
**Added by: Togrul-cmd**

Launch the Gradio web interface for interactive model demonstration:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demo app:**
   ```bash
   python demo_app.py
   ```

3. **Access the interface:**
   - Open your browser to `http://127.0.0.1:7860`
   - Upload your seismic traces (`.npy` file) and ground truth labels (`.npy` file)
   - Click "Run Inference" to see AI predictions overlaid on the seismic data
   - View performance metrics (MAE, RMSE, max error)

**Features:**
- 🎯 Real-time inference with trained model
- 📊 Visual comparison of AI predictions vs manual labels
- 📈 Automatic error metrics calculation
- ⚙️ Optional bias correction toggle
- 🖥️ Clean, professional web interface

### Option 2: Training Pipeline (Original)
**Developed by: Isa Maharramov**

1. Ensure raw datasets are placed in the `/data/` directory.
2. Run data processing: `python 01_process_hdf5.py`
3. Train the network: `python 04_train.py`
4. Visualize results: `python 05_visualize.py`

## Project Structure
```
.
├── demo_app.py                          # Gradio AI wrapper (Togrul-cmd)
├── requirements.txt                     # Python dependencies
├── model.py                             # Neural network architecture
├── dataset.py                           # PyTorch dataset class
├── 00_unzip.py                          # Data extraction
├── 01_process_hdf5.py                   # HDF5 preprocessing
├── 04_train.py                          # Training pipeline
├── 05_visualize.py                      # Visualization script
├── first_break_picker_finetuned.pth     # Trained model weights
└── README.md                            # This file
```

## AI Wrapper Technical Details

The Gradio interface (`demo_app.py`) provides:
- **Model Loading:** Automatically loads the fine-tuned PyTorch model on startup
- **Data Processing:** Handles transpose, normalization, and tensor conversion
- **Inference:** GPU-accelerated prediction with automatic device detection
- **Visualization:** Matplotlib-based overlay of predictions on seismic images
- **Metrics:** MAE, RMSE, and max error computation
- **User Experience:** Clean UI with instructions and model information

**Input Requirements:**
- Seismic traces: NumPy array of shape `(num_traces, time_samples)`
- Labels: NumPy array of shape `(num_traces,)` with first break times

---
*Authors:*  
*Core Model & Training Pipeline: Isa Maharramov*  
*AI Wrapper & Demo Interface: Togrul-cmd*