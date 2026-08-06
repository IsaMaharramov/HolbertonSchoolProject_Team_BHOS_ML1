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
1. Ensure raw datasets are placed in the `/data/` directory.
2. Run data processing: `python 01_process_hdf5.py`
3. Train the network: `python 04_train.py`
4. Visualize results: `python 05_visualize.py`

---
*Author: Isa Maharramov*