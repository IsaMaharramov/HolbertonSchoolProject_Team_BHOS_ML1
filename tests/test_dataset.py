import os
import numpy as np
import torch
from dataset import SeismicDataset

def test_seismic_dataset_transformations(tmp_path):
    # tmp_path is a built-in pytest fixture that creates a temporary directory
    data_dir = tmp_path / "mock_data"
    data_dir.mkdir()
    
    # Create mock un-transposed data: (Traces=120, Time=1500)
    mock_traces = np.random.rand(120, 1500).astype(np.float32)
    mock_labels = np.random.rand(120).astype(np.float32)
    
    np.save(data_dir / "image_0_traces.npy", mock_traces)
    np.save(data_dir / "image_0_labels.npy", mock_labels)
    
    dataset = SeismicDataset(data_dir=str(data_dir))
    
    # Check dataset length
    assert len(dataset) == 1
    
    traces_tensor, labels_tensor = dataset[0]
    
    # Verify Transpose: output should be (Channels=1, Time=1500, Traces=120)
    assert traces_tensor.shape == (1, 1500, 120)
    assert labels_tensor.shape == (120,)
    
    # Verify Standardization (zero mean)
    mean_val = traces_tensor.mean().item()
    assert abs(mean_val) < 1e-5, "Traces were not standardized to zero mean."
    