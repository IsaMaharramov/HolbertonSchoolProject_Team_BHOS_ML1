import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SeismicDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all traces files and sort them to ensure matching with labels
        self.trace_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_traces.npy')])
        self.label_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_labels.npy')])
        
        assert len(self.trace_files) == len(self.label_files), "Mismatch between trace and label files!"

    def __len__(self):
        return len(self.trace_files)

    def __getitem__(self, idx):
        # Load the 2D seismic image and 1D labels
        trace_path = os.path.join(self.data_dir, self.trace_files[idx])
        label_path = os.path.join(self.data_dir, self.label_files[idx])
        
        traces = np.load(trace_path).astype(np.float32)
        labels = np.load(label_path).astype(np.float32)
        
        # FLIP THE AXES: Convert from (Traces, Time) to (Time, Traces)
        traces = traces.T 
        
        # Normalize the seismic traces (Standardization: zero mean, unit variance)
        # This helps the neural network handle varying signal-to-noise ratios across assets
        mean = np.mean(traces)
        std = np.std(traces)
        if std != 0:
            traces = (traces - mean) / std
        
        # Convert to PyTorch tensors
        # PyTorch Conv2d expects shape (Channels, Height, Width). We have 1 channel (grayscale basically)
        traces_tensor = torch.from_numpy(traces).unsqueeze(0) 
        labels_tensor = torch.from_numpy(labels)
        
        if self.transform:
            traces_tensor = self.transform(traces_tensor)
            
        return traces_tensor, labels_tensor