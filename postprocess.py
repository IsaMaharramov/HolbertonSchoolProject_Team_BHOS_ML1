"""
postprocess.py - Geophysical Spatial Continuity Filter
Eliminates cycle-skipping artifacts and regularizes first break arrival curves.
"""
import numpy as np

def regularize_picks(picks, max_velocity_jump=15.0, median_window=5):
    """
    Suppresses isolated cycle-skip spikes using local median regularization
    and gradient thresholding across adjacent traces.

    Args:
        picks (np.ndarray): 1D array of predicted arrival times.
        max_velocity_jump (float): Maximum realistic time jump between adjacent traces.
        median_window (int): Moving window size (must be odd).

    Returns:
        np.ndarray: Smoothed, physically consistent arrival picks.
    """
    cleaned = np.array(picks, copy=True, dtype=np.float32)
    pad = median_window // 2
    
    # 1. Moving median across trace window
    padded = np.pad(cleaned, pad, mode='edge')
    median_curve = np.array([
        np.median(padded[i:i + median_window]) for i in range(len(cleaned))
    ])
    
    # 2. Identify and substitute unphysical gradient jumps
    diffs = np.abs(np.diff(cleaned, prepend=cleaned[0]))
    spike_mask = diffs > max_velocity_jump
    cleaned[spike_mask] = median_curve[spike_mask]
    
    return cleaned