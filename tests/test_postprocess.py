import numpy as np
from postprocess import regularize_picks

def test_regularize_picks_spike_removal():
    # Linear wavefront with an injected cycle-skip spike (+50 ms jump)
    picks = np.linspace(200.0, 300.0, 50).astype(np.float32)
    corrupted = picks.copy()
    spike_idx = 25
    corrupted[spike_idx] += 50.0  # Unrealistic seismic jump

    cleaned = regularize_picks(corrupted, max_velocity_jump=15.0, median_window=5)

    # Verify the spike was suppressed
    assert abs(cleaned[spike_idx] - picks[spike_idx]) < 10.0, "Spike was not suppressed."
    assert cleaned.shape == picks.shape
