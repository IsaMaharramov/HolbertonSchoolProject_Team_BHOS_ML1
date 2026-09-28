import subprocess
import sys
import os

def test_picker_cli_execution(tmp_path):
    output_ascii = tmp_path / "test_picks.txt"
    sample_trace = "sample_data/image_0_traces.npy"

    if not os.path.exists(sample_trace):
        return  # Skip if sample data not placed

    cmd = [
        sys.executable, "picker_cli.py",
        "--input", sample_trace,
        "--export", str(output_ascii),
        "--filter",
        "--cpu"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI exited with error: {result.stderr}"
    assert output_ascii.exists(), "Output ASCII pick table was not written."
    assert output_ascii.stat().st_size > 0, "Output pick file is empty."
