"""
export_picks.py - Industry Standard ASCII Pick Table Exporter
Exports predicted first break picks to industry-compliant whitespace-delimited tables.
"""
import os
import numpy as np

def export_picks_to_ascii(trace_ids, picks, output_path="predicted_picks.txt"):
    """
    Writes first break picks to a formatted seismic table.
    
    Args:
        trace_ids (iterable): Trace numbers or receiver channel IDs.
        picks (iterable): Continuous arrival times in milliseconds.
        output_path (str): Destination text file path.
    """
    with open(output_path, "w") as f:
        f.write("# SEISMIC FIRST BREAK AUTOMATED ARRIVAL PICKS\n")
        f.write("# Format: Trace_ID    Pick_Time_ms\n")
        f.write("# ----------------------------------------\n")
        for tid, tval in zip(trace_ids, picks):
            f.write(f"{int(tid):<12}\t{float(tval):<15.3f}\n")
    print(f"Picks successfully exported to: {output_path}")

if __name__ == "__main__":
    sample_label = "sample_data/image_0_labels.npy"
    if os.path.exists(sample_label):
        data = np.load(sample_label)
        export_picks_to_ascii(range(1, len(data) + 1), data)
