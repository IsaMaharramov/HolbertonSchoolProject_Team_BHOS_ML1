import h5py
import numpy as np
import os

def process_seismic_data(hdf5_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {hdf5_path}...")

    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        data = f['TRACE_DATA/DEFAULT']
        
        # Load the necessary keys and flatten the 1D arrays to prevent indexing errors!
        traces = data['data_array'][:]
        rec_x = data['REC_X'][:].flatten()
        rec_y = data['REC_Y'][:].flatten()
        labels = data['SPARE1'][:].flatten()

    # Combine REC_X and REC_Y to form unique receiver stations
    coords = np.column_stack((rec_x, rec_y))
    unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)

    print(f"Found {len(unique_coords)} unique receiver locations.")

    valid_images = 0
    
    # Reorganize the dataset by separating traces into 2D images based on receiver coordinates
    for i in range(len(unique_coords)):
        trace_indices = np.where(inverse_indices == i)[0]
        
        gather_traces = traces[trace_indices]
        gather_labels = labels[trace_indices]
        
        # Filter out traces where the first break time value is 0 or -1 (unlabeled)
        valid_mask = gather_labels > 0
        
        # Ensure the 2D image has a reasonable number of valid traces before saving
        if np.sum(valid_mask) > 15: 
            clean_traces = gather_traces[valid_mask]
            clean_labels = gather_labels[valid_mask]
            
            # Save as numpy arrays for rapid loading in PyTorch
            np.save(os.path.join(output_dir, f"image_{valid_images}_traces.npy"), clean_traces)
            np.save(os.path.join(output_dir, f"image_{valid_images}_labels.npy"), clean_labels)
            valid_images += 1
            
    print(f"Saved {valid_images} valid 2D seismic images to {output_dir}\n")

if __name__ == "__main__":
    # Map the raw extracted files from the data/ folder using exact filenames
    datasets = {
        "Brunswick": "data/Brunswick_orig_1500ms_V2.hdf5",
        "Halfmile": "data/Halfmile3D_add_geom_sorted.hdf5",
        "Lalor": "data/Lalor_raw_z_1500ms_norp_geom_v3.hdf5",
        "Sudbury": "data/preprocessed_Sudbury3D.hdf"
    }
    
    for name, file_path in datasets.items():
        if os.path.exists(file_path):
            output_dir = os.path.join("./processed_data", name)
            process_seismic_data(file_path, output_dir)
        else:
            print(f"Warning: File not found at '{file_path}'. Check filename or extraction status.")