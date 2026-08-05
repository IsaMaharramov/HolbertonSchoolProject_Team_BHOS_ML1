import lzma
import shutil
import glob
import os

def unzip_all(data_folder="data"):
    os.makedirs(data_folder, exist_ok=True)
    
    xz_files = glob.glob(os.path.join(data_folder, '*.xz'))
    
    if not xz_files:
        print(f"No .xz files found in '{data_folder}/'.")
        return

    for xz_file in xz_files:
        out_file = xz_file.replace('.xz', '')
        
        # Skip if the file is already unzipped
        if os.path.exists(out_file):
            print(f"Skipping {os.path.basename(xz_file)} (already extracted)")
            continue
            
        print(f"Extracting {xz_file} -> {out_file}...")
        
        with lzma.open(xz_file, 'rb') as f_in:
            with open(out_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
    print(f"\nExtraction check complete! All raw .hdf5 files are ready inside '{data_folder}'.")

if __name__ == "__main__":
    unzip_all()