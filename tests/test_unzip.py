import os
import lzma
import importlib
from pathlib import Path

# Dynamically import the script since it starts with numbers
unzip_module = importlib.import_module("00_unzip")

def test_unzip_all_logic(tmp_path):
    data_folder = tmp_path / "data"
    data_folder.mkdir()
    
    # Create a fake .xz file
    xz_file = data_folder / "test_asset.hdf5.xz"
    with lzma.open(xz_file, "wb") as f:
        f.write(b"mock_hdf5_binary_content")
        
    # Run the function pointing to the temporary directory
    unzip_module.unzip_all(data_folder=str(data_folder))
    
    # Check if extracted file exists and matches original content
    extracted_file = data_folder / "test_asset.hdf5"
    assert extracted_file.exists(), "The .hdf5 file was not extracted."
    assert extracted_file.read_bytes() == b"mock_hdf5_binary_content"
