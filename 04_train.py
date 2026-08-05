import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset import SeismicDataset
from model import SeismicFirstBreakNet

def train_model():
    # Setup CUDA device (leveraging RTX GPU hardware)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Cross-Asset Validation Setup:
    # Train on Brunswick, Halfmile, and Lalor; Validate on Sudbury to test generalization.
    train_dirs = ['./processed_data/Brunswick', './processed_data/Halfmile', './processed_data/Lalor']
    val_dir = './processed_data/Sudbury'

    # Load datasets (skipping any folder that hasn't finished processing yet)
    train_datasets = [SeismicDataset(d) for d in train_dirs if os.path.exists(d)]
    
    if not train_datasets:
        print("No processed training data found yet! Run 01_process_hdf5.py once downloads finish.")
        return

    # Using batch_size=1 to avoid tensor shape mismatches from variable trace lengths
    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=1, shuffle=True)
    
    # Initialize model
    model = SeismicFirstBreakNet().to(device)
    
    # --- FINE-TUNING SETUP ---
    # Load the baseline weights to avoid losing previous progress
    if os.path.exists("baseline_model.pth"):
        print("Loading baseline weights from baseline_model.pth to fine-tune...")
        model.load_state_dict(torch.load("baseline_model.pth", map_location=device, weights_only=True))
    elif os.path.exists("first_break_picker.pth"):
        print("Loading baseline weights from first_break_picker.pth to fine-tune...")
        model.load_state_dict(torch.load("first_break_picker.pth", map_location=device, weights_only=True))

    criterion = nn.L1Loss() # Mean Absolute Error (MAE) in milliseconds
    
    # Lower learning rate to take microscopic steps (1e-4 instead of 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 20
    print("Starting Fine-Tuning Loop...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(traces)
            
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * traces.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss (MAE msec): {epoch_loss:.4f}")

    # Save fine-tuned model weights separately to protect the baseline
    torch.save(model.state_dict(), "first_break_picker_finetuned.pth")
    print("Model fine-tuning complete! Weights saved to first_break_picker_finetuned.pth")

if __name__ == "__main__":
    train_model()