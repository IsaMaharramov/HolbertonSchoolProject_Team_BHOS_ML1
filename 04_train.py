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

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=1, shuffle=True)
    # Initialize model, loss, and optimizer
    model = SeismicFirstBreakNet().to(device)
    criterion = nn.L1Loss() # Mean Absolute Error (MAE) in milliseconds
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 15
    print("Starting Training Loop...")
    
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

    # Save trained model weights
    torch.save(model.state_dict(), "first_break_picker.pth")
    print("Model training complete! Weights saved to first_break_picker.pth")

if __name__ == "__main__":
    train_model()