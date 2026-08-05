import torch
import torch.nn as nn

class SeismicFirstBreakNet(nn.Module):
    """
    2D CNN architecture for seismic first break picking.
    Inputs 2D seismic gather image tensor of shape: (Batch, 1, Time_Samples, Traces)
    Outputs continuous predicted first break pick times of shape: (Batch, Traces)
    """
    def __init__(self):
        super(SeismicFirstBreakNet, self).__init__()
        
        # Spatial Feature Extraction: capturing wavefront continuity across neighboring seismic traces
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Reduce time dimension, keep trace resolution intact

            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Trace-wise Time Pick Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((64, None)), # Standardize feature height across variable trace lengths
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=2), # (Batch, Features, Traces)
            nn.Conv1d(64 * 64, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=1)   # Output scalar time value per trace column
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.head(features)
        return output.squeeze(1) # Shape: (Batch, Traces)

if __name__ == "__main__":
    # Test tensor with dummy shape (Batch=2, Channel=1, Time=1500, Traces=120)
    dummy_input = torch.randn(2, 1, 1500, 120)
    model = SeismicFirstBreakNet()
    dummy_output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape (Predicted times per trace): {dummy_output.shape}")