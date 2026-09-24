import torch
from model import SeismicFirstBreakNet

def test_seismic_net_forward_pass():
    # Setup dummy dimensions: Batch=2, Channel=1, Time=1500, Traces=120
    model = SeismicFirstBreakNet()
    dummy_input = torch.randn(2, 1, 1500, 120)
    
    # Run inference
    output = model(dummy_input)
    
    # Assert output shape matches (Batch, Traces)
    assert output.shape == (2, 120), f"Expected shape (2, 120), got {output.shape}"
    
    # Ensure gradients can flow (checking regression head connectivity)
    assert output.requires_grad == True
