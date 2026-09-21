"""
Quick test script to verify the demo app dependencies
"""
import sys

print("Testing dependencies...")

# Test PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

# Test NumPy
try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except ImportError:
    print("✗ NumPy not installed")
    sys.exit(1)

# Test Matplotlib
try:
    import matplotlib
    print(f"✓ Matplotlib: {matplotlib.__version__}")
except ImportError:
    print("✗ Matplotlib not installed")
    sys.exit(1)

# Test model loading
try:
    from model import SeismicFirstBreakNet
    model = SeismicFirstBreakNet()
    print(f"✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    sys.exit(1)

# Test Gradio (optional)
try:
    import gradio as gr
    print(f"✓ Gradio: {gr.__version__}")
except ImportError:
    print("⚠ Gradio not installed (run: pip install gradio)")

print("\n✅ Core dependencies OK! Ready for demo.")
