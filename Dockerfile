# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV/Matplotlib
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip to prevent wheel resolution bugs
RUN pip install --no-cache-dir --upgrade pip

# LAYER 1: Install PyTorch specifically for CPU
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision

# LAYER 2: Install Gradio and remaining requirements in a fresh memory layer
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Gradio runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "demo_app_2.py"]