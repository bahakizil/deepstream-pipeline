#!/bin/bash
set -e

echo "Validating DeepStream environment setup..."

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA GPU not detected. This container requires NVIDIA GPU with proper drivers."
    exit 1
fi

# Check NVIDIA drivers
echo "Checking NVIDIA drivers..."
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Check DeepStream installation
echo "Checking DeepStream installation..."
if [ ! -d "/opt/nvidia/deepstream/deepstream" ]; then
    echo "ERROR: DeepStream installation not found."
    exit 1
fi

# Check Python environment
echo "Checking Python environment..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"

# Check for required models
echo "Checking models directory..."
if [ ! -d "/app/models" ]; then
    echo "WARNING: Models directory is empty. You may need to run convert_models_to_onnx.py first."
    mkdir -p /app/models
fi

echo "Environment validation complete." 