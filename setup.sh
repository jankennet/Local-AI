#!/bin/bash
set -e

echo "Creating Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing base dependencies..."
pip install -r requirements.txt

# Auto-detect NVIDIA vs AMD / Generic GPU
if command -v nvidia-smi &> /dev/null; then
    echo "======================================================="
    echo "NVIDIA GPU Detected! Installing CUDA Acceleration..."
    echo "======================================================="
    
    # Enable CUDA build flag
    export CMAKE_ARGS="-DGGML_CUDA=on"
    pip install llama-cpp-python --no-cache-dir --force-reinstall
else
    echo "======================================================="
    echo "AMD / Generic GPU Detected! Installing Vulkan Acceleration..."
    echo "======================================================="
    
    # Enable Vulkan build flag
    export CMAKE_BUILD_PARALLEL_LEVEL=1
    export CMAKE_ARGS="-DGGML_VULKAN=1"
    pip install llama-cpp-python --no-cache-dir --force-reinstall
fi

echo ""
echo "Setup complete! Run your server with: ./venv/bin/python proxy_server.py"