#!/bin/bash
echo "Creating Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing base dependencies..."
pip install -r requirements.txt

echo "Compiling llama-cpp-python with Vulkan Acceleration (Single-Threaded)..."
export CMAKE_BUILD_PARALLEL_LEVEL=1
export CMAKE_ARGS="-DGGML_VULKAN=1"

pip install llama-cpp-python --no-cache-dir --force-reinstall

echo "Setup complete! Run your server with: ./venv/bin/python proxy_server.py"