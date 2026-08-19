#!/bin/bash
set -e

echo "======================================================="
echo "        LocalAI Server Setup"
echo "======================================================="

echo ""
echo "[1/4] Creating Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo ""
echo "[2/4] Installing base dependencies..."
pip install --upgrade pip setuptools wheel

# Install everything except llama-cpp-python (built separately below,
# with hardware-specific flags).
grep -v "^llama-cpp-python" requirements.txt > /tmp/requirements-base.txt
pip install -r /tmp/requirements-base.txt

echo ""
echo "[3/4] Detecting GPU backend..."

if command -v nvidia-smi &> /dev/null; then
    echo "======================================================="
    echo " NVIDIA GPU Detected"
    echo " Building llama-cpp-python with CUDA..."
    echo "======================================================="

    export CMAKE_ARGS="-DGGML_CUDA=on"

    pip install llama-cpp-python \
        --no-cache-dir \
        --force-reinstall \
        --no-binary llama-cpp-python

else
    echo "======================================================="
    echo " AMD / Intel / Generic GPU Detected"
    echo " Building llama-cpp-python with Vulkan..."
    echo "======================================================="

    export CMAKE_ARGS="-DGGML_VULKAN=on"
    export CMAKE_BUILD_PARALLEL_LEVEL=1

    pip install llama-cpp-python \
        --no-cache-dir \
        --force-reinstall \
        --no-binary llama-cpp-python
fi

echo ""
echo "[4/4] Verifying installation..."

python - <<'PY'
from llama_cpp import llama_supports_gpu_offload

print()
print("llama.cpp GPU offload support:", llama_supports_gpu_offload())
print()
PY

echo "======================================================="
echo " Setup complete!"
echo "======================================================="
echo ""
echo "Before starting, set an API key (required — the server is"
echo "reachable from your whole network):"
echo ""
echo "  export LLM_API_KEY=\"\$(python -c 'import secrets; print(secrets.token_urlsafe(32))')\""
echo "  echo \$LLM_API_KEY   # save this — every client needs it"
echo ""
echo "Then start the server with:"
echo "  ./start.sh"
echo ""