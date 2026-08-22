#!/bin/bash
set -e

echo "======================================================="
echo "        LocalAI Server Setup"
echo "======================================================="

echo ""
echo "[1/5] Creating Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo ""
echo "[2/5] Installing base dependencies..."
pip install --upgrade pip setuptools wheel

# Install everything except llama-cpp-python (built separately below,
# with hardware-specific flags).
grep -v "^llama-cpp-python" requirements.txt > /tmp/requirements-base.txt
pip install -r /tmp/requirements-base.txt

echo ""
echo "[3/5] Detecting GPU backend..."

if command -v nvidia-smi &> /dev/null; then
    echo " NVIDIA GPU detected — building with CUDA"
    BACKEND_CMAKE_ARGS="-DGGML_CUDA=on"
    BACKEND_BUILD_PARALLEL=""
else
    echo " AMD / Intel / Generic GPU detected — building with Vulkan"
    BACKEND_CMAKE_ARGS="-DGGML_VULKAN=on"
    BACKEND_BUILD_PARALLEL="1"
fi

echo ""
echo "[4/5] Building llama-cpp-python (Python bindings — still used for the"
echo "      exact-vocab tokenizer and model catalog/download, just not for"
echo "      serving completions anymore)..."

export CMAKE_ARGS="$BACKEND_CMAKE_ARGS"
if [ -n "$BACKEND_BUILD_PARALLEL" ]; then
    export CMAKE_BUILD_PARALLEL_LEVEL="$BACKEND_BUILD_PARALLEL"
fi

pip install llama-cpp-python \
    --no-cache-dir \
    --force-reinstall \
    --no-binary llama-cpp-python

echo ""
echo "[5/5] Building native llama-server (this is what actually serves"
echo "      completions now — it reads each GGUF's own chat template, which"
echo "      is what makes tool-calling work across different models instead"
echo "      of requiring a hand-picked format per model)..."

if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git
fi

cmake llama.cpp -B llama.cpp/build $BACKEND_CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF
cmake --build llama.cpp/build --config Release --target llama-server \
    -j "${BACKEND_BUILD_PARALLEL:-$(nproc)}"

echo ""
echo "Verifying llama-cpp-python installation..."

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