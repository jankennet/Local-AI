@echo off
setlocal EnableExtensions

echo =======================================================
echo          LocalAI Server Setup
echo =======================================================
echo.

echo [1/4] Creating Python Virtual Environment...
if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate.bat

echo.
echo [2/4] Upgrading Python build tools...
python -m pip install --upgrade pip setuptools wheel

echo.
echo [3/4] Installing base dependencies...

if exist requirements.txt (
    python -m pip install -r requirements.txt
)

echo.
echo =======================================================
echo Detecting GPU backend...
echo =======================================================

where nvidia-smi >nul 2>&1

if %ERRORLEVEL% EQU 0 (
    echo.
    echo NVIDIA GPU Detected!
    echo Building llama-cpp-python with CUDA...
    echo.

    set "CMAKE_ARGS=-DGGML_CUDA=on"

    python -m pip install llama-cpp-python ^
        --no-cache-dir ^
        --force-reinstall ^
        --no-binary llama-cpp-python

) else (
    echo.
    echo AMD / Intel / Generic GPU Detected!
    echo Building llama-cpp-python with Vulkan...
    echo.

    set "CMAKE_ARGS=-DGGML_VULKAN=on"
    set "CMAKE_BUILD_PARALLEL_LEVEL=1"

    python -m pip install llama-cpp-python ^
        --no-cache-dir ^
        --force-reinstall ^
        --no-binary llama-cpp-python
)

echo.
echo [4/4] Verifying installation...
echo.

python -c "from llama_cpp import llama_supports_gpu_offload; print('GPU offload support:', llama_supports_gpu_offload())"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =======================================================
    echo Installation verification FAILED.
    echo =======================================================
    echo.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo Setup complete!
echo =======================================================
echo.
echo Before starting, set an API key ^(required — the server is
echo reachable from your whole network^):
echo.
echo     set LLM_API_KEY=your-generated-key-here
echo.
echo Then start the server with:
echo.
echo     start.bat
echo.
echo =======================================================

pause
endlocal