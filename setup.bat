@echo off
echo Creating Python Virtual Environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing base dependencies...
pip install -r requirements.txt

:: Check if nvidia-smi exists in PATH
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo =======================================================
    echo NVIDIA GPU Detected! Installing CUDA Acceleration...
    echo =======================================================
    set CMAKE_ARGS=-DGGML_CUDA=on
) else (
    echo =======================================================
    echo AMD / Generic GPU Detected! Installing Vulkan Acceleration...
    echo =======================================================
    set CMAKE_BUILD_PARALLEL_LEVEL=1
    set CMAKE_ARGS=-DGGML_VULKAN=1
)

pip install llama-cpp-python --no-cache-dir --force-reinstall

echo.
echo Setup complete! Run your server with: venv\Scripts\python.exe proxy_server.py
pause