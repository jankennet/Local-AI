@echo off
echo Creating Python Virtual Environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing base dependencies...
pip install -r requirements.txt

echo Compiling llama-cpp-python with Vulkan Acceleration for RX 560 XT...
set CMAKE_ARGS=-DGGML_VULKAN=1
pip install llama-cpp-python --no-cache-dir --force-reinstall

echo Setup complete! Run your server with: venv\Scripts\python.exe proxy_server.py
pause