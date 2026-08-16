# LocalAI Proxy Server (llama.cpp + Vulkan)

> **Hardware Profile:**
> This specific build and guide is highly optimized for legacy and budget hardware configurations, such as:
> - **CPU:** AMD Athlon X4 860K (Non-AVX2 / 4-core)
> - **RAM:** 16GB 1866MHz DDR3
> - **GPU:** AMD Radeon RX 560 XT (8GB VRAM)

An OpenAI-compatible local API server built with Python, FastAPI, and `llama-cpp-python` with **Vulkan GPU Acceleration**. 

Designed to run lightweight or quantised GGUF models locally on AMD, Intel, or NVIDIA GPUs and integrate seamlessly with VS Code AI extensions like **Continue.dev**, **Roo Code**, or **Cline**.

---

## Features

- **OpenAI Compatible Endpoint**: Exposes standard `/v1/chat/completions` and `/v1/models` API routes.
- **Vulkan Acceleration**: Uses Vulkan compute shaders for high-performance inference.
- **Hardware Optimized**: Tuned to prevent CPU thread thrashing and memory bandwidth bottlenecks during compilation on older processors.
- **Interactive CLI Launcher**: Select and launch models dynamically on startup using a terminal UI.

---

## Requirements & Prerequisites

Before running the setup script, **you must install the required C++ build tools and Vulkan SPIR-V libraries on your operating system**. Without these, `llama-cpp-python` will fail to compile its GPU backend.

### 1. System Package Dependencies

#### Arch Linux / CachyOS / Manjaro
```bash
sudo pacman -S --needed cmake gcc vulkan-devel spirv-headers spirv-tools shaderc vulkan-icd-loader
```

#### Ubuntu / Debian
```bash
sudo apt update
sudo apt install build-essential cmake libvulkan-dev glslc spirv-headers spirv-tools vulkan-tools
```

#### Fedora
```bash
sudo dnf install cmake gcc-c++ vulkan-devel spirv-headers spirv-tools shaderc
```

---

## Setup & Installation

### Step 1: Clone or Place Files
Ensure your project has the following directory layout:

```text
LocalAI/
├── models/             # Place your downloaded .gguf files here
├── setup.sh            # Setup script for Linux
├── start.sh            # Launch script for Linux
├── setup.bat           # Setup script for Windows
├── start.bat           # Launch script for Windows
└── proxy_server.py     # FastAPI Server app
```

### Step 2: Run Setup
Run the setup script for your respective OS:

**Linux / CachyOS:**
```bash
chmod +x setup.sh start.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

> **Note on Compilation Time:** Building `llama-cpp-python` with Vulkan support compiles C++ source code and GPU shaders specifically for your hardware. On an older quad-core CPU with DDR3 RAM, this process may take **5 to 15 minutes**. 

---

## ⚠️ Troubleshooting Common Installation Errors

### Error 1: `Could not find SPIRV-HeadersConfig.cmake`
* **Cause**: The Vulkan backend requires SPIR-V headers to build GPU shaders.
* **Fix**: Install `spirv-headers` and `spirv-tools` via your package manager (see prerequisites above), then force a clean rebuild:
  ```bash
  pip install llama-cpp-python --no-cache-dir --force-reinstall
  ```

### Error 2: Compilation Hangs or Freezes Your PC
* **Cause**: CMake defaults to running as many compiler threads as CPU cores. On older 4-core CPUs with DDR3 memory, this causes severe RAM bandwidth thrashing.
* **Fix**: Limit CMake to 2 parallel build threads during installation. Update your `setup.sh` with these lines before the pip install command:
  ```bash
  export CMAKE_BUILD_PARALLEL_LEVEL=2
  export CMAKE_ARGS="-DGGML_VULKAN=1"
  pip install llama-cpp-python --no-cache-dir --force-reinstall
  ```

---

## How to Run the Server

1. Start the server:
   ```bash
   ./start.sh
   ```
   or run directly:
   ```bash
   ./venv/bin/python proxy_server.py
   ```
2. Select or Download Model:
    - If no .gguf file exists in models/, the terminal will display an interactive menu powered by questionary.
    - Use Up / Down Arrow keys and Enter to choose a model (Qwen 2.5 Coder 7B, Llama 3.1 8B, Phi-3.5 Mini, etc.).
    - The script automatically downloads the model from Hugging Face and saves it to models/.

3. The server will successfully launch at `http://127.0.0.1:8000`.
4. Connect to VS Code:

---

## Integrating with VS Code

You can connect this local server to VS Code using either the **Continue extension** or **VS Code Native / GitHub Copilot (BYOK)**.

### Option A: Continue Extension (Recommended)
`Continue` provides dedicated support for local models, offline chat, and instant inline tab autocomplete. 

1. Install **Continue** from the VS Code Marketplace.
2. Open the Continue sidebar, click the ⚙️ **Settings icon** (bottom right), and open `~/.continue/config.yaml`.
3. Paste the following configuration to enable the model for Chat, Edits, and Autocomplete:

```yaml
name: Main Config
version: 1.0.0
schema: v1

models:
  - name: Local Qwen 2.5 Coder
    provider: openai
    model: Qwen2.5-Coder-7B-Instruct-Q4_K_M
    apiBase: [http://127.0.0.1:8000/v1](http://127.0.0.1:8000/v1)
    roles:
      - chat
      - edit
      - autocomplete
```

### Option B: VS Code Native / GitHub Copilot (BYOK)
VS Code allows connecting custom OpenAI-compatible endpoints directly through its native Chat interface via **Bring Your Own Key (BYOK)**.

1. Press `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) to open the Command Palette.
2. Run: **`Chat: Manage Language Models`**.
3. Click **Add Models** $\rightarrow$ select **Custom Endpoint** (or **OpenAI**).
4. Fill in the details when prompted:
   - **Group / Provider Name**: `LocalAI`
   - **API Base URL**: `http://127.0.0.1:8000/v1`
   - **API Key**: Type `none` or `not-needed`
5. Select `LocalAI` from the model dropdown in your VS Code Chat panel.