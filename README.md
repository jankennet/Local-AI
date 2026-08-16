# LocalAI — Local LLM Server

> A lightweight local AI server for running quantized GGUF models with `llama.cpp` and Vulkan GPU acceleration.

LocalAI provides an **OpenAI-compatible API** for running local LLMs on consumer hardware. It automatically detects the available GPU, maps its VRAM to an appropriate model tier, and provides an interactive model selection and download workflow.

The server is designed primarily for **local development, coding assistants, and VS Code AI extensions** such as Continue, Roo Code, and Cline.

---

## Features

- **OpenAI-compatible API** powered by `llama-cpp-python`
- **Vulkan GPU acceleration** for supported GPUs
- **Automatic GPU detection**
  - NVIDIA via `nvidia-smi`
  - AMD via Windows PowerShell or Linux utilities
  - CPU / integrated graphics fallback
- **Automatic VRAM tier detection**
  - 4GB
  - 6GB
  - 8GB
- **Interactive model selection**
- **Automatic GGUF model downloads** from Hugging Face
- **Full GPU layer offloading** using `n_gpu_layers=-1`
- **Local-only server** bound to `127.0.0.1:8000`
- Works with applications that support OpenAI-compatible endpoints

---

## How It Works

The launcher follows this flow:

```text
Start LocalAI
     │
     ▼
Detect GPU + VRAM
     │
     ▼
Map VRAM to model tier
     │
     ├── 4GB
     ├── 6GB
     └── 8GB
     │
     ▼
Check models/
     │
     ├── Existing GGUF
     │       └── Run / Download New
     │
     └── No GGUF
             └── Select & Download Model
                     │
                     ▼
              Hugging Face
                     │
                     ▼
              llama-cpp-python
                     │
                     ▼
          OpenAI-compatible API
             127.0.0.1:8000
```

---

# Requirements

## Hardware

The model catalog is organized around GPU VRAM capacity.

| VRAM | Target Tier | Typical Use |
|---:|---|---|
| ≤ 5GB | 4GB | Lightweight 3B models |
| 6–7GB | 6GB | 7B Q3 models |
| ≥ 8GB | 8GB | 7B Q4 / 8B Q4 models |

The VRAM tier is used to determine which models are presented in the interactive selector.

> **Note:** The detected VRAM is only used for model selection. Actual model compatibility also depends on available system RAM, GPU driver support, context size, and model quantization.

---

# Software Prerequisites

Before installing `llama-cpp-python`, install the required C/C++ build tools and Vulkan development packages.

## Arch Linux / CachyOS / Manjaro

```bash
sudo pacman -S --needed \
  cmake \
  gcc \
  vulkan-devel \
  spirv-headers \
  spirv-tools \
  shaderc \
  vulkan-icd-loader
```

## Ubuntu / Debian

```bash
sudo apt update

sudo apt install \
  build-essential \
  cmake \
  libvulkan-dev \
  glslc \
  spirv-headers \
  spirv-tools \
  vulkan-tools
```

## Fedora

```bash
sudo dnf install \
  cmake \
  gcc-c++ \
  vulkan-devel \
  spirv-headers \
  spirv-tools \
  shaderc
```

You also need a working Vulkan driver for your GPU.

---

# Project Structure

The expected project structure is:

```text
LocalAI/
├── models/             # Downloaded .gguf models
├── setup.sh            # Linux setup script
├── start.sh            # Linux launcher
├── setup.bat           # Windows setup script
├── start.bat           # Windows launcher
└── proxy_server.py     # LocalAI launcher
```

The Python launcher creates the `models/` directory automatically if it does not exist.

---

# Installation

## Linux / CachyOS

Make the scripts executable:

```bash
chmod +x setup.sh start.sh
```

Run the setup:

```bash
./setup.sh
```

## Windows

Run:

```cmd
setup.bat
```

---

# Vulkan Build

Vulkan support must be enabled when `llama-cpp-python` is compiled.

The important build option is:

```bash
-DGGML_VULKAN=1
```

For older CPUs, limit the number of parallel compilation jobs to avoid excessive RAM usage:

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=2
export CMAKE_ARGS="-DGGML_VULKAN=1"

pip install llama-cpp-python --no-cache-dir --force-reinstall
```

### Why limit compilation threads?

Compiling `llama-cpp-python` can consume significant CPU and memory bandwidth. On older quad-core systems, unrestricted parallel compilation can cause heavy system thrashing or make the machine temporarily unresponsive.

Two parallel build jobs are a safer starting point for older hardware.

---

# Model Selection

When the server starts, it detects the GPU and determines its VRAM tier.

For example:

```text
=============================================================
                 GPU & VRAM DETECTED
=============================================================
  • Vendor        : AMD
  • Model         : AMD Radeon RX ...
  • VRAM Detected : ~8 GB (Target Tier: 8GB)
=============================================================
```

The detected hardware can be confirmed or manually overridden.

If the detected configuration is incorrect, the launcher allows you to select:

```text
NVIDIA
AMD
Intel / Integrated / CPU
```

and manually choose:

```text
4GB
6GB
8GB
```

---

# Included Models

## 4GB Tier

### Qwen 2.5 3B Instruct

```text
Qwen/Qwen2.5-3B-Instruct-GGUF
qwen2.5-3b-instruct-q4_k_m.gguf
~2.0 GB
```

Best general-purpose lightweight option.

### Llama 3.2 3B Instruct

```text
bartowski/Llama-3.2-3B-Instruct-GGUF
Llama-3.2-3B-Instruct-Q4_K_M.gguf
~2.0 GB
```

Lightweight Meta model.

### Phi-3.5 Mini 3.8B

```text
bartowski/Phi-3.5-mini-instruct-GGUF
Phi-3.5-mini-instruct-Q4_K_M.gguf
~2.3 GB
```

Lightweight model focused on reasoning and general tasks.

---

## 6GB Tier

### Qwen 2.5 Coder 7B — Q3_K_M

```text
Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
qwen2.5-coder-7b-instruct-q3_k_m.gguf
~3.8 GB
```

Recommended for coding.

### Qwen 2.5 7B — Q3_K_M

```text
Qwen/Qwen2.5-7B-Instruct-GGUF
qwen2.5-7b-instruct-q3_k_m.gguf
~3.8 GB
```

General chat and reasoning.

### Llama 3.2 3B — Q8_0

```text
bartowski/Llama-3.2-3B-Instruct-GGUF
Llama-3.2-3B-Instruct-Q8_0.gguf
~3.4 GB
```

Higher-quality 3B quantization.

---

## 8GB Tier

### Llama 3.1 8B — Q4_K_M

```text
bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
~4.9 GB
```

General-purpose 8B model.

### Qwen 2.5 Coder 7B — Q4_K_M

```text
Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
qwen2.5-coder-7b-instruct-q4_k_m.gguf
~4.7 GB
```

**Recommended for coding and VS Code.**

### Qwen 2.5 7B — Q4_K_M

```text
Qwen/Qwen2.5-7B-Instruct-GGUF
qwen2.5-7b-instruct-q4_k_m.gguf
~4.7 GB
```

General chat and mathematics.

---

# Running LocalAI

Start the launcher:

```bash
./start.sh
```

The launcher will:

1. Detect the GPU.
2. Detect available VRAM.
3. Determine the appropriate model tier.
4. Ask you to confirm the hardware.
5. Check the `models/` directory.
6. Download a model if necessary.
7. Start the local API server.

You can also launch the Python file directly:

```bash
./venv/bin/python proxy_server.py
```

---

# Server Configuration

The server currently runs on:

```text
Host: 127.0.0.1
Port: 8000
```

API base:

```text
http://127.0.0.1:8000/v1
```

The launcher configures the model with:

```text
GPU layers: all
Context size: 4096
Batch size: 512
```

`n_gpu_layers=-1` requests full GPU layer offloading.

---

# OpenAI-Compatible API

The server is provided by `llama-cpp-python`'s built-in server.

Typical endpoints include:

```text
GET  /v1/models
POST /v1/chat/completions
```

This means applications that support OpenAI-compatible APIs can connect to LocalAI without requiring a separate custom API implementation.

---

# VS Code Integration

LocalAI can be used as a local inference backend for coding assistants.

## Continue

Install the **Continue** VS Code extension and configure it to use the local OpenAI-compatible endpoint.

Example:

```yaml
name: Main Config
version: 1.0.0
schema: v1

models:
  - name: Local Qwen 2.5 Coder
    provider: openai
    model: qwen2.5-coder-7b-instruct-q4_k_m
    apiBase: http://127.0.0.1:8000/v1
    roles:
      - chat
      - edit
      - autocomplete
```

> Use the actual model identifier exposed by `/v1/models` if your Continue configuration requires an exact model name.

Other OpenAI-compatible VS Code clients can use the same API base:

```text
http://127.0.0.1:8000/v1
```

---

# Troubleshooting

## `Could not find SPIRV-HeadersConfig.cmake`

The Vulkan build dependencies are missing.

Install:

```text
spirv-headers
spirv-tools
shaderc
```

Then reinstall `llama-cpp-python`:

```bash
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

---

## Compilation Uses Too Much RAM

Reduce the number of build jobs:

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=2
```

Then rebuild:

```bash
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

---

## Vulkan GPU Is Not Being Used

First verify that your GPU has a working Vulkan installation.

Then verify that `llama-cpp-python` was compiled with:

```bash
-DGGML_VULKAN=1
```

The Python launcher itself does not compile or enable Vulkan; it relies on the installed `llama-cpp-python` build.

---

## Wrong GPU or VRAM Detected

The launcher provides a manual override during startup.

Select:

```text
NVIDIA
AMD
Intel / Integrated / CPU
```

and then choose the appropriate:

```text
4GB
6GB
8GB
```

tier.

---

## No Models Found

The launcher automatically creates:

```text
models/
```

If no `.gguf` model exists, it opens the model selector and downloads the selected model from Hugging Face.

You can also manually place compatible `.gguf` files inside:

```text
models/
```

---

# Recommended Configuration for Older Hardware

For systems with older CPUs and limited RAM, start with:

```text
CMAKE_BUILD_PARALLEL_LEVEL=2
```

and use a model appropriate for the GPU's VRAM tier.

For an 8GB GPU, the recommended coding model is:

```text
Qwen 2.5 Coder 7B — Q4_K_M
```

For lower-VRAM systems, use the 3B models in the 4GB tier.

---

# Architecture

```text
                 ┌─────────────────────┐
                 │      VS Code        │
                 │ Continue / Roo /    │
                 │       Cline         │
                 └──────────┬──────────┘
                            │
                            │ OpenAI API
                            ▼
                 ┌─────────────────────┐
                 │      LocalAI        │
                 │   Python Launcher   │
                 └──────────┬──────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
        GPU / VRAM Detection      Model Selection
                │                       │
                └───────────┬───────────┘
                            ▼
                 ┌─────────────────────┐
                 │   GGUF Model        │
                 │      models/        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  llama-cpp-python   │
                 │ Vulkan GPU Backend   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ OpenAI-Compatible   │
                 │ API :8000/v1        │
                 └─────────────────────┘
```

---

# License

Add the project's license information here.

Individual models downloaded from Hugging Face are subject to their respective model licenses.