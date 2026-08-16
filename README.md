# LocalAI — Local LLM Server

> Lightweight local LLM server for GGUF models using `llama.cpp`, with **CUDA on NVIDIA** and **Vulkan on AMD/Intel**.

Designed for local development, coding assistants, and VS Code tools such as **Continue, Roo Code, and Cline**.

## Features

- OpenAI-compatible API
- Automatic GPU + VRAM detection
- CUDA / Vulkan backend selection
- Interactive GGUF model download
- Full GPU offloading
- Adaptive `n_ctx` / `n_batch`
- Automatic fallback to safer configurations
- Local-only server at `127.0.0.1:8000`

---

## Quick Start

### Linux

```bash
chmod +x setup.sh start.sh
./setup.sh
./start.sh
```

### Windows

```cmd
setup.bat
start.bat
```

The launcher automatically:

```text
Detect GPU
   ↓
Detect VRAM
   ↓
Select model tier
   ↓
Download / select GGUF
   ↓
Find a working runtime configuration
   ↓
Start API server
```

---

## API

```text
http://127.0.0.1:8000/v1
```

Common endpoints:

```text
GET  /v1/models
POST /v1/chat/completions
```

Works with applications supporting OpenAI-compatible APIs.

---

## Recommended Models

| VRAM | Recommended Model | Quantization |
|---:|---|---|
| ≤ 5GB | Qwen 2.5 3B Instruct | Q4_K_M |
| 6–7GB | Qwen 2.5 Coder 7B | Q3_K_M |
| ≥ 8GB | Qwen 2.5 Coder 7B | Q4_K_M |

The launcher provides additional models for each tier.

<details>
<summary><strong>View full model catalog</strong></summary>

### 4GB

- **Qwen 2.5 3B Instruct** — Q4_K_M (~2.0 GB)
- **Llama 3.2 3B Instruct** — Q4_K_M (~2.0 GB)
- **Phi-3.5 Mini 3.8B** — Q4_K_M (~2.3 GB)

### 6GB

- **Qwen 2.5 Coder 7B** — Q3_K_M (~3.8 GB)
- **Qwen 2.5 7B** — Q3_K_M (~3.8 GB)
- **Llama 3.2 3B** — Q8_0 (~3.4 GB)

### 8GB

- **Llama 3.1 8B** — Q4_K_M (~4.9 GB)
- **Qwen 2.5 Coder 7B** — Q4_K_M (~4.7 GB)
- **Qwen 2.5 7B** — Q4_K_M (~4.7 GB)

</details>

---

## Adaptive Runtime

LocalAI automatically tries progressively safer configurations based on detected VRAM.

| Tier | Configurations |
|---|---|
| 4GB | `8K/128 → 4K/128 → 4K/64` |
| 6GB | `12K/256 → 8K/256 → 8K/128 → 4K/128` |
| 8GB | `16K/256 → 12K/256 → 8K/256 → 8K/128 → 4K/128` |

Format:

```text
n_ctx / n_batch
```

If a configuration cannot initialize, the next safer configuration is attempted automatically.

> `n_ctx` controls context/KV-cache capacity. `n_batch` primarily affects prompt-processing memory and performance.

---

## GPU Backends

| GPU | Backend |
|---|---|
| NVIDIA | CUDA |
| AMD | Vulkan |
| Intel | Vulkan |
| CPU fallback | CPU |

The setup scripts automatically select the backend.

<details>
<summary><strong>Linux prerequisites</strong></summary>

### Arch / CachyOS

```bash
sudo pacman -S --needed \
  cmake gcc vulkan-devel \
  spirv-headers spirv-tools \
  shaderc vulkan-icd-loader
```

### Ubuntu / Debian

```bash
sudo apt install \
  build-essential cmake \
  libvulkan-dev glslc \
  spirv-headers spirv-tools \
  vulkan-tools
```

A working GPU driver/Vulkan installation is also required.

</details>

<details>
<summary><strong>Windows prerequisites</strong></summary>

Install:

- Python 3
- CMake
- Visual Studio Build Tools with C++
- Git
- GPU drivers
- CUDA development tools for NVIDIA
- Vulkan runtime/development tools for AMD/Intel

</details>

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

Use the model ID returned by:

```text
GET /v1/models
```

if the client requires an exact identifier.

---

<details>
<summary><strong>Project Structure</strong></summary>

```text
LocalAI/
├── models/
├── setup.sh
├── start.sh
├── setup.bat
├── start.bat
├── proxy_server.py
└── requirements.txt
```

`models/` is created automatically.

</details>

<details>
<summary><strong>Troubleshooting</strong></summary>

### Vulkan build fails

Make sure the Vulkan development packages, SPIR-V headers, and shader compiler are installed.

### Build uses too much RAM

Limit compilation parallelism:

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=2
```

Windows:

```cmd
set CMAKE_BUILD_PARALLEL_LEVEL=2
```

### GPU is not detected

The launcher supports manual GPU/VRAM override.

### Model initialization fails

The launcher automatically tries smaller context/batch configurations. If all configurations fail, check available VRAM, system RAM, drivers, and backend installation.

</details>

---

## Architecture

```text
VS Code / Client
       │
       ▼
 LocalAI Launcher
       │
 ┌─────┴─────┐
 ▼           ▼
GPU/VRAM    Model
Detection   Selection
 └─────┬─────┘
       ▼
   GGUF Model
       │
       ▼
llama-cpp-python
   CUDA/Vulkan
       │
       ▼
Adaptive Runtime
       │
       ▼
OpenAI API :8000/v1
```

## License

Model licenses are determined by their respective Hugging Face repositories.