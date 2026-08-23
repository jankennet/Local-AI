# LocalAI — Local LLM Server

> Self-hosted LLM server for GGUF models using `llama.cpp`, with **CUDA on NVIDIA** and **Vulkan on AMD/Intel**. Reachable from every device on your network — laptop, phone, another machine — each with its own persistent conversation.

Designed for local development, coding assistants (VS Code's **Continue**, **Roo Code**, **Cline**), and general chat from any device on your LAN.

## Features

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`, ...)
- Per-device session memory — each client gets its own conversation, tracked server-side
- Automatic token-budget management: sessions are summarized/trimmed before they overflow the model's context window, instead of erroring out
- Sessions auto-expire after a configurable TTL (default 30 days)
- API-key authentication on every route, including the raw OpenAI-compatible ones
- Automatic GPU + VRAM detection, CUDA/Vulkan backend selection
- Interactive GGUF model download
- Full GPU offloading
- Adaptive `n_ctx` / `n_batch` with automatic fallback to safer configurations
- Built-in ReAct agent loop with tool calling (web search, code exec, file ops)
- GPU watchdog for health monitoring
- Bound to `0.0.0.0` — accessible from any device on your network, not just localhost

---

## Quick Start

### Linux

```bash
chmod +x setup.sh start.sh
./setup.sh
export LLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
echo "Save this — every client needs it: $LLM_API_KEY"
./start.sh
```

### Windows

```cmd
setup.bat
set LLM_API_KEY=your-generated-key-here
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
Find or download a GGUF in models/
   ↓
Find a working n_ctx/n_batch configuration
   ↓
Start OpenAI-compatible API + session endpoints, on 0.0.0.0:8000
   ↓
Start background sweep for expired sessions
```

Drop your `.gguf` file in `models/` at the project root before starting (see [Project Structure](#project-structure)) — if none is present, the launcher downloads the top recommendation for your detected VRAM tier automatically.

---

## API

```text
http://<server-lan-ip>:8000/v1
```

Every request — including the raw OpenAI-compatible ones — requires:

```text
X-API-Key: <your LLM_API_KEY>
```

### Raw completions (no memory)

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/agent/chat          # ReAct agent with tools
```

Works with any OpenAI-compatible client. The client is responsible for sending its own conversation history each time — nothing is remembered server-side on this path. Useful for tools (like Continue) that already manage their own context.

Add `"agent": true` to the chat completions request body, or use `/v1/agent/chat` directly, to invoke the ReAct agent with tool access.

### Managed sessions (per-device memory)

```text
POST   /sessions                     -> {"session_id": "..."}
GET    /sessions                     -> list of active sessions
DELETE /sessions/{session_id}
POST   /sessions/{session_id}/chat   -> {"message": "..."} -> {"reply": "..."}
```

Call `POST /sessions` once per device, store the returned `session_id` (see [`docs/CLIENT_SECURITY.md`](docs/CLIENT_SECURITY.md) for how — never in browser storage), then reuse it for every subsequent `POST /sessions/{id}/chat`. The server tracks token usage per session and automatically summarizes/trims older turns before the model's context window would overflow. Sessions untouched for `LLM_SESSION_TTL_DAYS` (default 30) are swept automatically.

---

## Agent Capabilities (Experimental)

The server includes a built-in ReAct-style agent (`app/llm/agent_loop.py`) with these tools:

| Tool | Description |
|---|---|
| `web_search` | DuckDuckGo HTML scrape (no API key) |
| `code_exec` | Sandboxed Python execution |
| `file_read` / `file_write` | Workspace file operations |
| `shell` | Restricted command execution |

Enable by adding `agent: true` to your chat request body, or use the `/v1/agent/chat` endpoint. The agent runs server-side with the same session memory and token budgeting.

---

## Recommended Models

| VRAM | Recommended Model | Quantization |
|---:|---|---|
| ≤ 5GB | Qwen 2.5 3B Instruct | Q4_K_M |
| 6–7GB | Qwen 2.5 7B | Q3_K_M |
| ≥ 8GB | Qwen 2.5 7B | Q4_K_M |

The launcher provides additional models for each tier.

<details>
<summary><strong>View full model catalog</strong></summary>

### 4GB

- **Qwen 2.5 3B Instruct** — Q4_K_M (~2.0 GB)
- **Llama 3.2 3B Instruct** — Q4_K_M (~2.0 GB)
- **Phi-3.5 Mini 3.8B** — Q4_K_M (~2.3 GB)

### 6GB

- **Qwen 2.5 7B** — Q3_K_M (~3.8 GB)
- **Qwen 2.5 7B** — Q3_K_M (~3.8 GB)
- **Llama 3.2 3B** — Q8_0 (~3.4 GB)

### 8GB

- **Llama 3.1 8B** — Q4_K_M (~4.9 GB)
- **Qwen 2.5 7B** — Q4_K_M (~4.7 GB)
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

Format: `n_ctx / n_batch`. If a configuration fails to initialize, the next safer one is tried automatically. Whichever `n_ctx` actually loads becomes the token budget the session manager enforces per device.

> `n_ctx` controls context/KV-cache capacity. `n_batch` primarily affects prompt-processing memory and performance.

---

## GPU Backends

| GPU | Backend |
|---|---|
| NVIDIA | CUDA |
| AMD | Vulkan |
| Intel | Vulkan |
| CPU fallback | CPU |

`setup.sh` / `setup.bat` detect the backend and build `llama-cpp-python` accordingly.

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

### Option A: Continue Extension (Recommended)

Continue manages its own conversation history client-side, so it talks to the raw `/v1/chat/completions` endpoint, not `/sessions`.

1. Install **Continue** from the VS Code Marketplace.
2. Open the Continue sidebar → ⚙️ **Settings** → `~/.continue/config.yaml`.
3. Paste:

```yaml
name: Main Config
version: 1.0.0
schema: v1

models:
  - name: Local Qwen 2.5 Coder
    provider: openai
    model: Qwen2.5-Coder-7B-Instruct-Q4_K_M
    apiBase: http://<server-lan-ip>:8000/v1
    apiKey: dummy
    requestOptions:
      headers:
        X-API-Key: "<your LLM_API_KEY value>"
    defaultCompletionOptions:
      contextLength: 15000   # a bit under your actual n_ctx — leaves room for the reply
      maxTokens: 1024
    roles:
      - chat
      - edit
      - autocomplete
```

- `<server-lan-ip>` — the server machine's LAN IP if connecting from another device, or `127.0.0.1` if VS Code is on the same machine.
- `X-API-Key` is required now — without it every request gets a 401, including from Continue.
- `contextLength` should sit below whichever `n_ctx` your adaptive runtime actually landed on (check the server's startup log), not equal to it.

### Option B: VS Code Native / GitHub Copilot (BYOK)

1. `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) → **`Chat: Manage Language Models`**.
2. **Add Models** → **Custom Endpoint** (or **OpenAI**).
3. Fill in:
   - **Group / Provider Name**: `LocalAI`
   - **API Base URL**: `http://<server-lan-ip>:8000/v1`
   - **API Key**: your `LLM_API_KEY` value (BYOK flows generally only support a bearer-style key field, not a custom header — if the field is sent as `Authorization: Bearer <value>` rather than `X-API-Key`, it won't currently authenticate against this server; Option A is the more reliable path until that's reconciled)
4. Select `LocalAI` from the model dropdown.

Use `GET /v1/models` (with the `X-API-Key` header) if the client needs an exact model identifier.

---

## Project Structure

```text
llm-server/
├── run.py                          # entrypoint: python run.py (or ./start.sh)
├── setup.sh / setup.bat            # first-time install (venv, deps, CUDA/Vulkan build)
├── start.sh / start.bat            # checks LLM_API_KEY, then runs run.py
├── requirements.txt
├── models/                         # your .gguf goes here
├── docs/
│   └── CLIENT_SECURITY.md          # how clients must store api_key/session_id
└── app/
    ├── config.py                   # env vars -> Settings (single source of truth)
    ├── auth.py                     # API key check — dependency + global middleware
    ├── tokenizer.py                # exact token counts via the model's own vocab
    ├── cleanup.py                  # background sweep for expired sessions
    ├── main.py                     # composition root — wires everything together
    │
    ├── models/
    │   └── schemas.py              # request/response Pydantic models
    │
    ├── sessions/
    │   ├── session.py              # Session entity
    │   ├── eviction.py             # Drop / Summarize eviction strategies
    │   ├── repository.py           # persistence interface + JSON implementation
    │   └── store.py                # coordinates sessions, budget, eviction, TTL
    │
    ├── llm/
    │   ├── gpu_detect.py           # hardware detection
    │   ├── catalog.py              # model catalog + download
    │   ├── server_launcher.py      # adaptive n_ctx/n_batch loop -> llama_cpp.server app
    │   ├── completion_client.py    # calls the model (loopback HTTP today)
    │   ├── agent_loop.py           # ReAct-style agent loop with tool calling
    │   ├── tools.py                # built-in tools (web search, code exec, etc.)
    │   └── watchdog.py             # GPU/health monitoring
    │
    └── routes/
        └── sessions_router.py      # HTTP layer for /sessions/*
        └── proxy_router.py         # OpenAI-compatible /v1/* proxy + auth
```

`models/`, `venv/`, and `llama.cpp/` are created automatically if missing.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_API_KEY` | *(required)* | Shared secret; every request needs matching `X-API-Key` header |
| `LLM_HOST` | `0.0.0.0` | Bind address |
| `LLM_PORT` | `8000` | Bind port |
| `LLM_MODELS_DIR` | `models` | Where `.gguf` files are looked for/downloaded to |
| `LLM_SESSIONS_FILE` | `sessions.json` | Persisted session store |
| `LLM_RESERVE_FOR_RESPONSE` | `768` | Tokens always kept free for the model's reply |
| `LLM_SESSION_TTL_DAYS` | `30` | Sessions untouched this long are purged |
| `LLM_CLEANUP_INTERVAL_SECONDS` | `3600` | How often the TTL sweep runs |

---

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

Hardware detection lives in `app/llm/gpu_detect.py` — it falls back to a 4GB CPU-safe tier if nothing is recognized.

### Model initialization fails

The launcher automatically tries smaller context/batch configurations (`app/llm/server_launcher.py`). If all configurations fail, check available VRAM, system RAM, drivers, and backend installation.

### 401 Unauthorized on every request

`LLM_API_KEY` isn't set, or the client isn't sending a matching `X-API-Key` header. The server refuses to start at all without `LLM_API_KEY` set — see `app/config.py`.

### A client's context keeps overflowing / errors out

If it's talking to `/v1/chat/completions` directly (e.g. Continue), it's managing its own history and needs its `contextLength` set below the server's actual `n_ctx` — see the Continue config above. If it's talking to `/sessions/{id}/chat`, the server already manages this automatically; check `app/sessions/eviction.py` if it's still misbehaving.

</details>

---

## Architecture

```text
VS Code / Phone / Laptop / Other Client
                │
      X-API-Key header required
                │
                ▼
      APIKeyMiddleware (app/auth.py)
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
 /v1/*      /sessions/*  /v1/agent/chat
(proxy)    (managed)     (ReAct agent)
        │       │        │
        └───────┼────────┘
                ▼
      llama-cpp-python (CUDA/Vulkan)
                │
                ▼
        Adaptive n_ctx/n_batch runtime
                │
                ▼
            GGUF Model
```

## License

Model licenses are determined by their respective Hugging Face repositories.