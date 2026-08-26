# LocalAI — Local LLM Server

> Self-hosted LLM server for GGUF models using `llama.cpp`, with **CUDA on NVIDIA** and **Vulkan on AMD/Intel**. Reachable from every device on your network — laptop, phone, another machine — each with its own persistent conversation.

Designed for local development, coding assistants (VS Code's **Continue**, **Roo Code**, **Cline**), and general chat from any device on your LAN.

## Features

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`, ...)
- Per-device session memory — each client gets its own conversation, tracked server-side
- Automatic token-budget management: sessions are summarized/trimmed before they overflow the model's context window
- Sessions auto-expire after configurable TTL (default 30 days)
- API-key authentication on every route
- Automatic GPU + VRAM detection, CUDA/Vulkan backend selection
- Interactive GGUF model download with adaptive `n_ctx` / `n_batch` fallback
- Built-in ReAct agent loop with tool calling (web search, code exec, file ops)
- GPU watchdog for health monitoring
- **Semantic conversation RAG** — vector store (Qdrant or in-memory) retrieves relevant past turns
- **Configurable embedding model** — select during setup
- **Debug endpoints** — inspect vector store, search, add test vectors
- Bound to `0.0.0.0` — accessible from any device on your network

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
1. Detects GPU + VRAM
2. Prompts for embedding model (for RAG)
3. Downloads recommended GGUF for your VRAM tier
4. Finds a working `n_ctx`/`n_batch` configuration
5. Starts API on `0.0.0.0:8000` with background session cleanup

Drop your `.gguf` in `models/` before starting — if missing, the launcher downloads one automatically.

---

## API

```
http://<server-lan-ip>:8000/v1
```

Every request requires:
```
X-API-Key: <your LLM_API_KEY>
```

### Raw completions (no server memory)
```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/agent/chat          # ReAct agent with tools
```

Works with any OpenAI-compatible client. Client sends full history each time — nothing remembered server-side.

Add `"agent": true` to request body, or use `/v1/agent/chat` directly.

### Managed sessions (per-device memory)
```text
POST   /sessions                     -> {"session_id": "..."}
GET    /sessions                     -> list of active sessions
DELETE /sessions/{session_id}
POST   /sessions/{session_id}/chat   -> {"message": "..."} -> {"reply": "..."}
```

Call `POST /sessions` once per device, store `session_id`, then reuse for every `POST /sessions/{id}/chat`. Server tracks tokens, summarizes/trims before overflow. Sessions untouched 30 days are purged.

### Debug endpoints (vector store inspection)
Enable with `LLM_ENABLE_DEBUG=true`:
```text
GET    /debug/vector/stats           -> stats (backend, count, model, dim)
POST   /debug/vector/add             -> add test vector with metadata
POST   /debug/vector/search          -> semantic search (optional session_id filter)
DELETE /debug/vector/clear           -> clear all vectors
GET    /debug/sessions/{id}/vectors  -> inspect session's vector state
```

### Monitoring endpoints
```text
GET    /health    -> server status, llama-server health, model info, active sessions
GET    /stats     -> human-readable summary (active sessions, tokens, tool calls, HTTP stats)
GET    /metrics   -> Prometheus scrape endpoint (raw metrics for Grafana/Prometheus)
```

---

## Semantic Context Retrieval (RAG)

Managed sessions (`/sessions/{id}/chat`) use a local embedding model to retrieve relevant past turns instead of stuffing everything into context:

1. Query embedded (combined with recent context + summary)
2. **Two-stage retrieval**: Broad vector search (default 20) → Cross-encoder rerank (default top 5)
3. Retrieved context + recent turns + summary sent to model

**Vector store backends:**
- **Qdrant (default)** — persistent, disk-based (`./qdrant_db/`), metadata filtering, scales to millions
- **Simple (in-memory)** — no deps, lost on restart, good for testing

Switch via `LLM_VECTOR_BACKEND=qdrant|simple`.

**RAG Configuration (env vars):**
| Variable | Default | Description |
|---|---|---|
| `LLM_RAG_TOP_K` | 5 | Final results after reranking |
| `LLM_RAG_INITIAL_K` | 20 | Initial vector search breadth (broader = better recall) |
| `LLM_RAG_TOKEN_BUDGET` | 1024 | Max tokens for retrieved context (prevents crowding recent history) |
| `LLM_RERANKER_ENABLED` | true | Enable cross-encoder reranking |
| `LLM_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model (CPU-friendly, 22M params) |

**Dynamic Response Reserve (env vars):**
| Variable | Default | Description |
|---|---|---|
| `LLM_RESERVE_FOR_RESPONSE_MIN` | 256 | Minimum tokens reserved for short/simple queries |
| `LLM_RESERVE_FOR_RESPONSE_MAX` | 2048 | Maximum tokens reserved for complex/code queries |

The server estimates response length from query complexity (length, code keywords, multi-part questions) and adjusts `max_tokens` per request. Simple Q&A gets ~256 tokens; complex coding tasks get ~1000-2000.

**CPU-friendly reranker options** (no AVX2 needed):
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M, fast, default)
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (33M, better quality)

> This is **conversation history RAG** (recalls what you discussed). It is **not** Continue's `@codebase` — that indexes your **repository files** separately. They work together: `@codebase` = relevant code files; server RAG = relevant past conversation turns.

---

## Recommended Models

| VRAM | Recommended Model | Quantization |
|---:|---|---|
| ≤ 5GB | Qwen 2.5 3B Instruct | Q4_K_M |
| 6–7GB | Qwen 2.5 7B | Q3_K_M |
| 8–9GB | Qwen 2.5 7B | Q4_K_M |
| 10–11GB | Qwen 2.5 7B / Qwen 2.5 14B | Q6_K / Q3_K_M |
| 12–13GB | Qwen 2.5 14B | Q4_K_M |
| 14–17GB | Qwen 2.5 14B / Qwen 2.5 32B | Q6_K / Q3_K_M |
| 18–21GB | Qwen 2.5 32B / Llama 3.1 70B | Q4_K_M / Q3_K_M |
| 22–27GB | Qwen 2.5 32B / Llama 3.1 70B | Q6_K / Q4_K_M |
| 28–36GB | Llama 3.1 70B / Qwen 2.5 32B | Q6_K / Q8_0 |
| 37–44GB | Llama 3.1 70B | Q8_0 |
| ≥ 45GB | Llama 3.1 70B | Q8_0 (full KV cache headroom) |

The launcher provides additional models for each tier.

---

## Adaptive Runtime

Automatically tries safer configurations based on detected VRAM:

| Tier | Configurations |
|---|---|
| 4GB | `8K/128 → 4K/128 → 4K/64` |
| 6GB | `12K/256 → 8K/256 → 8K/128 → 4K/128` |
| 8GB | `16K/256 → 12K/256 → 8K/256 → 8K/128 → 4K/128` |
| 10GB | `20K/256 → 16K/256 → 12K/256 → 8K/256 → 8K/128` |
| 12GB | `24K/256 → 20K/256 → 16K/256 → 12K/256 → 8K/256` |
| 16GB | `32K/512 → 24K/512 → 20K/512 → 16K/512 → 12K/256` |
| 20GB | `32K/512 → 24K/512 → 20K/512 → 16K/512` |
| 24GB+ | `32K/512` (max context) |

Format: `n_ctx / n_batch`. Failed configs fall back automatically. The `n_ctx` that loads becomes the token budget.

---

## GPU Backends

| GPU | Backend |
|---|---|
| NVIDIA | CUDA |
| AMD | Vulkan |
| Intel | Vulkan |
| CPU fallback | CPU |

`setup.sh` / `setup.bat` detect and build accordingly.

---

## Tool Calling (ReAct Agent)

Managed sessions (`/sessions/{id}/chat`) include a ReAct agent loop with
built-in tools (`read_file`, `write_file`, `list_dir`, `run_bash`).

Reliability improvements:
- **Parallel tool calls** — multiple tool invocations in one round execute concurrently
- **Schema validation** — arguments validated against OpenAI function schemas before execution
- **Automatic retries** — failed tools retry up to `LLM_TOOL_MAX_RETRIES` times (default 2)
- **Configurable timeouts** — per-tool timeout via `LLM_TOOL_TIMEOUT_SECONDS` (default 30s)
- **Structured logging** — every tool call logs duration, args, and success/failure

Environment variables:
| Variable | Default | Description |
|---|---:|---|
| `LLM_TOOL_TIMEOUT_SECONDS` | 30.0 | Max seconds per tool call |
| `LLM_TOOL_MAX_RETRIES` | 2 | Retry attempts for failed tools |
| `LLM_ALLOW_SHELL` | 0 | Set to `1` to enable `run_bash` tool |
| `LLM_WORKSPACE_DIR` | `workspace` | Root directory for file tools |

---

## Integrating with VS Code (Continue)

Continue manages its own history client-side, so it uses the raw `/v1/chat/completions` endpoint.

1. Install **Continue** from VS Code Marketplace
2. Open Continue sidebar → ⚙️ Settings → `~/.continue/config.yaml`
3. Paste:

```yaml
name: Main Config
version: 1.0.0
schema: v1

models:
  - name: local-qwen-coder
    provider: openai
    model: qwen2.5-7b-instruct-q4_k_m
    apiBase: http://127.0.0.1:8000/v1
    apiKey: dummy
    requestOptions:
      headers:
        X-API-Key: "<your LLM_API_KEY value>"
    roles: [chat, edit]
    capabilities:
      - tool_use
    defaultCompletionOptions:
      contextLength: 15000
      maxTokens: 1024

  - name: CodeBERT Embeddings
    provider: openai
    model: "microsoft/codebert-base"
    apiBase: "http://127.0.0.1:8000/v1"
    apiKey: "dummy"
    roles:
      - embed

context:
  - provider: codebase
```

- `<server-lan-ip>` — server machine's LAN IP (or `127.0.0.1` if local)
- `X-API-Key` required — without it every request gets 401
- `contextLength` should be below actual `n_ctx` (check server startup log)

---

## Project Structure

```
llm-server/
├── run.py                          # entrypoint
├── setup.sh / setup.bat            # first-time install (venv, deps, CUDA/Vulkan, embedding model)
├── start.sh / start.bat            # checks LLM_API_KEY, runs run.py
├── requirements.txt
├── models/                         # your .gguf goes here
├── qdrant_db/                      # vector store data (runtime, in .gitignore)
├── .env                            # created by setup, holds embedding model choice
├── docs/
│   └── CLIENT_SECURITY.md          # how clients store api_key/session_id
└── app/
    ├── config.py                   # env vars -> Settings (loads .env)
    ├── auth.py                     # API key middleware
    ├── tokenizer.py                # exact token counts via model vocab
    ├── cleanup.py                  # background TTL sweep
    ├── main.py                     # composition root
    ├── models/
    │   └── schemas.py              # Pydantic request/response models
    ├── sessions/
    │   ├── session.py              # Session entity
    │   ├── eviction.py             # Drop / Summarize strategies
    │   ├── repository.py           # persistence interface + JSON impl
    │   └── store.py                # coordinates sessions, budget, eviction, TTL
    ├── llm/
    │   ├── gpu_detect.py           # hardware detection
    │   ├── catalog.py              # model catalog + download
    │   ├── server_launcher.py      # adaptive n_ctx/n_batch -> llama_cpp.server
    │   ├── completion_client.py    # calls model (loopback HTTP)
    │   ├── agent_loop.py           # ReAct agent with tools
    │   ├── tools.py                # built-in tools
    │   └── watchdog.py             # GPU/health monitoring
    └── routes/
        ├── sessions_router.py      # /sessions/*
        ├── proxy_router.py         # /v1/* proxy + auth
        └── debug_router.py         # /debug/vector/*
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Vulkan build fails | Install Vulkan dev packages, SPIR-V headers, shader compiler |
| Build uses too much RAM | `export CMAKE_BUILD_PARALLEL_LEVEL=2` (or `set` on Windows) |
| GPU not detected | Falls back to 4GB CPU-safe tier; check `app/llm/gpu_detect.py` |
| Model init fails | Launcher tries smaller configs automatically; check VRAM, RAM, drivers |
| 401 on every request | `LLM_API_KEY` not set, or client not sending `X-API-Key` header |
| Context overflow | For `/v1/chat/completions`: set `contextLength` below server's `n_ctx`. For `/sessions`: check `app/sessions/eviction.py` |

---

## Architecture

```
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

---

## License

Model licenses are determined by their respective Hugging Face repositories.