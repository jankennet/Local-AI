"""
server_launcher.py

Owns exactly one job: given a model path and a VRAM tier, launch the
native llama-server binary with adaptive n_ctx/n_batch configs until one
comes up healthy, and hand back the running process + base URL + config.

This runs llama.cpp's own C++ server (not llama-cpp-python's bundled
Python one) because it's the one that actually reads each GGUF's own
embedded chat template for tool-calling (--jinja) — generalizing across
model families, instead of requiring a hand-picked chat_format string
per model the way the Python server's registry does.

Nothing here knows about sessions, tokenizers, or auth — those get
layered on top in main.py. The caller owns the returned process and is
responsible for terminating it on shutdown.
"""

import subprocess
import time

import requests

PERFORMANCE_TIERS = {
    "4GB": [{"n_ctx": 8192, "n_batch": 128},
            {"n_ctx": 4096, "n_batch": 128},
            {"n_ctx": 4096, "n_batch": 64}],
    "6GB": [{"n_ctx": 12288, "n_batch": 256},
            {"n_ctx": 8192, "n_batch": 256},
            {"n_ctx": 8192, "n_batch": 128},
            {"n_ctx": 4096, "n_batch": 128}],
    "8GB": [{"n_ctx": 16384, "n_batch": 256},
            {"n_ctx": 12288, "n_batch": 256},
            {"n_ctx": 8192, "n_batch": 256},
            {"n_ctx": 8192, "n_batch": 128},
            {"n_ctx": 4096, "n_batch": 128}],
}

HEALTH_TIMEOUT_SECONDS = 90  # model loading can be slow, especially on CPU fallback
HEALTH_POLL_INTERVAL = 1.0


def get_adaptive_configs(vram_tier: str) -> list:
    """Each base n_ctx/n_batch pair is tried twice: first with a quantized
    (q8_0) KV cache — roughly halves the VRAM the cache itself uses — then
    with the plain f16 cache if that fails to come up healthy. Quantized
    KV requires flash attention, which not every GPU/build supports, so
    the plain variant is the safety net, not a second-class option."""
    configs = []
    for base in PERFORMANCE_TIERS.get(vram_tier, PERFORMANCE_TIERS["4GB"]):
        configs.append({**base, "kv_quant": True})
        configs.append({**base, "kv_quant": False})
    return configs


def _build_command(binary: str, model_path: str, host: str, port: int, config: dict) -> list:
    cmd = [
        binary,
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--n-gpu-layers", "99",  # large-enough-to-mean-"all" is llama-server's own convention
        "--ctx-size", str(config["n_ctx"]),
        "-b", str(config["n_batch"]),
        # Renders the GGUF's own embedded chat template (via llama.cpp's
        # built-in Jinja engine) instead of a hardcoded format — this is
        # what makes tool-calling generalize across model families.
        "--jinja",
    ]
    if config["kv_quant"]:
        cmd += ["--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    return cmd


def _wait_until_healthy(base_url: str, process: subprocess.Popen, deadline: float) -> bool:
    while time.time() < deadline:
        if process.poll() is not None:
            return False  # process exited early — this config failed to load
        try:
            if requests.get(f"{base_url}/health", timeout=2).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    return False


def terminate_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def relaunch_with_config(binary: str, model_path: str, host: str, port: int, config: dict):
    """Try relaunching with the exact config that already worked once —
    used by the watchdog to recover from a crash without re-running the
    whole adaptive search. Returns (process, base_url) or None on failure."""
    cmd = _build_command(binary, model_path, host, port, config)
    process = subprocess.Popen(cmd)
    base_url = f"http://{host}:{port}"

    if _wait_until_healthy(base_url, process, time.time() + HEALTH_TIMEOUT_SECONDS):
        return process, base_url

    terminate_process(process)
    return None


def launch_llama_server(binary: str, model_path: str, host: str, port: int, vram_tier: str):
    """Returns (process, base_url, selected_config). Raises RuntimeError if
    every config in the tier fails to come up healthy."""
    base_url = f"http://{host}:{port}"

    for config in get_adaptive_configs(vram_tier):
        cmd = _build_command(binary, model_path, host, port, config)
        process = subprocess.Popen(cmd)  # inherits our stdout/stderr — same log visibility as before

        if _wait_until_healthy(base_url, process, time.time() + HEALTH_TIMEOUT_SECONDS):
            return process, base_url, config

        print(f"Config {config} failed to come up healthy — killing and trying next")
        terminate_process(process)

    raise RuntimeError(f"No working configuration found for tier {vram_tier}")