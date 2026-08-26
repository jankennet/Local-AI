"""
catalog.py

Model catalog + download/selection logic. Kept separate from GPU
detection and from server-launching, so changing what models are
offered never touches how the server boots.
"""

import glob
import os
from huggingface_hub import hf_hub_download

MODEL_CATALOG = {
    "4GB": {
        "Qwen 2.5 3B Instruct (Best 3B General & Logic - ~2.0GB)": {
            "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "filename": "qwen2.5-3b-instruct-q4_k_m.gguf", "size": "~2.0 GB"},
        "Llama 3.2 3B Instruct (Meta Lightweight AI - ~2.0GB)": {
            "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "size": "~2.0 GB"},
        "Phi-3.5 Mini 3.8B (High Reasoning Lightweight - ~2.3GB)": {
            "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
            "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf", "size": "~2.3 GB"},
    },
    "6GB": {
        "Qwen 2.5 Coder 7B (Q3_K_M - Best for Coding - ~3.8GB)": {
            "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "filename": "qwen2.5-coder-7b-instruct-q3_k_m.gguf", "size": "~3.8 GB"},
        "Qwen 2.5 7B (Q3_K_M - General Chat & Logic - ~3.8GB)": {
            "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "filename": "qwen2.5-7b-instruct-q3_k_m.gguf", "size": "~3.8 GB"},
        "Llama 3.2 3B Instruct (Q8_0 - Maximum 3B Accuracy - ~3.4GB)": {
            "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "filename": "Llama-3.2-3B-Instruct-Q8_0.gguf", "size": "~3.4 GB"},
    },
    "8GB": {
        "Qwen 2.5 7B (Q4_K_M - Reliable Tool-Calling & General Chat - ~4.7GB)": {
            "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "size": "~4.7 GB"},
        "Llama 3.1 8B Instruct (Meta Flagship General AI - ~4.9GB)": {
            "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "size": "~4.9 GB"},
        "Qwen 2.5 Coder 7B (Q4_K_M - Strong at Code, Unreliable Tool-Calling - ~4.7GB)": {
            "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "size": "~4.7 GB"},
    },
    "10GB": {
        "Qwen 2.5 7B (Q6_K - Near-FP16 Quality - ~5.8GB)": {
            "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "filename": "Qwen2.5-7B-Instruct-Q6_K.gguf", "size": "~5.8 GB"},
        "Llama 3.1 8B Instruct (Q6_K - Near-FP16 Quality - ~6.1GB)": {
            "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf", "size": "~6.1 GB"},
        "Qwen 2.5 14B (Q3_K_M - 14B Model Fits - ~7.9GB)": {
            "repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF",
            "filename": "Qwen2.5-14B-Instruct-Q3_K_M.gguf", "size": "~7.9 GB"},
    },
    "12GB": {
        "Qwen 2.5 14B (Q4_K_M - Strong 14B General & Code - ~9.1GB)": {
            "repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF",
            "filename": "Qwen2.5-14B-Instruct-Q4_K_M.gguf", "size": "~9.1 GB"},
        "Llama 3.1 8B Instruct (Q8_0 - Maximum 8B Accuracy - ~8.3GB)": {
            "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", "size": "~8.3 GB"},
        "Nemotron 3 Ultra 8B (Q4_K_M - NVIDIA Reasoning Model - ~4.9GB)": {
            "repo_id": "bartowski/Nemotron-3-Ultra-8B-GGUF",
            "filename": "Nemotron-3-Ultra-8B-Q4_K_M.gguf", "size": "~4.9 GB"},
    },
    "16GB": {
        "Qwen 2.5 14B (Q6_K - Near-FP16 14B - ~11.2GB)": {
            "repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF",
            "filename": "Qwen2.5-14B-Instruct-Q6_K.gguf", "size": "~11.2 GB"},
        "Llama 3.1 8B Instruct (Q8_0 + Headroom - ~8.3GB)": {
            "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", "size": "~8.3 GB"},
        "Qwen 2.5 32B (Q3_K_M - 32B Model Fits - ~18.9GB)": {
            "repo_id": "bartowski/Qwen2.5-32B-Instruct-GGUF",
            "filename": "Qwen2.5-32B-Instruct-Q3_K_M.gguf", "size": "~18.9 GB"},
    },
    "20GB": {
        "Qwen 2.5 32B (Q4_K_M - Strong 32B Reasoning - ~18.9GB)": {
            "repo_id": "bartowski/Qwen2.5-32B-Instruct-GGUF",
            "filename": "Qwen2.5-32B-Instruct-Q4_K_M.gguf", "size": "~18.9 GB"},
        "Llama 3.1 70B (Q3_K_M - 70B Model Fits - ~19.5GB)": {
            "repo_id": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-70B-Instruct-Q3_K_M.gguf", "size": "~19.5 GB"},
        "Nemotron 3 Ultra 8B (Q8_0 - Best 8B Quality - ~8.3GB)": {
            "repo_id": "bartowski/Nemotron-3-Ultra-8B-GGUF",
            "filename": "Nemotron-3-Ultra-8B-Q8_0.gguf", "size": "~8.3 GB"},
    },
    "24GB": {
        "Qwen 2.5 32B (Q6_K - Near-FP16 32B - ~23GB)": {
            "repo_id": "bartowski/Qwen2.5-32B-Instruct-GGUF",
            "filename": "Qwen2.5-32B-Instruct-Q6_K.gguf", "size": "~23 GB"},
        "Llama 3.1 70B (Q4_K_M - Strong 70B General - ~23GB)": {
            "repo_id": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf", "size": "~23 GB"},
    },
    "32GB": {
        "Llama 3.1 70B (Q6_K - Near-FP16 70B - ~29GB)": {
            "repo_id": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-70B-Instruct-Q6_K.gguf", "size": "~29 GB"},
        "Qwen 2.5 32B (Q8_0 - Maximum 32B Quality - ~27GB)": {
            "repo_id": "bartowski/Qwen2.5-32B-Instruct-GGUF",
            "filename": "Qwen2.5-32B-Instruct-Q8_0.gguf", "size": "~27 GB"},
    },
    "40GB": {
        "Llama 3.1 70B (Q8_0 - Maximum 70B Quality - ~38GB)": {
            "repo_id": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-70B-Instruct-Q8_0.gguf", "size": "~38 GB"},
    },
    "48GB": {
        "Llama 3.1 70B (Q8_0 + KV Cache Headroom - ~38GB)": {
            "repo_id": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-70B-Instruct-Q8_0.gguf", "size": "~38 GB"},
    },
}


def get_existing_models(models_dir: str) -> list:
    os.makedirs(models_dir, exist_ok=True)
    return glob.glob(os.path.join(models_dir, "*.gguf"))


def download_model(models_dir: str, vram_tier: str, selection_name: str) -> str:
    tier_catalog = MODEL_CATALOG.get(vram_tier, MODEL_CATALOG["8GB"])
    model_info = tier_catalog[selection_name]
    return hf_hub_download(
        repo_id=model_info["repo_id"],
        filename=model_info["filename"],
        local_dir=models_dir,
    )