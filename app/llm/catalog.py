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
        "Llama 3.1 8B Instruct (Meta Flagship General AI - ~4.9GB)": {
            "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "size": "~4.9 GB"},
        "Qwen 2.5 Coder 7B (Q4_K_M - Best for Coding & VS Code - ~4.7GB)": {
            "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "size": "~4.7 GB"},
        "Qwen 2.5 7B (Q4_K_M - Top General Chat & Math - ~4.7GB)": {
            "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "filename": "qwen2.5-7b-instruct-q4_k_m.gguf", "size": "~4.7 GB"},
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
