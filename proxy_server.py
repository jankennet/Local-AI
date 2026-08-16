import os
import sys
import glob
import shutil
import platform
import subprocess
import questionary
from huggingface_hub import hf_hub_download

MODELS_DIR = "models"

# Catalog organized into 4GB, 6GB, and 8GB target VRAM tiers (Top 3 recommendations each)
MODEL_CATALOG = {
    "4GB": {
        "Qwen 2.5 3B Instruct (Best 3B General & Logic - ~2.0GB)": {
            "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
            "size": "~2.0 GB"
        },
        "Llama 3.2 3B Instruct (Meta Lightweight AI - ~2.0GB)": {
            "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "size": "~2.0 GB"
        },
        "Phi-3.5 Mini 3.8B (High Reasoning Lightweight - ~2.3GB)": {
            "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
            "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
            "size": "~2.3 GB"
        }
    },
    "6GB": {
        "Qwen 2.5 Coder 7B (Q3_K_M - Best for Coding - ~3.8GB)": {
            "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "filename": "qwen2.5-coder-7b-instruct-q3_k_m.gguf",
            "size": "~3.8 GB"
        },
        "Qwen 2.5 7B (Q3_K_M - General Chat & Logic - ~3.8GB)": {
            "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "filename": "qwen2.5-7b-instruct-q3_k_m.gguf",
            "size": "~3.8 GB"
        },
        "Llama 3.2 3B Instruct (Q8_0 - Maximum 3B Accuracy - ~3.4GB)": {
            "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "filename": "Llama-3.2-3B-Instruct-Q8_0.gguf",
            "size": "~3.4 GB"
        }
    },
    "8GB": {
        "Llama 3.1 8B Instruct (Meta Flagship General AI - ~4.9GB)": {
            "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "size": "~4.9 GB"
        },
        "Qwen 2.5 Coder 7B (Q4_K_M - Best for Coding & VS Code - ~4.7GB)": {
            "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            "size": "~4.7 GB"
        },
        "Qwen 2.5 7B (Q4_K_M - Top General Chat & Math - ~4.7GB)": {
            "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
            "size": "~4.7 GB"
        }
    }
}


def detect_gpu():
    """Detects GPU Vendor, Model, and VRAM using standard system utilities without extra dependencies."""
    vendor = "Unknown"
    gpu_name = "Generic / Integrated Graphics"
    vram_gb = 4  # Default baseline fallback

    # 1. Detect NVIDIA GPU via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            cmd = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
            out = subprocess.check_output(cmd, text=True, errors="ignore").strip()
            if out:
                first_gpu = out.splitlines()[0]
                parts = [p.strip() for p in first_gpu.split(",")]
                if len(parts) >= 2:
                    gpu_name = parts[0]
                    vram_mb = float(parts[1])
                    vram_gb = round(vram_mb / 1024)
                    vendor = "NVIDIA"
                    return vendor, gpu_name, vram_gb
        except Exception:
            pass

    # 2. Detect AMD / General GPU on Windows via PowerShell
    if platform.system() == "Windows":
        try:
            ps_cmd = (
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name, AdapterRAM | "
                "ConvertTo-Json"
            )
            out = subprocess.check_output(["powershell", "-Command", ps_cmd], text=True, errors="ignore").strip()
            if out:
                import json
                data = json.loads(out)
                if isinstance(data, dict):
                    data = [data]

                for card in data:
                    name = card.get("Name", "")
                    ram = card.get("AdapterRAM", 0) or 0
                    if "NVIDIA" in name.upper():
                        vendor = "NVIDIA"
                    elif "AMD" in name.upper() or "RADEON" in name.upper():
                        vendor = "AMD"

                    if ram > 0:
                        vram_gb = round(ram / (1024**3))
                        gpu_name = name
                        if vendor != "Unknown":
                            return vendor, gpu_name, max(vram_gb, 2)
        except Exception:
            pass

    # 3. Detect AMD GPU on Linux via rocm-smi or lspci
    if platform.system() == "Linux":
        if shutil.which("rocm-smi"):
            try:
                out = subprocess.check_output(["rocm-smi", "--showid"], text=True, errors="ignore")
                if out:
                    vendor = "AMD"
                    gpu_name = "AMD ROCm Compatible GPU"
                    vram_gb = 8
                    return vendor, gpu_name, vram_gb
            except Exception:
                pass

        if shutil.which("lspci"):
            try:
                out = subprocess.check_output(["lspci"], text=True, errors="ignore")
                for line in out.splitlines():
                    if "VGA" in line or "3D" in line:
                        if any(k in line for k in ["AMD", "Radeon", "Advanced Micro Devices"]):
                            return "AMD", line.split(":")[-1].strip(), 8
                        elif "NVIDIA" in line:
                            return "NVIDIA", line.split(":")[-1].strip(), 8
            except Exception:
                pass

    return vendor, gpu_name, vram_gb


def get_vram_tier(vram_gb):
    """Maps numerical VRAM to target tier string ('4GB', '6GB', or '8GB')."""
    if vram_gb <= 5:
        return "4GB"
    elif vram_gb <= 7:
        return "6GB"
    else:
        return "8GB"


def confirm_gpu_settings():
    """Runs auto-detection, prompts the user to confirm detection, or lets them manually override."""
    detected_vendor, detected_name, detected_vram = detect_gpu()
    detected_tier = get_vram_tier(detected_vram)

    print("=" * 65)
    print("                 GPU & VRAM DETECTED")
    print("=" * 65)
    print(f"  • Vendor        : {detected_vendor}")
    print(f"  • Model         : {detected_name}")
    print(f"  • VRAM Detected : ~{detected_vram} GB (Target Tier: {detected_tier})")
    print("=" * 65)

    # Prompt user for confirmation
    is_correct = questionary.confirm(
        f"Is this GPU hardware correct? ({detected_vendor} - {detected_name} ~{detected_vram}GB)",
        default=True
    ).ask()

    if is_correct:
        vendor = detected_vendor
        tier = detected_tier
    else:
        print("\nManual Override Selected:")
        vendor = questionary.select(
            "Select your GPU Vendor:",
            choices=["NVIDIA", "AMD", "Intel / Integrated / CPU"]
        ).ask()

        tier = questionary.select(
            "Select your VRAM Capacity Tier:",
            choices=["4GB", "6GB", "8GB"]
        ).ask()

    # Inform user about setup differences
    print("\n" + "-" * 65)
    if vendor == "NVIDIA":
        print("💡 [NVIDIA Detected/Selected]: Minimal setup required!")
        print("   Using standard CUDA drivers. Ensure CUDA runtime is active.")
    elif vendor == "AMD":
        print("💡 [AMD Detected/Selected]: Uses Vulkan / ROCm runtime.")
        print("   Ensure llama-cpp-python is compiled with Vulkan support (GGML_VULKAN=1).")
    else:
        print("💡 [CPU / Fallback Selected]: Running on System RAM.")
    print("-" * 65 + "\n")

    return vendor, tier


def get_existing_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    return glob.glob(os.path.join(MODELS_DIR, "*.gguf"))


def select_and_download_model(vram_tier):
    tier_catalog = MODEL_CATALOG.get(vram_tier, MODEL_CATALOG["8GB"])
    choices = list(tier_catalog.keys()) + ["Cancel / Exit"]

    selection = questionary.select(
        f"Select an AI model optimized for your [{vram_tier}] tier:",
        choices=choices
    ).ask()

    if not selection or selection == "Cancel / Exit":
        print("Download cancelled.")
        return None

    model_info = tier_catalog[selection]
    print(f"\nDownloading {selection} from Hugging Face...")
    print(f"Estimated Download Size: {model_info['size']}\n")

    downloaded_path = hf_hub_download(
        repo_id=model_info["repo_id"],
        filename=model_info["filename"],
        local_dir=MODELS_DIR
    )

    print(f"\nDownload complete: {downloaded_path}\n")
    return downloaded_path


def main():
    # 1. Detect and confirm GPU vendor & VRAM tier
    vendor, vram_tier = confirm_gpu_settings()

    existing_models = get_existing_models()
    model_path = None

    if not existing_models:
        print(f"No local .gguf models found in '{MODELS_DIR}/'.\n")
        model_path = select_and_download_model(vram_tier)
        if not model_path:
            print("No model selected. Exiting.")
            sys.exit(0)
    else:
        current_model_name = os.path.basename(existing_models[0])
        action = questionary.select(
            f"Existing model found ({current_model_name}). What would you like to do?",
            choices=[
                f"Run existing model ({current_model_name})",
                f"Download a new model (Tailored for {vram_tier} GPU)",
                "Cancel / Exit"
            ]
        ).ask()

        if action and action.startswith("Download a new model"):
            model_path = select_and_download_model(vram_tier)
            if not model_path:
                model_path = existing_models[0]
        elif action == "Cancel / Exit" or not action:
            sys.exit(0)
        else:
            model_path = existing_models[0]

    # 2. Launch llama-cpp-python API server
    from llama_cpp.server.app import create_app
    from llama_cpp.server.settings import ModelSettings, ServerSettings
    import uvicorn

    print(f"\nStarting API Server with model: {os.path.basename(model_path)}")

    model_settings = ModelSettings(
        model=model_path,
        n_gpu_layers=-1,  # 100% offload to GPU
        n_ctx=4096,
        n_batch=512,
        verbose=True
    )

    server_settings = ServerSettings(host="127.0.0.1", port=8000)

    app = create_app(server_settings=server_settings, model_settings=[model_settings])
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()