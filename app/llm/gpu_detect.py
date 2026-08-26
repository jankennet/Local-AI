"""
gpu_detect.py

Hardware detection only. No model catalog, no server setup — just:
what GPU is this, how much VRAM does it have.
"""

import platform
import shutil
import subprocess


def detect_gpu():
    """Detects GPU Vendor, Model, and VRAM using standard system utilities."""
    vendor = "Unknown"
    gpu_name = "Generic / Integrated Graphics"
    vram_gb = 4  # baseline fallback

    if shutil.which("nvidia-smi"):
        try:
            cmd = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
            out = subprocess.check_output(cmd, text=True, errors="ignore").strip()
            if out:
                parts = [p.strip() for p in out.splitlines()[0].split(",")]
                if len(parts) >= 2:
                    return "NVIDIA", parts[0], round(float(parts[1]) / 1024)
        except Exception:
            pass

    if platform.system() == "Windows":
        try:
            ps_cmd = ("Get-CimInstance Win32_VideoController | "
                       "Select-Object Name, AdapterRAM | ConvertTo-Json")
            out = subprocess.check_output(["powershell", "-Command", ps_cmd],
                                           text=True, errors="ignore").strip()
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
                    if ram > 0 and vendor != "Unknown":
                        return vendor, name, max(round(ram / (1024 ** 3)), 2)
        except Exception:
            pass

    if platform.system() == "Linux":
        if shutil.which("rocm-smi"):
            try:
                subprocess.check_output(["rocm-smi", "--showid"], text=True, errors="ignore")
                return "AMD", "AMD ROCm Compatible GPU", 8
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


def get_vram_tier(vram_gb: int) -> str:
    if vram_gb <= 5:
        return "4GB"
    elif vram_gb <= 7:
        return "6GB"
    elif vram_gb <= 9:
        return "8GB"
    elif vram_gb <= 11:
        return "10GB"
    elif vram_gb <= 13:
        return "12GB"
    elif vram_gb <= 17:
        return "16GB"
    elif vram_gb <= 21:
        return "20GB"
    elif vram_gb <= 27:
        return "24GB"
    elif vram_gb <= 36:
        return "32GB"
    elif vram_gb <= 44:
        return "40GB"
    elif vram_gb <= 56:
        return "48GB"
    return "48GB"
