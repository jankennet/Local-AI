import os
import sys
import glob
from huggingface_hub import hf_hub_download
import questionary

MODELS_DIR = "models"

# Hugging Face catalog for Vulkan-compatible GGUF models
MODEL_CATALOG = {
    "Qwen 2.5 Coder 7B (Best for VS Code / Auto-Complete)": {
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    },
    "Qwen 2.5 7B (General Chat & Logic)": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf"
    },
    "Llama 3.1 8B Instruct (Meta General AI)": {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    },
    "Phi-3.5 Mini 3.8B (Lightweight & Fast)": {
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf"
    }
}


def get_existing_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    return glob.glob(os.path.join(MODELS_DIR, "*.gguf"))


def select_and_download_model():
    choices = list(MODEL_CATALOG.keys()) + ["Cancel / Exit"]

    # Interactive Arrow Key Menu
    selection = questionary.select(
        "Select an AI model to download for your RX 560 XT:",
        choices=choices
    ).ask()

    if not selection or selection == "Cancel / Exit":
        print("Download cancelled.")
        return None

    model_info = MODEL_CATALOG[selection]
    print(f"\nDownloading {selection} from Hugging Face...")
    print("Please wait, downloading ~2.5GB to ~4.5GB GGUF file...\n")

    downloaded_path = hf_hub_download(
        repo_id=model_info["repo_id"],
        filename=model_info["filename"],
        local_dir=MODELS_DIR
    )

    print(f"\nDownload complete: {downloaded_path}\n")
    return downloaded_path


def main():
    existing_models = get_existing_models()
    model_path = None

    if not existing_models:
        print("No local .gguf models found in 'models/' folder.\n")
        model_path = select_and_download_model()
        if not model_path:
            print("No model available to run. Exiting.")
            sys.exit(0)
    else:
        # Prompt user whether to run existing model or download a new one
        current_model_name = os.path.basename(existing_models[0])
        action = questionary.select(
            f"Existing model found ({current_model_name}). What would you like to do?",
            choices=[
                f"Run existing model ({current_model_name})",
                "Download a new model",
                "Cancel / Exit"
            ]
        ).ask()

        if action == "Download a new model":
            model_path = select_and_download_model()
            if not model_path:
                model_path = existing_models[0]
        elif action == "Cancel / Exit" or not action:
            sys.exit(0)
        else:
            model_path = existing_models[0]

    # Import llama-cpp and start the Vulkan API Server
    from llama_cpp.server.app import create_app
    from llama_cpp.server.settings import ModelSettings, ServerSettings
    import uvicorn

    print(f"\nStarting Vulkan Server with model: {os.path.basename(model_path)}")

    model_settings = ModelSettings(
        model=model_path,
        n_gpu_layers=-1,  # 100% offloaded to RX 560 XT 8GB
        n_ctx=4096,
        n_batch=512,
        verbose=True
    )

    server_settings = ServerSettings(host="127.0.0.1", port=8000)

    app = create_app(server_settings=server_settings, model_settings=[model_settings])
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()