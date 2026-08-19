"""
server_launcher.py

Owns exactly one job: given a model path and a VRAM tier, try adaptive
n_ctx/n_batch configs until one loads, and hand back the underlying
llama_cpp.server FastAPI app + whichever config actually worked.

Nothing here knows about sessions, tokenizers, or auth — those get
layered on top in main.py.
"""

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


def get_adaptive_configs(vram_tier: str) -> list:
    return PERFORMANCE_TIERS.get(vram_tier, PERFORMANCE_TIERS["4GB"])


def build_llama_app(model_path: str, vram_tier: str):
    """Returns (app, selected_config). Raises RuntimeError if every
    config in the tier fails to load."""
    from llama_cpp.server.app import create_app
    from llama_cpp.server.settings import ModelSettings, ServerSettings

    server_settings = ServerSettings(host="0.0.0.0", port=8000)

    for config in get_adaptive_configs(vram_tier):
        try:
            model_settings = ModelSettings(
                model=model_path,
                n_gpu_layers=-1,
                n_ctx=config["n_ctx"],
                n_batch=config["n_batch"],
                verbose=True,
            )
            app = create_app(server_settings=server_settings, model_settings=[model_settings])
            return app, config
        except Exception as error:
            print(f"Config {config} failed: {type(error).__name__}: {error}")
            continue

    raise RuntimeError(f"No working configuration found for tier {vram_tier}")
