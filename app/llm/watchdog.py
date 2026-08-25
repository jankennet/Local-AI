"""
watchdog.py

Background health check for the native llama-server subprocess. If it
crashes, relaunch it — first trying the exact config that already worked
(a crash is usually transient: a driver hiccup, a bad request, some other
process briefly stealing VRAM — not a sign the hardware suddenly can't
do what it just did), falling back to the full adaptive search only if
that immediate retry also fails.

Blocking subprocess/health-check calls run via asyncio.to_thread so a
relaunch attempt (which can take up to HEALTH_TIMEOUT_SECONDS) doesn't
freeze the rest of the app — session listing, etc. — while it's underway.
"""

import asyncio

from .server_launcher import launch_llama_server, relaunch_with_config
from ..metrics import record_llama_restart, update_llama_server_config

CHECK_INTERVAL_SECONDS = 15


async def watch_llama_server(
    binary: str, model_path: str, host: str, port: int, vram_tier: str,
    process_holder: dict, config_holder: dict,
) -> None:
    while True:
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)

        process = process_holder["process"]
        if process.poll() is None:
            continue  # still alive

        print("[watchdog] llama-server process died — relaunching with last-known config")
        try:
            result = await asyncio.to_thread(
                relaunch_with_config, binary, model_path, host, port, config_holder["config"]
            )
            if result is not None:
                process_holder["process"], _base_url = result
                record_llama_restart("crash_recovery_same_config")
                print("[watchdog] relaunch succeeded")
                continue

            print("[watchdog] relaunch with last-known config failed — retrying full adaptive search")
            process, _base_url, config = await asyncio.to_thread(
                launch_llama_server, binary, model_path, host, port, vram_tier
            )
            process_holder["process"] = process
            if config["n_ctx"] != config_holder["config"]["n_ctx"]:
                print(f"[watchdog] WARNING: recovered at a smaller n_ctx={config['n_ctx']} "
                      f"(was {config_holder['config']['n_ctx']}) — existing sessions' token "
                      f"budgets are now stale. Restart the whole app for full consistency.")
                record_llama_restart("config_downgrade")
            else:
                record_llama_restart("crash_recovery_new_config")
            config_holder["config"] = config
            update_llama_server_config(config["n_ctx"], config["n_batch"])
        except RuntimeError as error:
            print(f"[watchdog] recovery failed entirely: {error} — will retry on next check")
            record_llama_restart("recovery_failed")