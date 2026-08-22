"""
tools.py

Tiny tool registry: name -> {"schema": <OpenAI tool schema>, "fn": callable}.
Four tools, not a plugin framework — add a fifth by adding a dict entry,
not by inventing a Tool base class or auto-discovery.

Safety, not optional:
  - File tools are confined to WORKSPACE_DIR — no path traversal out of it.
  - run_bash is off by default (LLM_ALLOW_SHELL=1 to enable). This server
    is LAN-reachable and a shell tool the model can call unsupervised on
    any request is a real hole, not a hypothetical one — see README.
  - All tool output is truncated before it goes into session history, so
    a huge file read/command output doesn't blow the token budget or
    bloat sessions.json before eviction ever gets a chance to run.
"""

import os
import subprocess

WORKSPACE_DIR = os.path.abspath(os.environ.get("LLM_WORKSPACE_DIR", "workspace"))
os.makedirs(WORKSPACE_DIR, exist_ok=True)

MAX_OUTPUT_CHARS = 8000


def _resolve(path: str) -> str:
    full = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if full != WORKSPACE_DIR and not full.startswith(WORKSPACE_DIR + os.sep):
        raise ValueError(f"path '{path}' escapes the workspace")
    return full


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n...[truncated, {len(text)} chars total]"
    return text


def read_file(path: str) -> str:
    full = _resolve(path)
    if not os.path.isfile(full):
        return f"Error: no such file '{path}'"
    with open(full, "r", errors="replace") as f:
        return _truncate(f.read())


def write_file(path: str, content: str) -> str:
    full = _resolve(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}"


def list_dir(path: str = ".") -> str:
    full = _resolve(path)
    if not os.path.isdir(full):
        return f"Error: no such directory '{path}'"
    return "\n".join(sorted(os.listdir(full))) or "(empty)"


def run_bash(command: str) -> str:
    if os.environ.get("LLM_ALLOW_SHELL") != "1":
        return "Error: shell execution is disabled (set LLM_ALLOW_SHELL=1 to enable)"
    try:
        result = subprocess.run(
            command, shell=True, cwd=WORKSPACE_DIR,
            capture_output=True, text=True, timeout=30,
        )
        return _truncate(result.stdout + result.stderr) or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30s"


TOOLS = {
    "read_file": {
        "fn": read_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file's contents, path relative to the workspace directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    },
    "write_file": {
        "fn": write_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write text content to a file, path relative to the workspace directory. Overwrites if it already exists.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
    },
    "list_dir": {
        "fn": list_dir,
        "schema": {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List files in a directory, path relative to the workspace directory. Defaults to the workspace root.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        },
    },
    "run_bash": {
        "fn": run_bash,
        "schema": {
            "type": "function",
            "function": {
                "name": "run_bash",
                "description": "Run a shell command in the workspace directory. Disabled unless LLM_ALLOW_SHELL=1 is set on the server.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    },
}