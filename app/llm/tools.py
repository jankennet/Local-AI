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
  - Tool results exceeding token budget are summarized to key information.
"""

import logging
import os
import subprocess
import time

from ..config import settings

logger = logging.getLogger(__name__)

WORKSPACE_DIR = os.path.abspath(os.environ.get("LLM_WORKSPACE_DIR", "workspace"))
os.makedirs(WORKSPACE_DIR, exist_ok=True)

MAX_OUTPUT_CHARS = 8000

# Common patterns that indicate important content to keep
ERROR_PATTERNS = [
    "error", "exception", "traceback", "failed", "failure",
    "warning", "warn:", "err:", "fatal", "critical",
    "not found", "no such", "permission denied", "timeout",
]
SUCCESS_PATTERNS = [
    "success", "completed", "finished", "done", "ok",
    "created", "wrote", "updated", "deleted",
]


def _summarize_tool_output(text: str, max_tokens: int, tool_name: str) -> str:
    """
    Summarize tool output to fit within token budget.
    Uses heuristics to preserve errors, key results, and structure.
    """
    if not text:
        return text
    
    # Rough token estimate: ~4 chars per token
    estimated_tokens = len(text) // 4
    if estimated_tokens <= max_tokens:
        return text
    
    lines = text.splitlines()
    if not lines:
        return text[:max_tokens * 4]
    
    # Always keep first few lines (context/header)
    # Always keep last few lines (result/footer)
    # For middle, keep lines matching error/success patterns
    keep_first = 5
    keep_last = 10
    max_middle = 30
    
    kept = []
    kept.extend(lines[:keep_first])
    
    # Collect error/success lines from entire text (not just middle)
    # This ensures errors are never lost regardless of position
    error_lines = []
    success_lines = []
    for i, line in enumerate(lines):
        if i < keep_first or i >= len(lines) - keep_last:
            continue  # Already handled by first/last
        line_lower = line.lower()
        if any(p in line_lower for p in ERROR_PATTERNS):
            error_lines.append((i, f">>> {line}"))
        elif any(p in line_lower for p in SUCCESS_PATTERNS):
            success_lines.append((i, f"✓ {line}"))
    
    # Add error lines first (most important)
    for _, line in error_lines:
        kept.append(line)
    
    # Add success lines if space permits
    for _, line in success_lines:
        if len(kept) < keep_first + max_middle + keep_last:
            kept.append(line)
    
    # Fill remaining middle space with regular lines
    middle_lines = lines[keep_first:-keep_last] if len(lines) > keep_first + keep_last else []
    for line in middle_lines:
        if len(kept) >= keep_first + max_middle + keep_last:
            break
        line_lower = line.lower()
        # Skip if already added as error/success
        if any(p in line_lower for p in ERROR_PATTERNS) or any(p in line_lower for p in SUCCESS_PATTERNS):
            continue
        kept.append(line)
    
    if len(lines) > keep_first + keep_last:
        kept.extend(lines[-keep_last:])
    
    result = "\n".join(kept)
    
    # Final truncation if still too long
    if len(result) > max_tokens * 4:
        result = result[:max_tokens * 4] + f"\n...[summarized, {estimated_tokens} tokens → ~{max_tokens}]"
    
    return result


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n...[truncated, {len(text)} chars total]"
    return text


def _resolve(path: str) -> str:
    full = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if full != WORKSPACE_DIR and not full.startswith(WORKSPACE_DIR + os.sep):
        raise ValueError(f"path '{path}' escapes the workspace")
    return full


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n...[truncated, {len(text)} chars total]"
    return text


def _log_tool_call(name: str, args: dict, start_time: float, success: bool, error: str = None) -> None:
    duration = time.time() - start_time
    if success:
        logger.info(f"tool_call: {name} args={args} duration_ms={int(duration * 1000)}")
    else:
        logger.warning(f"tool_call_failed: {name} args={args} duration_ms={int(duration * 1000)} error={error}")


def read_file(path: str) -> str:
    start = time.time()
    try:
        full = _resolve(path)
        if not os.path.isfile(full):
            _log_tool_call("read_file", {"path": path}, start, False, "file not found")
            return f"Error: no such file '{path}'"
        with open(full, "r", errors="replace") as f:
            content = f.read()
        _log_tool_call("read_file", {"path": path}, start, True)
        content = _truncate(content)
        if settings.tool_output_summarize:
            content = _summarize_tool_output(content, settings.tool_output_max_tokens, "read_file")
        return content
    except Exception as e:
        _log_tool_call("read_file", {"path": path}, start, False, str(e))
        return f"Error reading file: {type(e).__name__}: {e}"


def write_file(path: str, content: str) -> str:
    start = time.time()
    try:
        full = _resolve(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        result = f"Wrote {len(content)} chars to {path}"
        _log_tool_call("write_file", {"path": path, "chars": len(content)}, start, True)
        return result
    except Exception as e:
        _log_tool_call("write_file", {"path": path, "chars": len(content)}, start, False, str(e))
        return f"Error writing file: {type(e).__name__}: {e}"


def list_dir(path: str = ".") -> str:
    start = time.time()
    try:
        full = _resolve(path)
        if not os.path.isdir(full):
            _log_tool_call("list_dir", {"path": path}, start, False, "directory not found")
            return f"Error: no such directory '{path}'"
        result = "\n".join(sorted(os.listdir(full))) or "(empty)"
        _log_tool_call("list_dir", {"path": path}, start, True)
        result = _truncate(result)
        if settings.tool_output_summarize:
            result = _summarize_tool_output(result, settings.tool_output_max_tokens, "list_dir")
        return result
    except Exception as e:
        _log_tool_call("list_dir", {"path": path}, start, False, str(e))
        return f"Error listing directory: {type(e).__name__}: {e}"


def run_bash(command: str) -> str:
    start = time.time()
    if os.environ.get("LLM_ALLOW_SHELL") != "1":
        _log_tool_call("run_bash", {"command": command}, start, False, "shell disabled")
        return "Error: shell execution is disabled (set LLM_ALLOW_SHELL=1 to enable)"
    try:
        result = subprocess.run(
            command, shell=True, cwd=WORKSPACE_DIR,
            capture_output=True, text=True, timeout=30,
        )
        output = (result.stdout + result.stderr) or "(no output)"
        success = result.returncode == 0
        _log_tool_call("run_bash", {"command": command}, start, success, None if success else f"exit_code={result.returncode}")
        output = _truncate(output)
        if settings.tool_output_summarize:
            output = _summarize_tool_output(output, settings.tool_output_max_tokens, "run_bash")
        return output
    except subprocess.TimeoutExpired:
        _log_tool_call("run_bash", {"command": command}, start, False, "timeout")
        return "Error: command timed out after 30s"
    except Exception as e:
        _log_tool_call("run_bash", {"command": command}, start, False, str(e))
        return f"Error running command: {type(e).__name__}: {e}"


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