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
  - Structured extraction reduces token usage while preserving actionable info.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

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


@dataclass
class ToolResult:
    """Structured tool result for efficient storage and retrieval."""
    tool_name: str
    success: bool
    summary: str
    data: dict
    raw_output: str
    token_estimate: int


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _extract_structured_read_file(path: str, content: str) -> dict:
    """Extract key info from file read output."""
    lines = content.splitlines()
    return {
        "path": path,
        "line_count": len(lines),
        "char_count": len(content),
        "preview": lines[:3] if lines else [],
        "has_errors": any("error" in l.lower() or "exception" in l.lower() for l in lines[:20]),
    }


def _extract_structured_list_dir(path: str, content: str) -> dict:
    """Extract key info from directory listing."""
    items = [line.strip() for line in content.splitlines() if line.strip()]
    dirs = [i for i in items if not "." in i.split("/")[-1]]
    files = [i for i in items if "." in i.split("/")[-1]]
    return {
        "path": path,
        "total_items": len(items),
        "directories": dirs[:20],
        "files": files[:20],
        "file_extensions": list(set(f.split(".")[-1] for f in files if "." in f))[:10],
    }


def _extract_structured_run_bash(command: str, output: str, returncode: int) -> dict:
    """Extract key info from bash command output."""
    lines = output.splitlines()
    error_lines = [l for l in lines if any(p in l.lower() for p in ERROR_PATTERNS)]
    return {
        "command": command,
        "exit_code": returncode,
        "line_count": len(lines),
        "char_count": len(output),
        "errors": error_lines[:5],
        "has_output": bool(output.strip()),
        "last_lines": lines[-3:] if lines else [],
    }


def _build_structured_result(tool_name: str, raw_output: str, **kwargs) -> ToolResult:
    """Build a structured ToolResult from raw tool output."""
    success = kwargs.get("success", True)
    token_estimate = _estimate_tokens(raw_output)
    
    if tool_name == "read_file":
        data = _extract_structured_read_file(kwargs.get("path", ""), raw_output)
        summary = f"Read {data['line_count']} lines ({data['char_count']} chars) from {data['path']}"
        if data["has_errors"]:
            summary += " — contains errors/exceptions"
    elif tool_name == "list_dir":
        data = _extract_structured_list_dir(kwargs.get("path", "."), raw_output)
        summary = f"Listed {data['total_items']} items in {data['path']} ({len(data['directories'])} dirs, {len(data['files'])} files)"
    elif tool_name == "run_bash":
        data = _extract_structured_run_bash(kwargs.get("command", ""), raw_output, kwargs.get("returncode", 0))
        summary = f"Command exited with code {data['exit_code']}"
        if data["errors"]:
            summary += f" — {len(data['errors'])} error lines"
    else:
        data = {"raw": raw_output[:500]}
        summary = f"{tool_name}: {token_estimate} tokens"
    
    return ToolResult(
        tool_name=tool_name,
        success=success,
        summary=summary,
        data=data,
        raw_output=raw_output,
        token_estimate=token_estimate,
    )


def _format_structured_result(result: ToolResult, max_tokens: int) -> str:
    """Format structured result for model consumption, respecting token budget."""
    # Always include summary
    parts = [f"## {result.tool_name}: {result.summary}"]
    
    # Include structured data as JSON (compact)
    data_json = json.dumps(result.data, separators=(",", ":"))
    data_tokens = _estimate_tokens(data_json)
    
    if data_tokens <= max_tokens * 0.5:
        parts.append(f"```json\n{data_json}\n```")
    else:
        # Truncate data to fit
        parts.append(f"```json\n{data_json[:max_tokens * 2]}...[truncated]\n```")
    
    # Include raw output only if small enough and has errors
    if result.token_estimate <= max_tokens * 0.3 and not result.success:
        parts.append(f"Raw output:\n{result.raw_output[:max_tokens * 2]}")
    
    formatted = "\n".join(parts)
    if _estimate_tokens(formatted) > max_tokens:
        # Fallback to summary only
        return f"## {result.tool_name}: {result.summary}\n[Output truncated to fit budget]"
    
    return formatted


def _summarize_tool_output(text: str, max_tokens: int, tool_name: str) -> str:
    """
    Summarize tool output to fit within token budget.
    Uses heuristics to preserve errors, key results, and structure.
    """
    if not text:
        return text
    
    estimated_tokens = _estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text
    
    lines = text.splitlines()
    if not lines:
        return text[:max_tokens * 4]
    
    keep_first = 5
    keep_last = 10
    max_middle = 30
    
    kept = []
    kept.extend(lines[:keep_first])
    
    error_lines = []
    success_lines = []
    for i, line in enumerate(lines):
        if i < keep_first or i >= len(lines) - keep_last:
            continue
        line_lower = line.lower()
        if any(p in line_lower for p in ERROR_PATTERNS):
            error_lines.append((i, f">>> {line}"))
        elif any(p in line_lower for p in SUCCESS_PATTERNS):
            success_lines.append((i, f"✓ {line}"))
    
    for _, line in error_lines:
        kept.append(line)
    
    for _, line in success_lines:
        if len(kept) < keep_first + max_middle + keep_last:
            kept.append(line)
    
    middle_lines = lines[keep_first:-keep_last] if len(lines) > keep_first + keep_last else []
    for line in middle_lines:
        if len(kept) >= keep_first + max_middle + keep_last:
            break
        line_lower = line.lower()
        if any(p in line_lower for p in ERROR_PATTERNS) or any(p in line_lower for p in SUCCESS_PATTERNS):
            continue
        kept.append(line)
    
    if len(lines) > keep_first + keep_last:
        kept.extend(lines[-keep_last:])
    
    result = "\n".join(kept)
    
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


def _log_tool_call(name: str, args: dict, start_time: float, success: bool, error: str = None) -> None:
    duration = time.time() - start_time
    if success:
        logger.info(f"tool_call: {name} args={args} duration_ms={int(duration * 1000)}")
    else:
        logger.warning(f"tool_call_failed: {name} args={args} duration_ms={int(duration * 1000)} error={error}")


async def read_file(path: str) -> str:
    start = time.time()
    try:
        full = _resolve(path)
        if not os.path.isfile(full):
            _log_tool_call("read_file", {"path": path}, start, False, "file not found")
            return f"Error: no such file '{path}'"
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, lambda: open(full, "r", errors="replace").read())
        _log_tool_call("read_file", {"path": path}, start, True)
        content = _truncate(content)
        if settings.tool_output_summarize:
            structured = _build_structured_result("read_file", content, path=path, success=True)
            content = _format_structured_result(structured, settings.tool_output_max_tokens)
        return content
    except Exception as e:
        _log_tool_call("read_file", {"path": path}, start, False, str(e))
        return f"Error reading file: {type(e).__name__}: {e}"


async def write_file(path: str, content: str) -> str:
    start = time.time()
    try:
        full = _resolve(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: open(full, "w").write(content))
        result = f"Wrote {len(content)} chars to {path}"
        _log_tool_call("write_file", {"path": path, "chars": len(content)}, start, True)
        return result
    except Exception as e:
        _log_tool_call("write_file", {"path": path, "chars": len(content)}, start, False, str(e))
        return f"Error writing file: {type(e).__name__}: {e}"


async def list_dir(path: str = ".") -> str:
    start = time.time()
    try:
        full = _resolve(path)
        if not os.path.isdir(full):
            _log_tool_call("list_dir", {"path": path}, start, False, "directory not found")
            return f"Error: no such directory '{path}'"
        loop = asyncio.get_event_loop()
        items = await loop.run_in_executor(None, lambda: sorted(os.listdir(full)))
        result = "\n".join(items) or "(empty)"
        _log_tool_call("list_dir", {"path": path}, start, True)
        result = _truncate(result)
        if settings.tool_output_summarize:
            structured = _build_structured_result("list_dir", result, path=path, success=True)
            result = _format_structured_result(structured, settings.tool_output_max_tokens)
        return result
    except Exception as e:
        _log_tool_call("list_dir", {"path": path}, start, False, str(e))
        return f"Error listing directory: {type(e).__name__}: {e}"


async def run_bash(command: str) -> str:
    start = time.time()
    if os.environ.get("LLM_ALLOW_SHELL") != "1":
        _log_tool_call("run_bash", {"command": command}, start, False, "shell disabled")
        return "Error: shell execution is disabled (set LLM_ALLOW_SHELL=1 to enable)"
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                command, shell=True, cwd=WORKSPACE_DIR,
                capture_output=True, text=True, timeout=30,
            ),
        )
        output = (result.stdout + result.stderr) or "(no output)"
        success = result.returncode == 0
        _log_tool_call("run_bash", {"command": command}, start, success, None if success else f"exit_code={result.returncode}")
        output = _truncate(output)
        if settings.tool_output_summarize:
            structured = _build_structured_result("run_bash", output, command=command, returncode=result.returncode, success=success)
            output = _format_structured_result(structured, settings.tool_output_max_tokens)
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