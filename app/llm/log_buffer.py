"""
log_buffer.py

Thread-safe rolling buffer for llama-server stdout/stderr.
Captures last N lines for error dumping. Also supports tee mode
to print logs to console in real-time.
"""

import sys
import threading
from collections import deque
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LogBuffer:
    """Thread-safe rolling buffer for log lines."""
    
    def __init__(self, max_lines: int = 100, tee: bool = False, output_stream=None):
        self._buffer = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self._max_lines = max_lines
        self._tee = tee
        self._output = output_stream or sys.stdout
    
    def add(self, line: str) -> None:
        """Add a line to the buffer, optionally printing to console."""
        with self._lock:
            self._buffer.append(line.rstrip("\n"))
        
        if self._tee:
            print(line, file=self._output, flush=True)
    
    def get_lines(self, n: Optional[int] = None) -> list[str]:
        """Get last N lines (or all if N is None)."""
        with self._lock:
            lines = list(self._buffer)
            if n is not None:
                return lines[-n:]
            return lines
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
    
    def dump(self, prefix: str = "LLAMA-SERVER LOG DUMP") -> str:
        """Format buffer contents as a visible warning block."""
        with self._lock:
            lines = list(self._buffer)
        
        if not lines:
            return f"\n{'='*60}\n{prefix}: (buffer empty)\n{'='*60}\n"
        
        output = [f"\n{'='*60}", f"!!! {prefix} ({len(lines)} lines) !!!", f"{'='*60}"]
        output.extend(lines)
        output.append(f"{'='*60}")
        output.append("END OF LOG DUMP")
        output.append(f"{'='*60}\n")
        return "\n".join(output)


# Global instance - created in main.py and passed around
_log_buffer: Optional[LogBuffer] = None


def get_log_buffer() -> Optional[LogBuffer]:
    """Get the global log buffer instance."""
    return _log_buffer


def set_log_buffer(buffer: LogBuffer) -> None:
    """Set the global log buffer instance."""
    global _log_buffer
    _log_buffer = buffer


def capture_stream(stream, buffer: LogBuffer, stream_name: str) -> None:
    """Background thread target: read lines from stream into buffer."""
    try:
        for line in iter(stream.readline, b""):
            try:
                decoded = line.decode("utf-8", errors="replace")
            except Exception:
                decoded = line.decode("latin-1", errors="replace")
            buffer.add(f"[{stream_name}] {decoded.rstrip()}")
    except Exception as e:
        logger.warning(f"Log capture thread for {stream_name} exited: {e}")