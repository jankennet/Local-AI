"""
metrics.py

Prometheus metrics for the LLM server. Tracks:
- Request counts, latencies, token usage
- Tool call counts, latencies, success/failure
- Session counts, token budgets
- llama-server health
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Optional


# HTTP Request Metrics
http_requests_total = Counter(
    "llm_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

http_request_duration_seconds = Histogram(
    "llm_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

http_request_tokens_in = Counter(
    "llm_http_request_tokens_in_total",
    "Total input tokens from HTTP requests",
    ["path"],
)

http_request_tokens_out = Counter(
    "llm_http_request_tokens_out_total",
    "Total output tokens from HTTP requests",
    ["path"],
)

# Session Metrics
active_sessions = Gauge(
    "llm_active_sessions",
    "Number of currently active sessions",
)

session_tokens_used = Gauge(
    "llm_session_tokens_used",
    "Tokens used in a session",
    ["session_id"],
)

session_tokens_limit = Gauge(
    "llm_session_tokens_limit",
    "Token limit for a session",
    ["session_id"],
)

session_created_total = Counter(
    "llm_sessions_created_total",
    "Total sessions created",
)

session_deleted_total = Counter(
    "llm_sessions_deleted_total",
    "Total sessions deleted",
)

session_expired_total = Counter(
    "llm_sessions_expired_total",
    "Total sessions expired by TTL",
)

# Tool Call Metrics
tool_calls_total = Counter(
    "llm_tool_calls_total",
    "Total tool calls",
    ["tool_name", "status"],
)

tool_call_duration_seconds = Histogram(
    "llm_tool_call_duration_seconds",
    "Tool call latency in seconds",
    ["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

tool_call_retries_total = Counter(
    "llm_tool_call_retries_total",
    "Total tool call retries",
    ["tool_name"],
)

# Agent Loop Metrics
agent_rounds_total = Counter(
    "llm_agent_rounds_total",
    "Total agent loop rounds",
    ["session_id"],
)

agent_turns_total = Counter(
    "llm_agent_turns_total",
    "Total agent turns (user + assistant + tool)",
    ["session_id", "role"],
)

# LLM Completion Metrics
llm_completion_duration_seconds = Histogram(
    "llm_completion_duration_seconds",
    "LLM completion latency in seconds",
    ["model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

llm_completion_tokens_total = Counter(
    "llm_completion_tokens_total",
    "Total tokens from LLM completions",
    ["model", "type"],
)

llm_completion_errors_total = Counter(
    "llm_completion_errors_total",
    "Total LLM completion errors",
    ["model", "error_type"],
)

# llama-server Metrics
llama_server_health = Gauge(
    "llm_llama_server_health",
    "llama-server health status (1=healthy, 0=unhealthy)",
)

llama_server_restarts_total = Counter(
    "llm_llama_server_restarts_total",
    "Total llama-server restarts",
    ["reason"],
)

llama_server_n_ctx = Gauge(
    "llm_llama_server_n_ctx",
    "Current llama-server n_ctx (context window)",
)

llama_server_n_batch = Gauge(
    "llm_llama_server_n_batch",
    "Current llama-server n_batch",
)

# Vector Store Metrics
vector_store_operations_total = Counter(
    "llm_vector_store_operations_total",
    "Total vector store operations",
    ["operation", "backend", "status"],
)

vector_store_duration_seconds = Histogram(
    "llm_vector_store_duration_seconds",
    "Vector store operation latency in seconds",
    ["operation", "backend"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

vector_store_vectors_total = Gauge(
    "llm_vector_store_vectors_total",
    "Total vectors in store",
    ["backend"],
)

# System Info
server_info = Info(
    "llm_server_info",
    "Server information",
)


def init_server_info(model_path: str, n_ctx: int, n_batch: int, kv_cache: str) -> None:
    """Initialize static server info."""
    server_info.info({
        "model": model_path,
        "n_ctx": str(n_ctx),
        "n_batch": str(n_batch),
        "kv_cache": kv_cache,
    })


def update_llama_server_config(n_ctx: int, n_batch: int) -> None:
    """Update llama-server config metrics."""
    llama_server_n_ctx.set(n_ctx)
    llama_server_n_batch.set(n_batch)


def record_http_request(
    method: str, path: str, status_code: int, duration: float,
    tokens_in: Optional[int] = None, tokens_out: Optional[int] = None
) -> None:
    """Record HTTP request metrics."""
    http_requests_total.labels(method=method, path=path, status_code=str(status_code)).inc()
    http_request_duration_seconds.labels(method=method, path=path).observe(duration)
    if tokens_in is not None:
        http_request_tokens_in.labels(path=path).inc(tokens_in)
    if tokens_out is not None:
        http_request_tokens_out.labels(path=path).inc(tokens_out)


def record_tool_call(tool_name: str, duration: float, success: bool, retries: int = 0) -> None:
    """Record tool call metrics."""
    status = "success" if success else "failure"
    tool_calls_total.labels(tool_name=tool_name, status=status).inc()
    tool_call_duration_seconds.labels(tool_name=tool_name).observe(duration)
    if retries > 0:
        tool_call_retries_total.labels(tool_name=tool_name).inc(retries)


def record_agent_round(session_id: str) -> None:
    """Record an agent loop round."""
    agent_rounds_total.labels(session_id=session_id).inc()


def record_agent_turn(session_id: str, role: str) -> None:
    """Record an agent turn (user/assistant/tool)."""
    agent_turns_total.labels(session_id=session_id, role=role).inc()


def record_completion(
    model: str, duration: float, prompt_tokens: int, completion_tokens: int,
    error: Optional[str] = None
) -> None:
    """Record LLM completion metrics."""
    llm_completion_duration_seconds.labels(model=model).observe(duration)
    llm_completion_tokens_total.labels(model=model, type="prompt").inc(prompt_tokens)
    llm_completion_tokens_total.labels(model=model, type="completion").inc(completion_tokens)
    if error:
        llm_completion_errors_total.labels(model=model, error_type=error).inc()


def record_vector_operation(operation: str, backend: str, duration: float, success: bool) -> None:
    """Record vector store operation metrics."""
    status = "success" if success else "failure"
    vector_store_operations_total.labels(operation=operation, backend=backend, status=status).inc()
    vector_store_duration_seconds.labels(operation=operation, backend=backend).observe(duration)


def set_vector_count(backend: str, count: int) -> None:
    """Set vector store count."""
    vector_store_vectors_total.labels(backend=backend).set(count)


def set_active_sessions(count: int) -> None:
    """Set active sessions count."""
    active_sessions.set(count)


def set_session_tokens(session_id: str, used: int, limit: int) -> None:
    """Set session token usage."""
    session_tokens_used.labels(session_id=session_id).set(used)
    session_tokens_limit.labels(session_id=session_id).set(limit)


def remove_session_metrics(session_id: str) -> None:
    """Remove session metrics on deletion."""
    session_tokens_used.labels(session_id=session_id).set(0)
    session_tokens_limit.labels(session_id=session_id).set(0)


def record_session_created() -> None:
    """Record session creation."""
    session_created_total.inc()


def record_session_deleted() -> None:
    """Record session deletion."""
    session_deleted_total.inc()


def record_session_expired() -> None:
    """Record session expiration."""
    session_expired_total.inc()


def record_llama_restart(reason: str) -> None:
    """Record llama-server restart."""
    llama_server_restarts_total.labels(reason=reason).inc()


def set_llama_health(healthy: bool) -> None:
    """Set llama-server health status."""
    llama_server_health.set(1 if healthy else 0)