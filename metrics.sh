#!/bin/bash
# Quick metrics summary for LocalAI
curl -s http://127.0.0.1:8000/metrics | grep -E "^llm_(active_sessions|http_requests_total|llama_server_health|llama_server_n_ctx|sessions_created_total|tool_calls_total|completion_tokens_total)" | grep -v "_bucket\|_sum\|_count\|_created" | column -t -s ' '
