#!/bin/bash
set -e

if [ -z "$LLM_API_KEY" ]; then
    echo "LLM_API_KEY is not set. Set it first, e.g.:"
    echo '  export LLM_API_KEY="$(python -c '"'"'import secrets; print(secrets.token_urlsafe(32))'"'"')"'
    exit 1
fi

source venv/bin/activate
python run.py