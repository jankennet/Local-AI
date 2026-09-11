#!/bin/bash
set -e

# Load .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$LLM_API_KEY" ]; then
    echo "LLM_API_KEY not set in .env"
    exit 1
fi

# Start server (foreground - you see logs)
source venv/bin/activate
python -m app.main