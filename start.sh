#!/bin/bash
set -e

if [ -z "$LLM_API_KEY" ]; then
    echo "LLM_API_KEY is not set. Set it first, e.g.:"
    echo '  export LLM_API_KEY="$(python -c '"'"'import secrets; print(secrets.token_urlsafe(32))'"'"')"'
    exit 1
fi

# Load .env if exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Start Qdrant server if configured for server mode
if [ "$LLM_QDRANT_SERVER_URL" != "" ] && [ "$LLM_VECTOR_DB_PATH" = "http://localhost:6333" ]; then
    echo "Starting Qdrant server (docker)..."
    if ! docker ps --format '{{.Names}}' | grep -q '^qdrant$'; then
        if docker ps -a --format '{{.Names}}' | grep -q '^qdrant$'; then
            docker start qdrant
        else
            docker run -d \
                --name qdrant \
                -p 6333:6333 \
                -v "$(pwd)/qdrant_db:/qdrant/storage" \
                qdrant/qdrant
        fi
        # Wait for Qdrant to be ready
        echo "Waiting for Qdrant to be ready..."
        for i in {1..10}; do
            if curl -s http://localhost:6333/healthz > /dev/null 2>&1; then
                echo "Qdrant is ready"
                break
            fi
            sleep 1
        done
    else
        echo "Qdrant already running"
    fi
fi

source venv/bin/activate
python run.py