# Test Suite

This directory contains the test suite for LocalAI, organized following TDD principles.

## Structure

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── pytest.ini            # Pytest configuration
├── README.md             # This file
├── unit/                 # Unit tests (fast, isolated)
│   ├── test_auth_cleanup_metrics.py
│   ├── test_config.py
│   ├── test_embeddings.py
│   ├── test_llm.py
│   ├── test_routes.py
│   ├── test_sessions.py
│   └── test_tokenizer.py
└── integration/          # Integration tests (slower, test full flows)
    └── test_full_flow.py
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with verbose output
pytest tests/ -v

# Run specific test
pytest tests/unit/test_sessions.py::TestSession::test_build_messages_with_rag -v
```

## Test Categories

### Unit Tests (tests/unit/)
Fast, isolated tests for individual components:
- **test_config.py** - Configuration loading and validation
- **test_embeddings.py** - Embedding service, vector stores, reranking
- **test_llm.py** - Model catalog, GPU detection, tools, agent loop
- **test_routes.py** - API route handlers (sessions, proxy, debug)
- **test_sessions.py** - Session store, eviction strategies, repository
- **test_tokenizer.py** - Token counting protocol
- **test_auth_cleanup_metrics.py** - Auth, cleanup, metrics

### Integration Tests (tests/integration/)
Full request flow tests:
- **test_full_flow.py** - Health/stats endpoints, RAG, tool flows, dynamic reserve

## Fixtures

Defined in `conftest.py`:
- `mock_tokenizer` - Deterministic token counter (4 chars = 1 token)
- `mock_embedding_service` - Deterministic embedding service with caching
- `temp_dir` - Temporary directory for test isolation
- `workspace_dir` - Workspace directory for file tools
- `sample_session_data` - Sample session data

## Test Isolation

Unit tests use mocks and don't require external services. Integration tests may require environment variables to be set (handled by fixtures).

## Configuration

Tests use environment variables for configuration. Key test env vars:
- `LLM_API_KEY` - Required for auth (set in conftest)
- `LLM_VECTOR_BACKEND=simple` - Use in-memory vector store
- `LLM_RERANKER_ENABLED=false` - Disable reranker for speed
- `LLM_EMBEDDING_MODEL` - Test embedding model

## Skipped Tests

One integration test is skipped due to config module caching issues when running the full test suite:
- `test_rag_retrieves_relevant_context` - Passes in isolation, fails in suite due to config module state leakage. Run with `pytest tests/integration/test_full_flow.py::TestSessionWithRAG::test_rag_retrieves_relevant_context -v` to verify.